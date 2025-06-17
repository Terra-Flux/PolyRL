use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering, AtomicBool};
use dashmap::{DashMap, DashSet};
use tokio::sync::{RwLock, Notify};
use crate::models::{Instance, Config, InstanceInfo};

#[derive(Clone)]
pub struct AppState {
    // Using DashMap for concurrent access without RwLock
    pub instances: Arc<DashMap<String, Instance>>, // endpoint -> Instance
    pub pending_instances: Arc<DashSet<String>>, // endpoints currently being health checked
    pub pending_shutdown: Arc<DashSet<String>>, // endpoints to be removed in next prepare round
    pub rr_counter: Arc<AtomicUsize>,
    pub config: Arc<RwLock<Config>>,
    pub weight_sender_counter: Arc<AtomicUsize>,
    pub client: reqwest::Client,
    
    // For coordinating weight updates with instance registration
    pub instance_registration_notify: Arc<Notify>,
    
    // Track which instances are used for current generation and weight updates (unified)
    pub active_instances: Arc<RwLock<Vec<Instance>>>,
    
    // Track weight sender assignments
    pub instance_weight_sender_map: Arc<DashMap<String, String>>, // instance_endpoint -> weight_sender_endpoint
    
    // For waiting on weight senders
    pub weight_sender_notify: Arc<Notify>,
    
    // Track prepare_weight_update round state
    pub instances_frozen_for_round: Arc<AtomicBool>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        Self {
            instances: Arc::new(DashMap::new()),
            pending_instances: Arc::new(DashSet::new()),
            pending_shutdown: Arc::new(DashSet::new()),
            rr_counter: Arc::new(AtomicUsize::new(0)),
            config: Arc::new(RwLock::new(config)),
            weight_sender_counter: Arc::new(AtomicUsize::new(0)),
            client: reqwest::Client::builder()
                .tcp_keepalive(std::time::Duration::from_secs(30))
                .timeout(std::time::Duration::from_secs(3000))
                .build()
                .expect("failed building reqwest client"),
            instance_registration_notify: Arc::new(Notify::new()),
            active_instances: Arc::new(RwLock::new(Vec::new())),
            instance_weight_sender_map: Arc::new(DashMap::new()),
            weight_sender_notify: Arc::new(Notify::new()),
            instances_frozen_for_round: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn next_instance(&self) -> Option<Instance> {
        let instances = self.active_instances.read().await;
        if instances.is_empty() {
            return None;
        }
        let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed) % instances.len();
        Some(instances[idx].clone())
    }
    
    pub async fn get_next_weight_sender(&self) -> String {
        let config = self.config.read().await;
        if config.weight_sender_rpyc_endpoints.is_empty() {
            panic!("No weight sender endpoints available");
        }
        let idx = self.weight_sender_counter.fetch_add(1, Ordering::Relaxed) % config.weight_sender_rpyc_endpoints.len();
        config.weight_sender_rpyc_endpoints[idx].clone()
    }
    
    pub async fn wait_for_weight_senders(&self) {
        loop {
            let config = self.config.read().await;
            if !config.weight_sender_rpyc_endpoints.is_empty() {
                break;
            }
            drop(config);
            
            // Wait for notification of weight sender update
            self.weight_sender_notify.notified().await;
        }
    }
    
    pub async fn get_all_instances(&self) -> Vec<Instance> {
        self.instances.iter()
            .map(|entry| entry.value().clone())
            .collect()
    }
    
    pub async fn add_instance_after_health_check(&self, instance: Instance) {
        // Check for duplicates
        if self.instances.contains_key(&instance.endpoint) {
            log::info!("Instance {} already exists, skipping duplicate", instance.endpoint);
            return;
        }
        
        // Assign a weight sender to this instance
        let weight_sender = self.get_next_weight_sender().await;
        self.instance_weight_sender_map.insert(instance.endpoint.clone(), weight_sender.clone());
        
        log::info!("Instance {} is now ready and added to instances list (id: {}) with weight sender {}", 
                   instance.endpoint, instance.id, weight_sender);
        self.instances.insert(instance.endpoint.clone(), instance.clone());
        
        // Remove from pending list
        self.pending_instances.remove(&instance.endpoint);
        
        // Notify waiters that a new instance has been registered
        self.instance_registration_notify.notify_waiters();
    }
    
    pub fn remove_from_pending(&self, endpoint: &str) {
        self.pending_instances.remove(endpoint);
    }
    
    pub fn is_pending(&self, endpoint: &str) -> bool {
        self.pending_instances.contains(endpoint)
    }
    
    pub fn add_to_pending(&self, endpoint: String) {
        self.pending_instances.insert(endpoint);
    }

    pub fn shutdown_instances(&self, endpoints: Vec<String>) {
        for endpoint in endpoints {
            self.pending_shutdown.insert(endpoint.clone());
            log::info!("Instance {} marked for shutdown in next prepare round", endpoint);
        }
    }

    pub async fn prepare_instances_for_weight_update_round(&self, weight_sender_endpoint: String) -> Result<Vec<InstanceInfo>, String> {
        use tokio::time::{timeout, Duration};
        
        // Check if instances are already frozen for this round
        if !self.instances_frozen_for_round.load(Ordering::Acquire) {
            // First caller - need to freeze instances for the round
            log::info!("First prepare_weight_update call - freezing instances for round");
            
            // Wait for at least one instance to be registered using async notify
            let wait_result = timeout(Duration::from_secs(120), async {
                loop {
                    let instances = self.get_all_instances().await;
                    if !instances.is_empty() {
                        return instances;
                    }
                    log::info!("Waiting for rollout instances to register...");
                    
                    // Wait for notification of new instance registration
                    self.instance_registration_notify.notified().await;
                }
            }).await;
            
            let mut all_instances = match wait_result {
                Ok(instances) => instances,
                Err(_) => {
                    log::warn!("Timeout waiting for rollout instances to register");
                    return Err("timeout waiting for rollout instances to register".to_string());
                }
            };
            
            // Remove instances that are pending shutdown
            let shutdown_endpoints: Vec<String> = self.pending_shutdown.iter()
                .map(|entry| entry.key().clone())
                .collect();
            
            for endpoint in &shutdown_endpoints {
                all_instances.retain(|instance| instance.endpoint != *endpoint);
                // Also remove from the main instances map
                self.instances.remove(endpoint);
                // Remove from weight sender map
                self.instance_weight_sender_map.remove(endpoint);
                log::info!("Removed instance {} from active instances due to shutdown", endpoint);
            }
            
            // Clear the pending shutdown list
            self.pending_shutdown.clear();
            
            // Set the fixed instances for weight update round
            {
                let mut active_instances = self.active_instances.write().await;
                *active_instances = all_instances;
            }
            
            // Mark instances as frozen for this round
            self.instances_frozen_for_round.store(true, Ordering::Release);
            
            log::info!("Instances frozen for weight update round");
        } else {
            log::info!("Using already frozen instances for weight update round");
        }
        
        // Get the frozen instances
        let frozen_instances = self.active_instances.read().await.clone();
        
        // Filter instances by weight sender
        let instance_infos = frozen_instances.into_iter()
            .filter_map(|instance| {
                if let Some(ws) = self.instance_weight_sender_map.get(&instance.endpoint) {
                    if ws.value() == &weight_sender_endpoint {
                        Some(InstanceInfo {
                            id: instance.id,
                            endpoint: instance.endpoint.clone(),
                            weight_sender_endpoint: ws.value().clone(),
                            mooncake_handshake_endpoint: instance.mooncake_handshake_endpoint.clone(),
                        })
                    } else {
                        None
                    }
                } else {
                    log::error!("Instance {} does not have a weight sender assigned", instance.endpoint);
                    None
                }
            })
            .collect();
        
        Ok(instance_infos)
    }
    
    pub fn mark_weight_update_round_complete(&self) {
        // Mark the current round as complete and allow next round to freeze new instances
        self.instances_frozen_for_round.store(false, Ordering::Release);
        log::info!("Weight update round marked as complete");
    }


}