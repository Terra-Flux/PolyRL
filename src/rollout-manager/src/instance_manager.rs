use tokio::time::Duration;
use crate::models::Instance;
use crate::state::AppState;

pub async fn health_check_instance(state: AppState, instance: Instance) {
    let endpoint = instance.endpoint();
    log::info!("Starting health check for instance: {}", endpoint);
    
    // Maximum wait time: 5 minutes
    let max_wait_time = Duration::from_secs(300);
    let check_interval = Duration::from_secs(2);
    let start_time = std::time::Instant::now();
    
    while start_time.elapsed() < max_wait_time {
        let health_url = format!("{}/health_generate", endpoint);
        
        match state.client.get(&health_url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    log::info!("Health check passed for instance: {}", endpoint);
                    state.add_instance_after_health_check(instance).await;
                    return;
                } else {
                    log::debug!("Health check failed for instance: {} (status: {})", endpoint, response.status());
                }
            }
            Err(e) => {
                log::debug!("Health check request failed for instance: {} (error: {})", endpoint, e);
            }
        }
        
        tokio::time::sleep(check_interval).await;
    }
    
    log::warn!("Health check timed out for instance: {} after {} seconds", endpoint, max_wait_time.as_secs());
    state.remove_from_pending(&instance.addr);
}