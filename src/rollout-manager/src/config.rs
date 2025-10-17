use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use crate::models::Config;

#[derive(Parser, Debug)]
#[command(name = "rollout-manager")]
#[command(about = "Rollout manager for managing SGLang instances", long_about = None)]
pub struct Args {
    #[arg(long, value_name = "DEVICE", default_value = "")]
    pub mooncake_transfer_device_name: Option<String>,
    
    #[arg(long, value_name = "PROTOCOL", default_value = "tcp")]
    pub mooncake_transfer_protocol: Option<String>,
    
    #[arg(long, value_name = "ENDPOINTS", num_args = 1..)]
    pub weight_sender_rpyc_endpoints: Option<Vec<String>>,
    
    #[arg(long, value_name = "FILE")]
    pub config_file: Option<String>,
    
    #[arg(long, default_value = "0.0.0.0:5000")]
    pub bind_addr: String,
    
    #[arg(long, value_name = "COUNT", default_value = "10")]
    pub max_pending_requests_per_instance: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TempConfig {
    pub mooncake_transfer_device_name: String,
    pub mooncake_transfer_protocol: String,
    pub weight_sender_rpyc_endpoints: Vec<String>,
    pub max_pending_requests_per_instance: usize,
}

pub async fn load_config(args: &Args) -> Result<Config> {
    let mut config = if let Some(config_file) = &args.config_file {
        match tokio::fs::read_to_string(config_file).await {
            Ok(contents) => {
                match toml::from_str::<TempConfig>(&contents) {
                    Ok(temp_config) => {
                        log::info!("Loaded config from file: {:?}", config_file);
                        let mut parsed_endpoints = Vec::new();
                        for endpoint in &temp_config.weight_sender_rpyc_endpoints {
                            match endpoint.parse::<SocketAddr>() {
                                Ok(addr) => parsed_endpoints.push(addr),
                                Err(e) => {
                                    log::error!("Failed to parse weight sender endpoint '{}' from config file: {}", endpoint, e);
                                    return Err(anyhow::anyhow!("Failed to parse weight sender endpoint '{}' from config file: {}", endpoint, e));
                                }
                            }
                        }
                        Config {
                            mooncake_transfer_device_name: temp_config.mooncake_transfer_device_name,
                            mooncake_transfer_protocol: temp_config.mooncake_transfer_protocol,
                            weight_sender_rpyc_endpoints: parsed_endpoints,
                            num_mooncake_groups: 1,
                            num_mooncake_engines_per_group: 1,
                            max_pending_requests_per_instance: temp_config.max_pending_requests_per_instance,
                            train_batch_size: None,
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to parse config file: {}", e);
                        return Err(anyhow::anyhow!("Failed to parse config file: {}", e));
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to read config file: {}", e);
                return Err(anyhow::anyhow!("Failed to read config file: {}", e));
            }
        }
    } else {
        Config {
            mooncake_transfer_device_name: String::new(),
            mooncake_transfer_protocol: "tcp".to_string(),
            weight_sender_rpyc_endpoints: vec![],
            num_mooncake_groups: 1,
            num_mooncake_engines_per_group: 1,
            max_pending_requests_per_instance: 10,
            train_batch_size: None,
        }
    };
    
    // Override with command line arguments
    if let Some(device) = &args.mooncake_transfer_device_name {
        config.mooncake_transfer_device_name = device.clone();
    }
    if let Some(protocol) = &args.mooncake_transfer_protocol {
        config.mooncake_transfer_protocol = protocol.clone();
    }
    if let Some(endpoints) = &args.weight_sender_rpyc_endpoints {
        let mut parsed_endpoints = Vec::new();
        for endpoint in endpoints {
            match endpoint.parse::<SocketAddr>() {
                Ok(addr) => parsed_endpoints.push(addr),
                Err(e) => {
                    log::error!("Failed to parse weight sender endpoint '{}': {}", endpoint, e);
                    return Err(anyhow::anyhow!("Failed to parse weight sender endpoint '{}': {}", endpoint, e));
                }
            }
        }
        config.weight_sender_rpyc_endpoints = parsed_endpoints;
    }
    if let Some(max_pending) = args.max_pending_requests_per_instance {
        config.max_pending_requests_per_instance = max_pending;
    }
    

    
    Ok(config)
}