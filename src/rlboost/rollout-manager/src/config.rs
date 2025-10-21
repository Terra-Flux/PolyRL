use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use crate::models::Config;

// Config override order: command line > config file > default

#[derive(Parser, Debug)]
#[command(name = "rollout-manager")]
#[command(about = "Rollout manager for managing SGLang instances", long_about = None)]
pub struct Args {
    #[arg(long, value_name = "DEVICE")]
    pub mooncake_transfer_device_name: Option<String>,
    
    #[arg(long, value_name = "PROTOCOL")]
    pub mooncake_transfer_protocol: Option<String>,
    
    #[arg(long, value_name = "ENDPOINTS", num_args = 1..)]
    pub weight_sender_rpyc_endpoints: Option<Vec<String>>,
    
    #[arg(long, value_name = "FILE")]
    pub config_file: Option<String>,
    
    #[arg(long, default_value = "0.0.0.0:5000")]
    pub bind_addr: String,
    
    #[arg(long, value_name = "COUNT")]
    pub max_assigned_batches_per_stats_check: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TempConfig {
    pub mooncake_transfer_device_name: String,
    pub mooncake_transfer_protocol: String,
    pub allowed_sender_ips: String,
    pub weight_sender_rpyc_endpoints: Vec<String>,
    pub weight_sender_rpyc_base_port: usize,
    pub num_mooncake_groups_per_sender: usize,
    pub num_mooncake_engines_per_group: usize,
    pub max_assigned_batches_per_stats_check: usize,
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
                            allowed_sender_ips: temp_config.allowed_sender_ips,
                            weight_sender_rpyc_endpoints: parsed_endpoints,
                            weight_sender_rpyc_base_port: temp_config.weight_sender_rpyc_base_port,
                            num_mooncake_groups_per_sender: temp_config.num_mooncake_groups_per_sender,
                            num_mooncake_engines_per_group: temp_config.num_mooncake_engines_per_group,
                            max_assigned_batches_per_stats_check: temp_config.max_assigned_batches_per_stats_check,
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
            allowed_sender_ips: "0.0.0.0/0".to_string(),
            weight_sender_rpyc_endpoints: vec![],
            weight_sender_rpyc_base_port: 18861,
            num_mooncake_groups_per_sender: 1,
            num_mooncake_engines_per_group: 1,
            max_assigned_batches_per_stats_check: 4,
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
    if let Some(max_assigned_batches_per_stats_check) = args.max_assigned_batches_per_stats_check {
        config.max_assigned_batches_per_stats_check = max_assigned_batches_per_stats_check;
    }
    

    
    Ok(config)
}