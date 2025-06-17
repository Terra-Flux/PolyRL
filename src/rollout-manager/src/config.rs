use anyhow::Result;
use clap::Parser;
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
}

pub async fn load_config(args: &Args) -> Result<Config> {
    let mut config = if let Some(config_file) = &args.config_file {
        match tokio::fs::read_to_string(config_file).await {
            Ok(contents) => {
                match toml::from_str::<Config>(&contents) {
                    Ok(file_config) => {
                        log::info!("Loaded config from file: {:?}", config_file);
                        file_config
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
        // Initialize with defaults for P2P handshake
        Config {
            mooncake_transfer_device_name: String::new(),
            mooncake_transfer_protocol: "tcp".to_string(),
            weight_sender_rpyc_endpoints: vec![],
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
        config.weight_sender_rpyc_endpoints = endpoints.clone();
    }
    
    // For P2P handshake, we don't need to validate mooncake_metadata_server
    // as it will be set to "p2phandshake" by default
    
    Ok(config)
}