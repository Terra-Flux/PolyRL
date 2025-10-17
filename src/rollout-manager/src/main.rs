mod models;
mod state;
mod config;
mod handlers;
mod instance_manager;
mod balance;
mod utils;

use std::net::SocketAddr;
use anyhow::Result;
use axum::{
    routing::{get, post, put},
    Router,
};
use clap::Parser;

use crate::config::{Args, load_config};
use crate::state::AppState;
use crate::handlers::*;

// compiler automatically use physical core number
#[tokio::main(flavor="multi_thread")]
async fn main() -> Result<()> {
    // Initialize logger with custom format and info level as default
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .format(|buf, record| {
            use std::io::Write;
            writeln!(
                buf,
                "[{} {} {}:{}] {}",
                buf.timestamp(),
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .init();
    
    let args = Args::parse();
    let config = load_config(&args).await?;
    
    log::info!("Starting rollout manager with config:");
    log::info!("  mooncake_transfer_device_name: {}", config.mooncake_transfer_device_name);
    log::info!("  mooncake_transfer_protocol: {}", config.mooncake_transfer_protocol);
    log::info!("  weight_sender_rpyc_endpoints: {:?}", config.weight_sender_rpyc_endpoints);
    log::info!("  bind_addr: {}", args.bind_addr);
    
    let state = AppState::new(config);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/get_instances_status", get(get_instances_status))
        .route("/register_rollout_instance", post(register_rollout_instance))
        .route("/register_local_rollout_instances", post(register_local_rollout_instances))
        .route("/generate", post(generate_request))
        .route("/batch_generate_requests", post(timed_batch_generate_requests))
        .route("/update_weight_version", post(update_weight_version))
        .route("/get_receive_instances", post(get_receive_instances))
        .route("/update_weights", post(update_weights))
        .route("/update_weight_senders", put(update_weight_senders))
        .route("/shutdown_instances", post(shutdown_instances_handler))
        .route("/update_metrics", post(update_metrics))
        .route("/abort_local_requests", post(abort_local_requests))
        .with_state(state);

    let addr: SocketAddr = args.bind_addr.parse()?;
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    log::info!("Rollout manager listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}