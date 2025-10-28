// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use anyhow::Result;
use axum::{routing::{get, post}, Router};
use kube::Client;
use std::{net::SocketAddr, sync::Arc};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod server;
mod watcher;
mod settings;
mod scaler;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cfg = {
        // Basic, lightweight CLI arg parse for optional --config <path|name>
        let mut args = std::env::args().skip(1);
        let mut config_arg: Option<String> = None;
        while let Some(a) = args.next() {
            if a == "--config" {
                if let Some(v) = args.next() { config_arg = Some(v); }
            }
        }
        settings::AppSettings::from_config_arg(config_arg)?
    };
    let client = Client::try_default().await?;
    let shared = Arc::new(server::AppState::new(client.clone(), cfg.clone()));

    // spawn watcher task
    let state_for_watcher = shared.clone();
    tokio::spawn(async move {
        if let Err(err) = watcher::run_pod_monitor(state_for_watcher).await {
            tracing::error!(error = ?err, "pod monitor terminated with error");
        }
    });

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/pods", get(server::list_pods))
        .route("/deployments", get(server::list_deployments))
        .route("/scale", post(server::scale_pool))
        .layer(TraceLayer::new_for_http())
        .with_state(shared);

    // Optionally apply deployment manifests on startup if configured
    for path in &cfg.deployment_files {
        tracing::info!(file=%path.display(), "applying deployment manifest from config");
        if let Err(e) = scaler::apply_yaml_manifest(client.clone(), &cfg.namespace, path).await {
            tracing::error!(file=%path.display(), error=?e, "failed to apply deployment manifest");
        }
    }

    let addr: SocketAddr = cfg.bind_addr.parse()?;
    tracing::info!(%addr, "starting server");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}
