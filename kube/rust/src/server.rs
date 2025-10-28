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
use std::sync::Arc;

use anyhow::Result;
use axum::{extract::State, http::StatusCode, Json};
use k8s_openapi::api::{apps::v1::Deployment, core::v1::Pod};
use kube::{Api, Client};
use serde::{Deserialize, Serialize};

use crate::{scaler::scale_deployment, settings::AppSettings};

#[derive(Clone)]
pub struct AppState {
    pub client: Client,
    pub cfg: AppSettings,
}

impl AppState {
    pub fn new(client: Client, cfg: AppSettings) -> Self {
        Self { client, cfg }
    }
}

#[derive(Deserialize)]
pub struct ScaleRequest { pub app: String, pub replicas: i32 }

#[derive(Serialize)]
pub struct ScaleResponse { pub deployment: String, pub replicas: i32 }

pub async fn scale_pool(State(state): State<Arc<AppState>>, Json(req): Json<ScaleRequest>) -> Result<(StatusCode, Json<ScaleResponse>), (StatusCode, String)> {
    let deploy = req.app.clone();
    tracing::info!(deployment=%deploy, replicas=req.replicas, namespace=%state.cfg.namespace, "received scale request");
    match scale_deployment(state.client.clone(), &state.cfg.namespace, &deploy, req.replicas).await {
        Ok(()) => {
            tracing::info!(deployment=%deploy, replicas=req.replicas, "scale request succeeded");
            Ok((StatusCode::OK, Json(ScaleResponse { deployment: deploy, replicas: req.replicas })))
        }
        Err(e) => {
            tracing::error!(deployment=%deploy, replicas=req.replicas, error=?e, "scale request failed");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

#[derive(Serialize)]
pub struct PodSummary {
    pub name: String,
    pub phase: Option<String>,
    pub node_name: Option<String>,
    pub host_ip: Option<String>,
    pub pod_ip: Option<String>,
}

#[derive(Serialize)]
pub struct PodsResponse { pub items: Vec<PodSummary> }

pub async fn list_pods(State(state): State<Arc<AppState>>) -> Result<Json<PodsResponse>, (StatusCode, String)> {
    tracing::debug!(namespace=%state.cfg.namespace, "list pods request");
    let api: Api<Pod> = Api::namespaced(state.client.clone(), &state.cfg.namespace);
    let pods = api
        .list(&Default::default())
        .await
        .map_err(|e| {
            tracing::error!(error=?e, "failed to list pods");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;
    let items: Vec<PodSummary> = pods
        .into_iter()
        .map(|p| PodSummary {
            name: p.metadata.name.unwrap_or_default(),
            phase: p.status.as_ref().and_then(|s| s.phase.clone()),
            node_name: p.spec.as_ref().and_then(|s| s.node_name.clone()),
            host_ip: p.status.as_ref().and_then(|s| s.host_ip.clone()),
            pod_ip: p.status.as_ref().and_then(|s| s.pod_ip.clone()),
        })
        .collect::<Vec<PodSummary>>();
    tracing::info!(count=items.len(), "list pods succeeded");
    Ok(Json(PodsResponse { items }))
}

#[derive(Serialize)]
pub struct DeploymentSummary { pub name: String, pub replicas: Option<i32>, pub available: Option<i32> }

#[derive(Serialize)]
pub struct DeploymentsResponse { pub items: Vec<DeploymentSummary> }

pub async fn list_deployments(State(state): State<Arc<AppState>>) -> Result<Json<DeploymentsResponse>, (StatusCode, String)> {
    tracing::debug!(namespace=%state.cfg.namespace, "list deployments request");
    let api: Api<Deployment> = Api::namespaced(state.client.clone(), &state.cfg.namespace);
    let list = api.list(&Default::default()).await.map_err(|e| {
        tracing::error!(error=?e, "failed to list deployments");
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;
    let items: Vec<DeploymentSummary> = list.into_iter().map(|d| {
        DeploymentSummary {
            name: d.metadata.name.unwrap_or_default(),
            replicas: d.spec.as_ref().and_then(|s| s.replicas),
            available: d.status.as_ref().and_then(|s| s.available_replicas),
        }
    }).collect::<Vec<DeploymentSummary>>();
    tracing::info!(count=items.len(), "list deployments succeeded");
    Ok(Json(DeploymentsResponse { items }))
}

