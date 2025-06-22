use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};

use uuid::Uuid;
use tokio::time::{sleep, Duration};

use crate::models::*;
use crate::state::AppState;
use crate::instance_manager;

pub async fn register_rollout_instance(
    State(state): State<AppState>,
    Json(payload): Json<RegisterRequest>,
) -> impl IntoResponse {
    // Wait for weight senders if none are available
    state.wait_for_weight_senders().await;
    
    let mooncake_handshake_endpoint = payload.mooncake_handshake_port
        .map(|port| format!("{}:{}", payload.host, port));
        
    let instance = Instance {
        id: Uuid::new_v4(),
        endpoint: format!("http://{}:{}", payload.host, payload.port),
        mooncake_handshake_endpoint,
    };

    // Check if this instance is already registered (ready) or pending
    if state.instances.contains_key(&instance.endpoint) {
        log::info!("Instance already registered and ready: {}", instance.endpoint);
        let config = state.config.read().await;
        let weight_sender = state.get_next_weight_sender().await;
        let resp = RegisterResponse {
            mooncake_transfer_device_name: config.mooncake_transfer_device_name.clone(),
            mooncake_transfer_protocol: config.mooncake_transfer_protocol.clone(),
            weight_sender_rpyc_endpoint: weight_sender,
        };
        return (StatusCode::OK, Json(resp));
    }

    // Check if this instance is already pending health check
    if state.is_pending(&instance.endpoint) {
        log::info!("Instance already pending health check: {}", instance.endpoint);
        let config = state.config.read().await;
        let weight_sender = state.get_next_weight_sender().await;
        let resp = RegisterResponse {
            mooncake_transfer_device_name: config.mooncake_transfer_device_name.clone(),
            mooncake_transfer_protocol: config.mooncake_transfer_protocol.clone(),
            weight_sender_rpyc_endpoint: weight_sender,
        };
        return (StatusCode::OK, Json(resp));
    }

    // Add to pending list and start health check
    state.add_to_pending(instance.endpoint.clone());
    log::info!("Received registration request for instance: {} (id: {})", instance.endpoint, instance.id);

    // Spawn health check task
    let state_clone = state.clone();
    let instance_clone = instance.clone();
    tokio::spawn(async move {
        instance_manager::health_check_instance(state_clone, instance_clone).await;
    });

    let config = state.config.read().await;
    let weight_sender = state.get_next_weight_sender().await;
    
    // Store the weight sender assignment
    state.instance_weight_sender_map.insert(instance.endpoint.clone(), weight_sender.clone());
    
    let resp = RegisterResponse {
        mooncake_transfer_device_name: config.mooncake_transfer_device_name.clone(),
        mooncake_transfer_protocol: config.mooncake_transfer_protocol.clone(),
        weight_sender_rpyc_endpoint: weight_sender,
    };
    (StatusCode::OK, Json(resp))
}

pub async fn update_weight_senders(
    State(state): State<AppState>,
    Json(payload): Json<UpdateWeightSendersRequest>,
) -> impl IntoResponse {
    {
        let mut config = state.config.write().await;
        config.weight_sender_rpyc_endpoints = payload.weight_sender_rpyc_endpoints.clone();
        log::info!("Updated weight sender endpoints: {:?}", config.weight_sender_rpyc_endpoints);
    }
    
    // Notify any waiters that weight senders are now available
    state.weight_sender_notify.notify_waiters();
    
    (StatusCode::OK, Json(serde_json::json!({"success": true})))
}

pub async fn generate_request(
    State(state): State<AppState>,
    Json(body): Json<GenerateRequest>,
) -> impl IntoResponse {
    
    let Some(instance) = state.next_instance().await else {
        log::warn!("No rollout instances registered for generate request");
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": "no rollout instances registered"})));
    };

    log::debug!("Forwarding generate request to instance: {}", instance.endpoint);
    let url = format!("{}/generate", instance.endpoint);
    
    const MAX_RETRIES: u32 = 3;
    let mut retry_count = 0;
    
    loop {
        log::debug!("Attempting request to {} (attempt {})", url, retry_count + 1);
        match state.client.post(&url).json(&body.0).send().await {
            Ok(res) => {
                let status = res.status();
                let headers = res.headers().clone();
                match res.json::<serde_json::Value>().await {
                    Ok(json) => {
                        log::debug!("Successfully forwarded generate request to {}", instance.endpoint);
                        return (status, Json(json));
                    }
                    Err(e) => {
                        log::error!("Failed to parse response from {}: {}. Status code: {}, Response headers: {:?}", 
                            instance.endpoint, e, status, headers);
                        return (
                            StatusCode::BAD_GATEWAY,
                            Json(serde_json::json!({"error": format!("failed to parse response: {}", e)})),
                        );
                    }
                }
            }
            Err(e) => {
                // More detailed error analysis
                let error_details = if e.is_timeout() {
                    "timeout"
                } else if e.is_connect() {
                    "connection_error"
                } else if e.is_request() {
                    "request_error"
                } else if e.is_body() {
                    "body_error"
                } else if e.is_decode() {
                    "decode_error"
                } else {
                    "unknown_error"
                };
                
                retry_count += 1;
                if retry_count >= MAX_RETRIES {
                    log::error!("Request to {} failed after {} retries: {} (type: {})", 
                        instance.endpoint, MAX_RETRIES, e, error_details);
                    return (
                        StatusCode::BAD_GATEWAY,
                        Json(serde_json::json!({"error": format!("request failed after {} retries: {} ({})", MAX_RETRIES, e, error_details)})),
                    );
                }
                let backoff = Duration::from_millis(100 * 2u64.pow(retry_count - 1));
                log::warn!("Request to {} failed (attempt {}/{}, type: {}), retrying in {:?}: {}", 
                    instance.endpoint, retry_count, MAX_RETRIES, error_details, backoff, e);
                sleep(backoff).await;
            }
        }
    }
}

pub async fn prepare_weight_update(
    State(state): State<AppState>,
    Json(payload): Json<PrepareWeightUpdateRequest>,
) -> impl IntoResponse {
    // Use the new round-based prepare logic
    match state.prepare_instances_for_weight_update_round(payload.weight_sender_endpoint).await {
        Ok(instance_infos) => {
            let response = PrepareWeightUpdateResponse {
                current_instances: instance_infos,
            };
            
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(error_msg) => {
            log::warn!("Failed to prepare weight update: {}", error_msg);
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": error_msg })),
            ).into_response();
        }
    }
}

pub async fn update_weights_from_agent_handler(
    State(state): State<AppState>,
    Json(payload): Json<UpdateWeightsFromAgentRequest>,
) -> impl IntoResponse {
    // Use the active instances for weight update
    let instances = state.active_instances.read().await.clone();
    
    if instances.is_empty() {
        log::warn!("No instances were prepared for weight update. Call prepare_weight_update first.");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "no instances prepared for weight update. Call prepare_weight_update first." })),
        );
    }

    log::info!("Updating weights for {} prepared instances", instances.len());
    let mut results = Vec::new();
    let mut all_successful = true;

    for instance in instances {
        log::debug!("Updating weights for instance: {}", instance.endpoint);
        let url = format!("{}/update_weights_from_agent", instance.endpoint);
        match state.client.post(&url).json(&payload).send().await {
            Ok(res) => {
                let status = res.status();
                match res.json::<serde_json::Value>().await {
                    Ok(json_response) => {
                        results.push(serde_json::json!({
                            "instance_id": instance.id,
                            "endpoint": instance.endpoint,
                            "status_code": status.as_u16(),
                            "response": json_response
                        }));
                        if !status.is_success() || json_response.get("success").and_then(|s| s.as_bool()) != Some(true) {
                            all_successful = false;
                            log::warn!("Weight update failed for instance {}: status={}, response={:?}", 
                                instance.endpoint, status, json_response);
                        } else {
                            log::debug!("Weight update successful for instance {}", instance.endpoint);
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to parse response from instance {}: {}", instance.endpoint, e);
                        results.push(serde_json::json!({
                            "instance_id": instance.id,
                            "endpoint": instance.endpoint,
                            "status_code": StatusCode::BAD_GATEWAY.as_u16(),
                            "error": format!("failed to parse response from instance: {}", e)
                        }));
                        all_successful = false;
                    }
                }
            }
            Err(e) => {
                log::error!("Request to instance {} failed: {}", instance.endpoint, e);
                results.push(serde_json::json!({
                    "instance_id": instance.id,
                    "endpoint": instance.endpoint,
                    "status_code": StatusCode::BAD_GATEWAY.as_u16(),
                    "error": format!("request to instance failed: {}", e)
                }));
                all_successful = false;
            }
        }
    }

    // Mark the weight update round as complete after first update_weights_from_agent call
    state.mark_weight_update_round_complete();

    if all_successful {
        log::info!("Weight update completed successfully for all instances");
        (StatusCode::OK, Json(serde_json::json!({ "success": true, "details": results })))
    } else {
        log::warn!("Weight update failed for some instances");
        (StatusCode::MULTI_STATUS, Json(serde_json::json!({ "success": false, "details": results })))
    }
}

pub async fn health_check() -> impl IntoResponse {
    log::debug!("Health check requested");
    (StatusCode::OK, Json(serde_json::json!({"status": "healthy", "message": "Rollout manager is ready"})))
}

pub async fn get_instances_status(State(state): State<AppState>) -> impl IntoResponse {
    let instances = state.get_all_instances().await;
    let pending: Vec<String> = state.pending_instances.iter()
        .map(|entry| entry.key().clone())
        .collect();
    
    let status = serde_json::json!({
        "ready_instances": instances.len(),
        "pending_instances": pending.len(),
        "ready_instance_endpoints": instances.iter().map(|i| &i.endpoint).collect::<Vec<_>>(),
        "pending_instance_endpoints": pending
    });
    
    (StatusCode::OK, Json(status))
}

pub async fn shutdown_instances_handler(
    State(state): State<AppState>,
    Json(payload): Json<ShutdownInstancesRequest>,
) -> impl IntoResponse {
    if payload.endpoints.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "no endpoints provided"})));
    }
    
    log::info!("Received shutdown request for {} instances: {:?}", payload.endpoints.len(), payload.endpoints);
    
    // Mark instances for shutdown in next prepare round
    state.shutdown_instances(payload.endpoints.clone());
    
    let response = serde_json::json!({
        "success": true,
        "message": format!("Marked {} instances for shutdown in next prepare round", payload.endpoints.len()),
        "endpoints": payload.endpoints
    });
    
    (StatusCode::OK, Json(response))
}

// elastic support utils


pub fn build_partial_current_response(
    is_batch_input: bool,
    current_received_responses: &Vec<Option<serde_json::Value>>,
) -> serde_json::Value {
    if is_batch_input {
        serde_json::json!(
            current_received_responses
                .iter()
                .map(|r| r.clone().unwrap_or(serde_json::json!({})))
                .collect::<Vec<_>>()
        )
    } else {
        current_received_responses
            .get(0)
            .and_then(|r| r.clone())
            .unwrap_or(serde_json::json!({}))
    }
}

pub fn merge_arrays(first: &serde_json::Value, second: &mut serde_json::Value, key_path: &[&str]) {
    let mut first_val = first;
    let mut second_val = second;
    for (i, key) in key_path.iter().enumerate() {
        if i == key_path.len() - 1 {
            if let (Some(first_arr), Some(second_arr)) = (
                first_val.get(key).and_then(|v| v.as_array()),
                second_val.get_mut(key).and_then(|v| v.as_array_mut()),
            ) {
                let mut combined = first_arr.clone();
                combined.extend(second_arr.clone());
                second_val[key] = serde_json::json!(combined);
            }
        } else {
            first_val = match first_val.get(key) {
                Some(v) => v,
                None => return,
            };
            second_val = match second_val.get_mut(key) {
                Some(v) => v,
                None => return,
            };
        }
    }
}

pub fn merge_responses(first: &serde_json::Value, second: &mut serde_json::Value) {
    merge_arrays(first, second, &["meta_info", "output_token_logprobs"]);
    
    let first_completion_tokens = first.get("meta_info")
        .and_then(|m| m.get("completion_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    
    let second_completion_tokens = second.get("meta_info")
        .and_then(|m| m.get("completion_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    
    let total_tokens = first_completion_tokens + second_completion_tokens;
    
    if let Some(meta_info) = second.get_mut("meta_info") {
        if let Some(meta_obj) = meta_info.as_object_mut() {
            meta_obj.insert("completion_tokens".to_string(), serde_json::json!(total_tokens));
        }
    }
}