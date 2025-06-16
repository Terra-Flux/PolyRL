use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub host: String,
    pub port: u16,
    pub mooncake_handshake_port: Option<u16>,
}

#[derive(Debug, Serialize, Clone)]
pub struct RegisterResponse {
    pub mooncake_transfer_device_name: String,
    pub mooncake_transfer_protocol: String,
    pub weight_sender_rpyc_endpoint: String,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest(pub serde_json::Value);

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct UpdateWeightsFromAgentRequest {
    pub tensors_meta: Vec<(String, (Vec<i64>, String))>,
    pub load_format: Option<String>,
    pub flush_cache: Option<bool>,
    pub bootstrap: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateWeightSendersRequest {
    pub weight_sender_rpyc_endpoints: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ShutdownInstancesRequest {
    pub endpoints: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct PrepareWeightUpdateRequest {
    pub weight_sender_endpoint: String,
}

#[derive(Clone)]
pub struct Instance {
    pub id: Uuid,
    pub endpoint: String, // full url prefix like http://host:port
    pub mooncake_handshake_endpoint: Option<String>, // host:port for mooncake handshake
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub mooncake_transfer_device_name: String,
    pub mooncake_transfer_protocol: String,
    pub weight_sender_rpyc_endpoints: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct PrepareWeightUpdateResponse {
    pub current_instances: Vec<InstanceInfo>,
}

#[derive(Debug, Serialize)]
pub struct InstanceInfo {
    pub id: Uuid,
    pub endpoint: String,
    pub weight_sender_endpoint: String,
    pub mooncake_handshake_endpoint: Option<String>,
}