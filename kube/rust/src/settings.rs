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
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct AppSettings {
    pub bind_addr: String,
    pub logs_dir: PathBuf,
    pub namespace: String,
    pub scale_up_step: i32,
    pub deployment_files: Vec<PathBuf>,
    pub default_pool: Option<DefaultPoolToml>,
    pub worker_pools: Vec<WorkerPoolToml>,
}

impl AppSettings {
    pub fn from_env() -> Result<Self> { Self::from_config_arg(None) }

    pub fn from_config_arg(config_arg: Option<String>) -> Result<Self> {
        let file_cfg = if let Some(spec) = config_arg {
            let path = resolve_config_path(&spec);
            let bytes = std::fs::read_to_string(&path)
                .with_context(|| format!("reading config at {}", path.display()))?;
            let parsed: FileConfig = toml::from_str(&bytes).context("parsing TOML config")?;
            Some(parsed)
        } else {
            None
        };

        // Defaults
        let mut bind_addr = "0.0.0.0:5000".to_string();
        let mut logs_dir = PathBuf::from("pod-logs");
        let mut namespace = "default".to_string();
        let mut scale_up_step: i32 = 1;
        let mut deployment_files: Vec<PathBuf> = Vec::new();
        let mut default_pool: Option<DefaultPoolToml> = None;
        let mut worker_pools: Vec<WorkerPoolToml> = Vec::new();

        if let Some(cfg) = &file_cfg {
            if let Some(g) = &cfg.general {
                if let Some(v) = &g.bind_addr { bind_addr = v.clone(); }
                if let Some(v) = &g.logs_dir { logs_dir = PathBuf::from(v); }
                if let Some(v) = &g.namespace { namespace = v.clone(); }
            }
            if let Some(sc) = &cfg.scaler {
                if let Some(v) = sc.scale_up_step { scale_up_step = v; }
            }
            if let Some(d) = &cfg.deployments {
                if let Some(files) = &d.files {
                    for f in files {
                        deployment_files.push(resolve_config_relative(f));
                    }
                }
            }
            if let Some(dp) = &cfg.default_pool { default_pool = Some(dp.clone()); }
            if let Some(wps) = &cfg.worker_pools { worker_pools = wps.clone(); }
        }

        // Environment overrides
        if let Ok(v) = std::env::var("BIND_ADDR") { bind_addr = v; }
        if let Ok(v) = std::env::var("LOGS_DIR") { logs_dir = PathBuf::from(v); }
        if let Ok(v) = std::env::var("NAMESPACE") { namespace = v; }
        if let Ok(v) = std::env::var("SCALE_UP_STEP") { scale_up_step = v.parse().unwrap_or(scale_up_step); }

        Ok(Self {
            bind_addr,
            logs_dir,
            namespace,
            scale_up_step,
            deployment_files,
            default_pool,
            worker_pools,
        })
    }
}

#[derive(Debug, Deserialize)]
struct FileConfig {
    #[serde(default)]
    pub general: Option<General>,
    #[serde(default)]
    pub scaler: Option<Scaler>,
    #[serde(default)]
    pub deployments: Option<DeploymentsSection>,
    #[serde(default)]
    pub default_pool: Option<DefaultPoolToml>,
    #[serde(default)]
    pub worker_pools: Option<Vec<WorkerPoolToml>>,
}

#[derive(Debug, Deserialize)]
struct General { pub project: Option<String>, pub region: Option<String>, pub zone: Option<String>, pub cluster_name: Option<String>, pub vpc_net: Option<String>, pub vpc_subnet: Option<String>, pub namespace: Option<String>, pub logs_dir: Option<String>, pub bind_addr: Option<String> }

#[derive(Debug, Deserialize)]
struct Scaler { pub scale_up_step: Option<i32> }

#[derive(Debug, Deserialize)]
struct DeploymentsSection { pub files: Option<Vec<String>> }

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct DefaultPoolToml { pub machine_type: Option<String>, pub min_nodes: Option<i32>, pub max_nodes: Option<i32> }

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct WorkerPoolToml { pub machine_type: Option<String>, pub provision: Option<String>, pub accelerator: Option<String>, pub min_nodes: Option<i32>, pub max_nodes: Option<i32> }

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points to rust/; repo root is its parent
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn resolve_config_path(spec: &str) -> PathBuf {
    let candidate = PathBuf::from(spec);
    if candidate.exists() { return candidate; }
    let by_name = repo_root().join("config").join(format!("{}.toml", spec));
    by_name
}

fn resolve_config_relative(rel: &str) -> PathBuf {
    let from_root = repo_root().join(rel);
    from_root
}

