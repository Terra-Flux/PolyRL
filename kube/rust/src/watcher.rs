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
use std::{fs::{self, File}, io::Write, path::PathBuf, sync::Arc};

use anyhow::Result;
use futures::{StreamExt, TryStreamExt, AsyncBufReadExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{api::{ListParams, LogParams}, Api, Client, runtime::watcher};

use crate::server::AppState;

pub async fn run_pod_monitor(state: Arc<AppState>) -> Result<()> {
    // Recreate top-level logs dir on startup to ensure a clean directory
    if state.cfg.logs_dir.exists() {
        tracing::warn!(dir=%state.cfg.logs_dir.display(), "removing existing logs directory");
        match fs::remove_dir_all(&state.cfg.logs_dir) {
            Ok(_) => {}
            Err(e) => tracing::warn!(dir=%state.cfg.logs_dir.display(), error=?e, "failed to remove logs directory, continuing"),
        }
    }
    fs::create_dir_all(&state.cfg.logs_dir)?;
    tracing::info!(dir=%state.cfg.logs_dir.display(), namespace=%state.cfg.namespace, "starting pod monitor and ensuring logs dir");
    let namespace = state.cfg.namespace.clone();
    let client = state.client.clone();
    let pod_api: Api<Pod> = Api::namespaced(client.clone(), &namespace);

    // initial sweep: fetch current pods and ensure log files exist
    let pods = pod_api.list(&ListParams::default()).await?;
    for p in pods {
        if let Some(name) = p.metadata.name {
            let dir = pod_logs_dir(&state.cfg.logs_dir, p.metadata.labels.as_ref());
            tracing::debug!(pod=%name, dir=%dir.display(), "ensuring initial log file");
            ensure_log_file(&dir, &name)?;
        }
    }

    // watcher for pod events
    let mut w = watcher(pod_api.clone(), watcher::Config::default()).boxed();
    while let Some(ev) = w.try_next().await? {
        use kube::runtime::watcher::Event;
        match ev {
            Event::Applied(p) => {
                if let Some(name) = p.metadata.name.clone() {
                    if is_running(&p) {
                        let client = client.clone();
                        let namespace = namespace.clone();
                        let logs_dir = pod_logs_dir(&state.cfg.logs_dir, p.metadata.labels.as_ref());
                        tracing::info!(pod=%name, dir=%logs_dir.display(), "starting log stream for running pod");
                        tokio::spawn(async move {
                            if let Err(e) = stream_pod_logs(client, &namespace, &name, logs_dir.clone()).await {
                                tracing::warn!(pod=%name, dir=%logs_dir.display(), error=?e, "log stream ended with error");
                            }
                        });
                    }
                }
            }
            Event::Deleted(p) => {
                if let Some(name) = p.metadata.name { tracing::info!(pod=%name, "pod deleted"); }
            }
            Event::Restarted(ps) => {
                for p in ps {
                    if let Some(name) = p.metadata.name.clone() {
                        if is_running(&p) {
                            let client = client.clone();
                            let namespace = namespace.clone();
                            let logs_dir = pod_logs_dir(&state.cfg.logs_dir, p.metadata.labels.as_ref());
                            tracing::info!(pod=%name, dir=%logs_dir.display(), "restarting log stream after watcher restart");
                            tokio::spawn(async move {
                                if let Err(e) = stream_pod_logs(client, &namespace, &name, logs_dir.clone()).await {
                                    tracing::warn!(pod=%name, dir=%logs_dir.display(), error=?e, "log stream ended with error");
                                }
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn is_running(p: &Pod) -> bool {
    p.status
        .as_ref()
        .and_then(|s| s.phase.as_deref())
        .map(|ph| ph == "Running")
        .unwrap_or(false)
}

fn ensure_log_file(dir: &PathBuf, pod: &str) -> Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.log", pod));
    if !path.exists() { File::create(&path)?; }
    Ok(path)
}

async fn stream_pod_logs(client: Client, ns: &str, pod: &str, dir: PathBuf) -> Result<()> {
    let api: Api<Pod> = Api::namespaced(client, ns);
    let mut lp = LogParams::default();
    lp.follow = true;
    lp.tail_lines = Some(100);
    tracing::debug!(pod=%pod, namespace=%ns, "requesting log stream from kube API");
    let reader = api.log_stream(pod, &lp).await?;
    let log_path = ensure_log_file(&dir, pod)?;
    tracing::debug!(pod=%pod, file=%log_path.display(), "opened log file for append");
    let mut file = File::options().append(true).open(&log_path)?;
    let mut lines = reader.lines();
    while let Some(line) = lines.next().await {
        match line {
            Ok(l) => {
                file.write_all(l.as_bytes())?;
                file.write_all(b"\n")?;
                file.flush()?;
                // keep this at debug to avoid noise
                tracing::debug!(pod=%pod, bytes=l.len(), "wrote log line");
            }
            Err(e) => {
                tracing::warn!(pod=%pod, error=?e, "error reading log line");
                break;
            }
        }
    }
    Ok(())
}

fn pod_logs_dir(base: &PathBuf, labels: Option<&std::collections::BTreeMap<String, String>>) -> PathBuf {
    let app = labels
        .and_then(|m| m.get("app").cloned())
        .unwrap_or_else(|| "unknown".to_string());
    base.join(app)
}


