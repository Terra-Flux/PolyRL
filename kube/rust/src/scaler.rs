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
use anyhow::{anyhow, bail, Context, Result};
use k8s_openapi::api::apps::v1::Deployment;
use kube::{
    api::{Patch, PatchParams},
    core::{DynamicObject, GroupVersionKind},
    discovery::{Discovery, Scope},
    Api, Client,
};
use kube::ResourceExt; // for name_any
use serde::de::DeserializeOwned;
use std::path::Path;
use kube::error::Error as KubeError;

pub async fn scale_deployment(client: Client, namespace: &str, deployment: &str, replicas: i32) -> Result<()> {
    if replicas < 0 { bail!("replicas must be >= 0"); }
    let api: Api<Deployment> = Api::namespaced(client, namespace);
    let patch = serde_json::json!({
        "spec": {"replicas": replicas}
    });
    tracing::debug!(deployment=%deployment, namespace=%namespace, replicas=replicas, "sending merge patch to scale deployment");
    match api.patch(
        deployment,
        &PatchParams::default(),
        &Patch::Merge(patch),
    ).await {
        Ok(_) => {
            tracing::info!(deployment=%deployment, namespace=%namespace, replicas=replicas, "deployment scaled successfully");
            Ok(())
        }
        Err(KubeError::Api(err)) => {
            // Surface the Kubernetes API error details directly
            Err(anyhow!("Kubernetes API error ({}): {} [reason={}, code={}]", deployment, err.message, err.reason, err.code))
        }
        Err(e) => {
            Err(e).with_context(|| format!("scaling deployment {} in {}", deployment, namespace))
        }
    }
}

pub async fn apply_yaml_manifest(client: Client, default_namespace: &str, path: &Path) -> Result<()> {
    let yaml = std::fs::read_to_string(path)
        .with_context(|| format!("reading manifest at {}", path.display()))?;
    let discovery = Discovery::new(client.clone()).run().await?;
    let ssapply = PatchParams::apply("gke-rl-scaler").force();
    for doc in yaml_documents(&yaml)? {
        let obj: DynamicObject = serde_yaml::from_value(doc)?;
        let ns = obj
            .metadata
            .namespace
            .as_deref()
            .unwrap_or(default_namespace);
        let gvk = if let Some(tm) = &obj.types {
            GroupVersionKind::try_from(tm)?
        } else {
            bail!("cannot apply object without valid TypeMeta: {:?}", obj);
        };
        let name = obj.name_any();
        if let Some((ar, caps)) = discovery.resolve_gvk(&gvk) {
            let api: Api<DynamicObject> = if caps.scope == Scope::Namespaced {
                Api::namespaced_with(client.clone(), ns, &ar)
            } else {
                Api::all_with(client.clone(), &ar)
            };
            let data: serde_json::Value = serde_json::to_value(&obj)?;
            tracing::info!(kind=%gvk.kind, name=%name, namespace=%ns, "applying manifest via SSA");
            api.patch(&name, &ssapply, &Patch::Apply(data)).await?;
        } else {
            tracing::warn!(?gvk, "cannot apply document for unknown GVK");
        }
    }
    Ok(())
}

fn yaml_documents<T: DeserializeOwned>(yaml: &str) -> Result<Vec<T>> {
    let mut docs = Vec::new();
    for d in serde_yaml::Deserializer::from_str(yaml) {
        let v = T::deserialize(d).context("parsing one YAML document")?;
        docs.push(v);
    }
    Ok(docs)
}

