// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached gRPC client wrapper around an [`AuthedChannel`].
//!
//! The orchestrator (`openshell-supervisor`) builds a single
//! [`AuthedChannel`] via [`openshell_core::grpc::connect_authed_channel`]
//! and shares it with every component that talks to the gateway. This
//! module wraps that channel in typed RPC helpers used by the supervisor
//! (policy polling, settings, denial summaries, draft proposals, status
//! reporting, provider environment, inference bundle).

use std::collections::HashMap;

use miette::{IntoDiagnostic, Result, WrapErr};
use openshell_core::grpc::AuthedChannel;
use openshell_core::proto::{
    DenialSummary, GetDraftPolicyRequest, GetInferenceBundleRequest, GetInferenceBundleResponse,
    GetSandboxConfigRequest, GetSandboxProviderEnvironmentRequest, PolicyChunk, PolicySource,
    PolicyStatus, ReportPolicyStatusRequest, SandboxPolicy as ProtoSandboxPolicy,
    SubmitPolicyAnalysisRequest, SubmitPolicyAnalysisResponse, UpdateConfigRequest,
    inference_client::InferenceClient, open_shell_client::OpenShellClient,
};
use tracing::debug;

/// Settings poll result returned by [`CachedOpenShellClient::poll_settings`].
pub struct SettingsPollResult {
    pub policy: Option<ProtoSandboxPolicy>,
    pub version: u32,
    pub policy_hash: String,
    pub config_revision: u64,
    pub policy_source: PolicySource,
    /// Effective settings keyed by name.
    pub settings: HashMap<String, openshell_core::proto::EffectiveSetting>,
    /// When `policy_source` is `Global`, the version of the global policy revision.
    pub global_policy_version: u32,
    pub provider_env_revision: u64,
}

pub struct ProviderEnvironmentResult {
    pub environment: HashMap<String, String>,
    pub provider_env_revision: u64,
    pub credential_expires_at_ms: HashMap<String, i64>,
}

/// Cached `OpenShell` gRPC client built on top of an [`AuthedChannel`].
///
/// Cloning is cheap — the underlying channel is multiplexed over a single
/// HTTP/2 connection and the [`AuthInterceptor`](openshell_core::grpc::AuthInterceptor)
/// reads the bearer token from a process-wide slot.
#[derive(Clone)]
pub struct CachedOpenShellClient {
    channel: AuthedChannel,
    client: OpenShellClient<AuthedChannel>,
}

impl CachedOpenShellClient {
    /// Wrap an already-authenticated channel.
    pub fn new(channel: AuthedChannel) -> Self {
        Self {
            client: OpenShellClient::new(channel.clone()),
            channel,
        }
    }

    /// Get a clone of the underlying tonic client for direct RPC calls
    /// (e.g. server-streaming RPCs that the typed helpers below don't cover).
    pub fn raw_client(&self) -> OpenShellClient<AuthedChannel> {
        self.client.clone()
    }

    /// Fetch sandbox policy from `OpenShell` server via gRPC.
    ///
    /// Returns `Ok(Some(policy))` when the server has a policy configured,
    /// or `Ok(None)` when the sandbox was created without a policy (the
    /// caller should discover one from disk or use the restrictive default).
    pub async fn fetch_policy(&self, sandbox_id: &str) -> Result<Option<ProtoSandboxPolicy>> {
        debug!(sandbox_id = %sandbox_id, "Fetching sandbox policy");
        let response = self
            .client
            .clone()
            .get_sandbox_config(GetSandboxConfigRequest {
                sandbox_id: sandbox_id.to_string(),
            })
            .await
            .into_diagnostic()?;

        let inner = response.into_inner();
        if inner.version == 0 && inner.policy.is_none() {
            return Ok(None);
        }
        Ok(Some(inner.policy.ok_or_else(|| {
            miette::miette!("Server returned non-zero version but empty policy")
        })?))
    }

    /// Sync a locally-discovered or enriched policy to the gateway.
    pub async fn sync_policy(&self, sandbox: &str, policy: &ProtoSandboxPolicy) -> Result<()> {
        debug!(sandbox = %sandbox, "Syncing policy to gateway");
        self.client
            .clone()
            .update_config(UpdateConfigRequest {
                name: sandbox.to_string(),
                policy: Some(policy.clone()),
                setting_key: String::new(),
                setting_value: None,
                delete_setting: false,
                global: false,
                merge_operations: vec![],
                expected_resource_version: 0,
            })
            .await
            .into_diagnostic()
            .wrap_err("failed to sync policy to server")?;
        Ok(())
    }

    /// Discover-and-sync flow: push the discovered policy, then re-fetch the
    /// canonical version from the gateway. Returns the gateway's authoritative
    /// view (with version/hash assigned).
    pub async fn discover_and_sync_policy(
        &self,
        sandbox_id: &str,
        sandbox: &str,
        discovered_policy: &ProtoSandboxPolicy,
    ) -> Result<ProtoSandboxPolicy> {
        debug!(
            sandbox_id = %sandbox_id,
            sandbox = %sandbox,
            "Syncing discovered policy and re-fetching canonical version"
        );
        self.sync_policy(sandbox, discovered_policy).await?;
        self.fetch_policy(sandbox_id).await?.ok_or_else(|| {
            miette::miette!("Server still returned no policy after sync — this is a bug")
        })
    }

    /// Fetch provider environment variables for a sandbox.
    pub async fn fetch_provider_environment(
        &self,
        sandbox_id: &str,
    ) -> Result<ProviderEnvironmentResult> {
        debug!(sandbox_id = %sandbox_id, "Fetching provider environment");
        let response = self
            .client
            .clone()
            .get_sandbox_provider_environment(GetSandboxProviderEnvironmentRequest {
                sandbox_id: sandbox_id.to_string(),
            })
            .await
            .into_diagnostic()?;

        let inner = response.into_inner();
        Ok(ProviderEnvironmentResult {
            environment: inner.environment,
            provider_env_revision: inner.provider_env_revision,
            credential_expires_at_ms: inner.credential_expires_at_ms,
        })
    }

    /// Fetch the resolved inference route bundle.
    pub async fn fetch_inference_bundle(&self) -> Result<GetInferenceBundleResponse> {
        debug!("Fetching inference route bundle");
        let mut client = InferenceClient::new(self.channel.clone());
        let response = client
            .get_inference_bundle(GetInferenceBundleRequest {})
            .await
            .into_diagnostic()?;
        Ok(response.into_inner())
    }

    /// Poll for current effective sandbox settings and policy metadata.
    pub async fn poll_settings(&self, sandbox_id: &str) -> Result<SettingsPollResult> {
        let response = self
            .client
            .clone()
            .get_sandbox_config(GetSandboxConfigRequest {
                sandbox_id: sandbox_id.to_string(),
            })
            .await
            .into_diagnostic()?;

        let inner = response.into_inner();

        Ok(SettingsPollResult {
            policy: inner.policy,
            version: inner.version,
            policy_hash: inner.policy_hash,
            config_revision: inner.config_revision,
            policy_source: PolicySource::try_from(inner.policy_source)
                .unwrap_or(PolicySource::Unspecified),
            settings: inner.settings,
            global_policy_version: inner.global_policy_version,
            provider_env_revision: inner.provider_env_revision,
        })
    }

    /// Submit denial summaries and/or agent-authored proposals for policy analysis.
    ///
    /// Returns the gateway response so callers can surface accepted/rejected
    /// counts, rejection reasons, and server-assigned `accepted_chunk_ids`
    /// (e.g., the `policy.local` API forwards these to the in-sandbox agent
    /// so it can watch proposal state via `GET /v1/proposals/{id}`).
    pub async fn submit_policy_analysis(
        &self,
        sandbox_name: &str,
        summaries: Vec<DenialSummary>,
        proposed_chunks: Vec<PolicyChunk>,
        analysis_mode: &str,
    ) -> Result<SubmitPolicyAnalysisResponse> {
        let response = self
            .client
            .clone()
            .submit_policy_analysis(SubmitPolicyAnalysisRequest {
                name: sandbox_name.to_string(),
                summaries,
                proposed_chunks,
                analysis_mode: analysis_mode.to_string(),
            })
            .await
            .into_diagnostic()?;

        Ok(response.into_inner())
    }

    /// Fetch the current draft chunks for a sandbox. `status_filter` may be
    /// `"pending"`, `"approved"`, `"rejected"`, or empty for all. Used by
    /// `policy.local`'s `GET /v1/proposals/{id}` and `/wait` routes to
    /// inspect proposal state.
    pub async fn get_draft_policy(
        &self,
        sandbox_name: &str,
        status_filter: &str,
    ) -> Result<Vec<PolicyChunk>> {
        let response = self
            .client
            .clone()
            .get_draft_policy(GetDraftPolicyRequest {
                name: sandbox_name.to_string(),
                status_filter: status_filter.to_string(),
            })
            .await
            .into_diagnostic()?;
        Ok(response.into_inner().chunks)
    }

    /// Report policy load status back to the server.
    pub async fn report_policy_status(
        &self,
        sandbox_id: &str,
        version: u32,
        loaded: bool,
        error_msg: &str,
    ) -> Result<()> {
        let status = if loaded {
            PolicyStatus::Loaded
        } else {
            PolicyStatus::Failed
        };

        self.client
            .clone()
            .report_policy_status(ReportPolicyStatusRequest {
                sandbox_id: sandbox_id.to_string(),
                version,
                status: status.into(),
                load_error: error_msg.to_string(),
            })
            .await
            .into_diagnostic()?;

        Ok(())
    }
}
