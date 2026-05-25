// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-side gRPC client.
//!
//! Owns the channel/TLS plumbing for RPCs that originate on the supervisor's
//! process side (log push, and — once moved — the policy/settings poll loop
//! and the inference-route fetcher).

use std::time::Duration;

use miette::{IntoDiagnostic, Result, WrapErr};
use openshell_core::proto::open_shell_client::OpenShellClient;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};
use tracing::debug;

/// Create a channel to the `OpenShell` server.
///
/// When the endpoint uses `https://`, mTLS is configured using these env vars:
/// - `OPENSHELL_TLS_CA` -- path to the CA certificate
/// - `OPENSHELL_TLS_CERT` -- path to the client certificate
/// - `OPENSHELL_TLS_KEY` -- path to the client private key
///
/// When the endpoint uses `http://`, a plaintext connection is used (for
/// deployments where TLS is disabled, e.g. behind a Cloudflare Tunnel).
pub(crate) async fn connect_channel(endpoint: &str) -> Result<Channel> {
    let mut ep = Endpoint::from_shared(endpoint.to_string())
        .into_diagnostic()
        .wrap_err("invalid gRPC endpoint")?
        .connect_timeout(Duration::from_secs(10))
        .http2_keep_alive_interval(Duration::from_secs(10))
        .keep_alive_while_idle(true)
        .keep_alive_timeout(Duration::from_secs(20))
        .http2_adaptive_window(true);

    let tls_enabled = endpoint.starts_with("https://");

    if tls_enabled {
        let ca_path = std::env::var(openshell_core::sandbox_env::TLS_CA)
            .into_diagnostic()
            .wrap_err("OPENSHELL_TLS_CA is required")?;
        let cert_path = std::env::var(openshell_core::sandbox_env::TLS_CERT)
            .into_diagnostic()
            .wrap_err("OPENSHELL_TLS_CERT is required")?;
        let key_path = std::env::var(openshell_core::sandbox_env::TLS_KEY)
            .into_diagnostic()
            .wrap_err("OPENSHELL_TLS_KEY is required")?;

        let ca_pem = std::fs::read(&ca_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to read CA cert from {ca_path}"))?;
        let cert_pem = std::fs::read(&cert_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to read client cert from {cert_path}"))?;
        let key_pem = std::fs::read(&key_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to read client key from {key_path}"))?;

        let tls_config = ClientTlsConfig::new()
            .ca_certificate(Certificate::from_pem(ca_pem))
            .identity(Identity::from_pem(cert_pem, key_pem));

        ep = ep
            .tls_config(tls_config)
            .into_diagnostic()
            .wrap_err("failed to configure TLS")?;
    }

    ep.connect()
        .await
        .into_diagnostic()
        .wrap_err("failed to connect to OpenShell server")
}

/// Reusable gRPC client owned by the process side.
///
/// Currently exposes the underlying `OpenShellClient` for the log-push streamer
/// to call `PushSandboxLogs`. As process-owned RPCs migrate over from the
/// network-side `CachedOpenShellClient`, they will be added as methods here.
#[derive(Clone)]
pub struct ProcessGrpcClient {
    client: OpenShellClient<Channel>,
}

impl ProcessGrpcClient {
    pub async fn connect(endpoint: &str) -> Result<Self> {
        debug!(endpoint = %endpoint, "Connecting process-side gRPC client");
        let channel = connect_channel(endpoint).await?;
        Ok(Self {
            client: OpenShellClient::new(channel),
        })
    }

    /// Get a clone of the underlying tonic client for direct RPC calls.
    pub fn raw_client(&self) -> OpenShellClient<Channel> {
        self.client.clone()
    }
}
