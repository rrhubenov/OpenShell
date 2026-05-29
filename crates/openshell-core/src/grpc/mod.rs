// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway gRPC client wiring: plain channel construction, Bearer-token auth,
//! and the once-per-process refresh loop.
//!
//! Entry point: [`connect_authed_channel`] builds an [`AuthedChannel`] —
//! a tonic [`Channel`](tonic::transport::Channel) wrapped in an
//! [`AuthInterceptor`] that injects the gateway-minted JWT on every outbound
//! request. The first call per process resolves the sandbox JWT (env → file
//! → K8s SA bootstrap exchange), installs it into a process-wide token slot,
//! and spawns the renewal loop. Later calls reuse the same slot — token
//! refreshes mutate the slot in place, so existing clients pick up new
//! tokens on their next request without rebuilding.
//!
//! The orchestrator (`openshell-supervisor`) calls this once and shares the
//! resulting [`AuthedChannel`] with every leaf component that needs to talk
//! to the gateway. Cloning an [`AuthedChannel`] is cheap and reuses the
//! underlying HTTP/2 connection.

mod auth;
mod transport;

pub use auth::{AuthInterceptor, AuthedChannel};

use miette::Result;
use tonic::service::interceptor::InterceptedService;

/// Build a Bearer-authenticated gateway channel.
///
/// First call per process resolves the sandbox JWT via the three-step
/// lookup (env → file → K8s SA bootstrap exchange) and installs it into
/// the process-wide token slot. Subsequent calls reuse the cached slot —
/// the renewal loop keeps the value fresh, so re-running the bootstrap
/// is both unnecessary and (on the K8s SA path) expensive (one apiserver
/// round-trip per call). The renewal loop itself is spawned once per
/// process via an internal one-shot guard.
pub async fn connect_authed_channel(endpoint: &str) -> Result<AuthedChannel> {
    let channel = transport::build_plain_channel(endpoint).await?;
    let (slot, source) = auth::token_slot(endpoint, &channel).await?;
    let plain_channel = channel.clone();
    let intercepted = InterceptedService::new(channel, AuthInterceptor::new(slot.clone()));
    if auth::REFRESH_SPAWNED.set(()).is_ok() {
        let refresh_channel = intercepted.clone();
        let endpoint = endpoint.to_string();
        tokio::spawn(async move {
            auth::refresh_token_loop(refresh_channel, slot, source, endpoint, plain_channel).await;
        });
    }
    Ok(intercepted)
}
