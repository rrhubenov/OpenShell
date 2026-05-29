// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-side gRPC client.
//!
//! Wraps an [`AuthedChannel`] for RPCs that originate on the supervisor's
//! process side (today: log push). The channel is built once by the
//! orchestrator (`openshell-supervisor`) and shared with this side, so the
//! gateway-minted JWT installed by [`openshell_core::grpc::connect_authed_channel`]
//! is automatically applied to every outbound request.

use openshell_core::grpc::AuthedChannel;
use openshell_core::proto::open_shell_client::OpenShellClient;

/// Reusable gRPC client owned by the process side.
///
/// Cloning is cheap — the underlying channel is multiplexed over a single
/// HTTP/2 connection and the bearer token comes from a process-wide slot
/// that the renewal loop keeps fresh.
#[derive(Clone)]
pub struct ProcessGrpcClient {
    client: OpenShellClient<AuthedChannel>,
}

impl ProcessGrpcClient {
    /// Wrap an already-authenticated channel.
    pub fn new(channel: AuthedChannel) -> Self {
        Self {
            client: OpenShellClient::new(channel),
        }
    }

    /// Get a clone of the underlying tonic client for direct RPC calls
    /// (e.g. `PushSandboxLogs` server-streaming).
    pub fn raw_client(&self) -> OpenShellClient<AuthedChannel> {
        self.client.clone()
    }
}
