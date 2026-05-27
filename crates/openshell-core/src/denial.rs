// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared denial event type emitted by the proxy and bypass monitor and
//! consumed by the supervisor's denial aggregator.
//!
//! Lifted out of `openshell-sandbox` so the network and process supervisor
//! components can both reference the same type without depending on each
//! other (issue #1305 / RFC-0001).

/// A single denial event emitted by the proxy or bypass monitor.
#[derive(Debug, Clone)]
pub struct DenialEvent {
    /// Destination host that was denied.
    pub host: String,
    /// Destination port that was denied.
    pub port: u16,
    /// Binary path that initiated the connection (if resolved).
    pub binary: String,
    /// Ancestor binary paths from process tree walk.
    pub ancestors: Vec<String>,
    /// Reason for denial (e.g. "no matching policy", "internal address").
    pub deny_reason: String,
    /// Denial stage: "connect", "forward", "ssrf", "l7", "bypass".
    pub denial_stage: String,
    /// L7 request details (method, path, decision) if this is an L7 denial.
    pub l7_method: Option<String>,
    /// L7 target path.
    pub l7_path: Option<String>,
}
