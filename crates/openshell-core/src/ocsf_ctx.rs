// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide OCSF sandbox context and agent-proposals feature flag.
//!
//! Both supervisor components (network and process) need to read these values
//! to attach the same identity to OCSF events and to gate the agent-proposal
//! mutation surface. The storage and getters live here so neither leaf crate
//! owns the `OnceLock`s; the supervisor binary calls [`init_ocsf_ctx`] and
//! [`init_agent_proposals_enabled`] once at startup.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use openshell_ocsf::SandboxContext;

/// Process-wide OCSF sandbox context. Initialized once during supervisor
/// startup via [`init_ocsf_ctx`] and accessible from any crate via
/// [`ocsf_ctx`].
static OCSF_CTX: OnceLock<SandboxContext> = OnceLock::new();

/// Fallback context used when [`OCSF_CTX`] has not been initialized (e.g. in
/// unit tests that exercise individual functions without calling the
/// supervisor entrypoint).
static OCSF_CTX_FALLBACK: LazyLock<SandboxContext> = LazyLock::new(|| SandboxContext {
    sandbox_id: String::new(),
    sandbox_name: String::new(),
    container_image: String::new(),
    hostname: "test".to_string(),
    product_version: crate::VERSION.to_string(),
    proxy_ip: std::net::IpAddr::from([127, 0, 0, 1]),
    proxy_port: 3128,
});

/// Initialize the process-wide [`SandboxContext`]. Returns `Err(ctx)` with the
/// passed-in context if it was already initialized, matching `OnceLock::set`.
#[allow(
    clippy::result_large_err,
    reason = "mirrors OnceLock::set, which returns the rejected value"
)]
pub fn init_ocsf_ctx(ctx: SandboxContext) -> Result<(), SandboxContext> {
    OCSF_CTX.set(ctx)
}

/// Return a reference to the process-wide [`SandboxContext`].
///
/// Falls back to a default context if [`init_ocsf_ctx`] has not yet been
/// called (e.g. during unit tests).
pub fn ocsf_ctx() -> &'static SandboxContext {
    OCSF_CTX.get().unwrap_or(&OCSF_CTX_FALLBACK)
}

/// Process-wide flag for the agent-driven policy proposal surface. Set once
/// during supervisor startup and updated by the settings poll loop when
/// `agent_policy_proposals_enabled` changes.
static AGENT_PROPOSALS_ENABLED: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Initialize the process-wide agent-proposals feature flag.
///
/// Returns `Err(flag)` if the flag was already initialized, matching
/// `OnceLock::set`.
pub fn init_agent_proposals_enabled(flag: Arc<AtomicBool>) -> Result<(), Arc<AtomicBool>> {
    AGENT_PROPOSALS_ENABLED.set(flag)
}

/// Return the underlying flag handle, if it has been initialized.
pub fn agent_proposals_enabled_flag() -> Option<&'static Arc<AtomicBool>> {
    AGENT_PROPOSALS_ENABLED.get()
}

/// Read the current value of the agent proposals feature flag.
///
/// Returns `false` if the flag has not been initialized (e.g. during unit
/// tests), matching the documented default for the setting.
pub fn agent_proposals_enabled() -> bool {
    AGENT_PROPOSALS_ENABLED
        .get()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}
