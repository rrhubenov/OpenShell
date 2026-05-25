// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `OpenShell` supervisor network component.
//!
//! This crate hosts the network-side modules of the supervisor: the CONNECT
//! proxy, OPA policy engine, L7 enforcement (HTTP/WebSocket/GraphQL), denial
//! aggregation, secret rewriting, and the binary identity cache. It is one of
//! two leaf crates that together form the supervisor binary; the other is
//! `openshell-supervisor-process`.
//!
//! Cross-cutting context (OCSF context, agent-proposals flag) is re-exported
//! from `openshell-core` so existing `crate::ocsf_ctx()` call sites in the
//! moved modules resolve without modification.

pub(crate) use openshell_core::ocsf_ctx::{agent_proposals_enabled, ocsf_ctx};

/// Test-only helpers shared across sibling test modules.
#[cfg(test)]
pub(crate) mod test_helpers {
    #![allow(
        clippy::redundant_pub_crate,
        reason = "intentional crate-private module"
    )]
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::MutexGuard;

    static PROPOSALS_FLAG_LOCK: LazyLock<tokio::sync::Mutex<()>> =
        LazyLock::new(|| tokio::sync::Mutex::new(()));

    /// Process-wide flag handle used by tests when the supervisor binary has
    /// not initialized the global flag. Tests use [`ProposalsFlagGuard`] to
    /// flip this value through a serialized async mutex; the guard restores
    /// the previous value on drop.
    static TEST_PROPOSALS_FLAG: LazyLock<Arc<AtomicBool>> =
        LazyLock::new(|| Arc::new(AtomicBool::new(false)));

    /// Guard for tests that toggle the process-wide agent-proposals flag.
    /// Acquires a process-wide async mutex, swaps in the requested value, and
    /// restores the previous value on drop. Hold the guard for the duration
    /// of any code that reads `agent_proposals_enabled()`.
    pub(crate) struct ProposalsFlagGuard {
        prev: bool,
        flag: Arc<AtomicBool>,
        _lock: MutexGuard<'static, ()>,
    }

    impl ProposalsFlagGuard {
        pub(crate) async fn set(enabled: bool) -> Self {
            let lock = PROPOSALS_FLAG_LOCK.lock().await;
            Self::with_lock(enabled, lock)
        }

        pub(crate) fn set_blocking(enabled: bool) -> Self {
            let lock = PROPOSALS_FLAG_LOCK.blocking_lock();
            Self::with_lock(enabled, lock)
        }

        fn with_lock(enabled: bool, lock: MutexGuard<'static, ()>) -> Self {
            let flag = openshell_core::ocsf_ctx::agent_proposals_enabled_flag()
                .cloned()
                .unwrap_or_else(|| {
                    let f = TEST_PROPOSALS_FLAG.clone();
                    let _ = openshell_core::ocsf_ctx::init_agent_proposals_enabled(f.clone());
                    f
                });
            let prev = flag.swap(enabled, Ordering::Relaxed);
            Self {
                prev,
                flag,
                _lock: lock,
            }
        }
    }

    impl Drop for ProposalsFlagGuard {
        fn drop(&mut self) {
            self.flag.store(self.prev, Ordering::Relaxed);
        }
    }
}

pub mod denial_aggregator;
pub mod grpc_client;
pub mod identity;
pub mod l7;
pub mod mechanistic_mapper;
pub mod opa;
pub mod policy_local;
pub mod proxy;
pub mod secrets;
