// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `OpenShell` supervisor process component.
//!
//! This crate hosts the process-side modules of the supervisor: sandbox
//! isolation primitives (Linux netns, Landlock, seccomp), the SSH server,
//! the supervised-child lifecycle, provider credential snapshots, the bypass
//! monitor, and the process-side gRPC client. It is one of two leaf crates
//! that together form the supervisor binary; the other is
//! `openshell-supervisor-network`.
//!
//! Cross-cutting context (OCSF context) is re-exported from `openshell-core`
//! so existing `crate::ocsf_ctx()` call sites in the moved modules resolve
//! without modification.

pub(crate) use openshell_core::ocsf_ctx::ocsf_ctx;

pub mod bypass_monitor;
pub mod child_env;
pub mod grpc_client;
pub mod log_push;
pub mod managed_children;
pub mod process;
pub mod provider_credentials;
pub mod sandbox;
pub mod skills;
pub mod ssh;

pub use managed_children::{register_managed_child, unregister_managed_child};
