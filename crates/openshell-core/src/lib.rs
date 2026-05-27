// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `OpenShell` Core - shared library for `OpenShell` components.
//!
//! This crate provides:
//! - Protocol buffer definitions and generated code
//! - Configuration management
//! - Common error types
//! - Build version metadata

pub mod auth;
pub mod config;
pub mod denial;
pub mod driver_utils;
pub mod error;
pub mod forward;
pub mod gpu;
pub mod grpc_retry;
pub mod image;
pub mod inference;
pub mod metadata;
pub mod net;
pub mod ocsf_ctx;
pub mod paths;
pub mod policy;
pub mod procfs;
pub mod progress;
pub mod proto;
pub mod sandbox_env;
pub mod secrets;
pub mod settings;
pub mod time;

pub use config::{
    ComputeDriverKind, Config, GatewayAuthConfig, GatewayJwtConfig, MtlsAuthConfig, OidcConfig,
    TlsConfig,
};
pub use denial::DenialEvent;
pub use error::{ComputeDriverError, Error, Result};
pub use grpc_retry::{grpc_retry, is_retryable_error};
pub use metadata::{GetResourceVersion, ObjectId, ObjectLabels, ObjectName, SetResourceVersion};
pub use ocsf_ctx::{
    agent_proposals_enabled, agent_proposals_enabled_flag, init_agent_proposals_enabled,
    init_ocsf_ctx, ocsf_ctx,
};

/// Build version string derived from git metadata.
///
/// For local builds this is computed by `build.rs` via `git describe` using
/// the guess-next-dev scheme (e.g. `0.0.4-dev.6+g2bf9969`). In Docker/CI
/// builds where `.git` is absent, falls back to `CARGO_PKG_VERSION` which
/// is already set correctly by the build pipeline's sed patch.
pub const VERSION: &str = match option_env!("OPENSHELL_GIT_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};
