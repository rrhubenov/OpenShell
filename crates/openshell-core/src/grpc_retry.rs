// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared gRPC retry helpers used by both supervisor components.
//!
//! Lifted out of `openshell-sandbox` so the network and process gRPC clients
//! can share the same transient-error classification and exponential backoff
//! loop (issue #1305 / RFC-0001).

use std::future::Future;
use std::time::Duration;

use miette::Result;
use tracing::warn;

/// Returns `true` if the error is transient and worth retrying.
///
/// Walks the `miette::Report` error chain looking for a `tonic::Status`. If
/// found, only the gRPC codes that represent transient failures are retryable.
/// If no `tonic::Status` is present (e.g. a raw connection error), assume the
/// failure is transient.
pub fn is_retryable_error(err: &miette::Report) -> bool {
    let mut source: Option<&dyn std::error::Error> = Some(err.as_ref());
    while let Some(e) = source {
        if let Some(status) = e.downcast_ref::<tonic::Status>() {
            return matches!(
                status.code(),
                tonic::Code::Unavailable
                    | tonic::Code::DeadlineExceeded
                    | tonic::Code::ResourceExhausted
                    | tonic::Code::Aborted
                    | tonic::Code::Internal
                    | tonic::Code::Unknown
            );
        }
        source = e.source();
    }
    true
}

/// Retry a gRPC operation with exponential backoff (capped at 4 s).
///
/// Non-transient gRPC errors (e.g. `NOT_FOUND`, `INVALID_ARGUMENT`,
/// `PERMISSION_DENIED`) are returned immediately without retrying.
pub async fn grpc_retry<T, F, Fut>(op_name: &str, f: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_err = None;
    for attempt in 1..=5u32 {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                if !is_retryable_error(&e) {
                    return Err(e);
                }
                if attempt < 5 {
                    warn!(
                        attempt,
                        max_attempts = 5,
                        error = %e,
                        "{op_name} failed, retrying"
                    );
                    let backoff = Duration::from_secs((1u64 << (attempt - 1)).min(4));
                    tokio::time::sleep(backoff).await;
                }
                last_err = Some(e);
            }
        }
    }
    Err(miette::miette!(
        "{op_name} failed after 5 attempts: {}",
        last_err.expect("loop executed at least once")
    ))
}
