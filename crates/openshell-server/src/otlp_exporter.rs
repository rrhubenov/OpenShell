// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP log exporter for streaming gateway logs to an OpenTelemetry Collector.
//!
//! This module provides non-blocking OTLP export of `SandboxLogLine` messages.
//! When configured, all logs (both gateway-generated and sandbox-pushed) are
//! exported to the specified OTLP endpoint in addition to the existing
//! broadcast channels and tail buffers.

use std::borrow::Cow;
use std::time::Duration;

use openshell_core::proto::SandboxLogLine;
use opentelemetry::logs::{LogRecord, Logger, LoggerProvider, Severity};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::logs::BatchLogProcessor;
use opentelemetry_sdk::runtime::Tokio;
use opentelemetry_sdk::Resource;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// OTLP export configuration.
#[derive(Debug, Clone)]
pub struct OtlpConfig {
    /// OTLP gRPC endpoint (e.g., "http://otel-collector:4317").
    pub endpoint: String,
    /// Service name for OTel resource attributes.
    pub service_name: String,
    /// Export batch size.
    pub batch_size: usize,
    /// Export timeout in milliseconds.
    pub timeout_ms: u64,
}

/// Handle to the OTLP export background task.
///
/// Dropping this handle will close the channel and trigger graceful shutdown
/// of the export loop.
pub struct OtlpExportHandle {
    tx: mpsc::Sender<SandboxLogLine>,
}

impl std::fmt::Debug for OtlpExportHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpExportHandle")
            .field("channel_capacity", &self.tx.capacity())
            .finish()
    }
}

impl OtlpExportHandle {
    /// Queue a log for OTLP export.
    ///
    /// This is non-blocking. If the channel is full, the log is dropped
    /// silently to avoid blocking the main logging path.
    pub fn try_export(&self, log: &SandboxLogLine) {
        // Best-effort: drop if channel full
        let _ = self.tx.try_send(log.clone());
    }
}

/// Initialize the OTLP exporter and spawn the background export task.
///
/// Returns `None` if the endpoint is empty or initialization fails.
/// Initialization failures are logged but do not prevent the gateway from starting.
pub fn init_otlp_exporter(config: &OtlpConfig) -> Option<OtlpExportHandle> {
    if config.endpoint.is_empty() {
        return None;
    }

    // Build the OTLP log exporter
    let exporter = match opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(&config.endpoint)
        .with_timeout(Duration::from_millis(config.timeout_ms))
        .build_log_exporter()
    {
        Ok(e) => e,
        Err(e) => {
            error!(
                error = %e,
                endpoint = %config.endpoint,
                "Failed to create OTLP log exporter"
            );
            return None;
        }
    };

    // Build the logger provider with batch processor
    let resource = Resource::new(vec![opentelemetry::KeyValue::new(
        "service.name",
        config.service_name.clone(),
    )]);

    let provider = opentelemetry_sdk::logs::LoggerProvider::builder()
        .with_resource(resource)
        .with_log_processor(
            BatchLogProcessor::builder(exporter, Tokio)
                .build(),
        )
        .build();

    let logger = provider.logger("openshell-gateway");

    // Channel for non-blocking export
    let (tx, rx) = mpsc::channel::<SandboxLogLine>(config.batch_size * 2);

    // Spawn the background export loop
    tokio::spawn(run_export_loop(rx, logger, provider));

    info!(
        endpoint = %config.endpoint,
        service_name = %config.service_name,
        batch_size = config.batch_size,
        "OTLP log export initialized"
    );

    Some(OtlpExportHandle { tx })
}

/// Background task that receives logs from the channel and exports them via OTLP.
async fn run_export_loop(
    mut rx: mpsc::Receiver<SandboxLogLine>,
    logger: opentelemetry_sdk::logs::Logger,
    provider: opentelemetry_sdk::logs::LoggerProvider,
) {
    while let Some(log) = rx.recv().await {
        emit_log_record(&logger, &log);
    }

    // Channel closed (handle dropped) - flush pending logs and shutdown
    if let Err(e) = provider.shutdown() {
        warn!(error = %e, "OTLP provider shutdown error");
    }
}

/// Convert a `SandboxLogLine` to an OTLP `LogRecord` and emit it.
fn emit_log_record(logger: &opentelemetry_sdk::logs::Logger, log: &SandboxLogLine) {
    let severity = match log.level.to_uppercase().as_str() {
        "TRACE" => Severity::Trace,
        "DEBUG" => Severity::Debug,
        "INFO" => Severity::Info,
        "WARN" | "WARNING" => Severity::Warn,
        "ERROR" => Severity::Error,
        _ => Severity::Info,
    };

    // Build attributes from structured log fields
    let mut attributes: Vec<(Cow<'static, str>, opentelemetry::Value)> = vec![
        (Cow::Borrowed("sandbox_id"), log.sandbox_id.clone().into()),
        (Cow::Borrowed("source"), log.source.clone().into()),
        (Cow::Borrowed("target"), log.target.clone().into()),
    ];

    // Add custom fields from the log
    for (key, value) in &log.fields {
        attributes.push((Cow::Owned(key.clone()), value.clone().into()));
    }

    // Build and emit the log record
    let mut builder = logger.create_log_record();
    builder.set_severity_number(severity);
    builder.set_severity_text(Cow::Owned(log.level.clone()));
    builder.set_body(log.message.clone().into());
    builder.set_timestamp(
        std::time::UNIX_EPOCH + Duration::from_millis(log.timestamp_ms.unsigned_abs()),
    );
    builder.add_attributes(attributes);

    logger.emit(builder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_otlp_config_debug() {
        let config = OtlpConfig {
            endpoint: "http://localhost:4317".to_string(),
            service_name: "test-service".to_string(),
            batch_size: 512,
            timeout_ms: 5000,
        };
        // Should not panic
        let _ = format!("{config:?}");
    }

    #[test]
    fn test_empty_endpoint_returns_none() {
        let config = OtlpConfig {
            endpoint: String::new(),
            service_name: "test".to_string(),
            batch_size: 100,
            timeout_ms: 1000,
        };
        assert!(init_otlp_exporter(&config).is_none());
    }

    /// Integration test that sends logs to a real OTLP collector.
    /// Run with: OTLP_TEST_ENDPOINT=http://localhost:4317 cargo test otlp_integration --nocapture
    #[tokio::test]
    async fn test_otlp_integration_with_collector() {
        let endpoint = match std::env::var("OTLP_TEST_ENDPOINT") {
            Ok(e) => e,
            Err(_) => {
                eprintln!("Skipping OTLP integration test: OTLP_TEST_ENDPOINT not set");
                return;
            }
        };

        let config = OtlpConfig {
            endpoint,
            service_name: "openshell-gateway-test".to_string(),
            batch_size: 10,
            timeout_ms: 5000,
        };

        let handle = init_otlp_exporter(&config);
        assert!(handle.is_some(), "Failed to initialize OTLP exporter");
        let handle = handle.unwrap();

        // Send several test logs
        for i in 0..5 {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;

            let log = SandboxLogLine {
                sandbox_id: "test-sandbox-integration".to_string(),
                timestamp_ms: now_ms,
                level: "INFO".to_string(),
                target: "openshell_server::otlp_exporter::tests".to_string(),
                message: format!("OTLP integration test log #{i}"),
                source: "gateway".to_string(),
                fields: {
                    let mut m = HashMap::new();
                    m.insert("test_field".to_string(), format!("value_{i}"));
                    m
                },
            };
            handle.try_export(&log);
        }

        // Give the batch processor time to flush
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        eprintln!("Sent 5 test logs to OTLP collector - check collector output");
    }
}
