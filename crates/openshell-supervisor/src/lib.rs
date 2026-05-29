// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `OpenShell` Sandbox library.
//!
//! This crate provides process sandboxing and monitoring capabilities.

pub mod debug_rpc;
pub use openshell_supervisor_process::{
    bypass_monitor, child_env, log_push, process, provider_credentials, sandbox, skills, ssh,
};
mod supervisor_session;

pub use openshell_supervisor_network::{
    denial_aggregator, grpc_client, identity, l7, mechanistic_mapper, opa, policy_local, proxy,
    secrets,
};

use miette::{IntoDiagnostic, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, trace, warn};

use openshell_ocsf::{
    ActionId, ActivityId, AppLifecycleBuilder, ConfigStateChangeBuilder, DetectionFindingBuilder,
    DispositionId, FindingInfo, LaunchTypeId, Process as OcsfProcess, ProcessActivityBuilder,
    SandboxContext, SeverityId, StateId, StatusId, ocsf_emit,
};

use openshell_core::grpc::AuthedChannel;

// ---------------------------------------------------------------------------
// OCSF Context
// ---------------------------------------------------------------------------
//
// The following log sites intentionally remain as plain `tracing` macros
// and are NOT migrated to OCSF builders:
//
// - DEBUG/TRACE events (zombie reaping, ip commands, gRPC connects, PTY state)
// - Transient "about to do X" events where the result is logged separately
//   (e.g., "Fetching sandbox policy via gRPC", "Creating OPA engine from proto")
// - Internal SSH channel warnings (unknown channel, PTY resize failures)
// - Denial flush telemetry (the individual denials are already OCSF events)
// - Status reporting failures (sync to gateway, non-actionable)
// - Route refresh interval validation warnings
//
// These are operational plumbing that don't represent security decisions,
// policy changes, or observable sandbox behavior worth structuring.
// ---------------------------------------------------------------------------

// The process-wide OCSF context and the agent-proposals feature flag now live
// in `openshell_core::ocsf_ctx`. These re-exports keep the existing call sites
// (`crate::ocsf_ctx()`, `crate::agent_proposals_enabled()`) compiling without
// duplicating the storage.
pub(crate) use openshell_core::ocsf_ctx::{agent_proposals_enabled, ocsf_ctx};

use crate::identity::BinaryIdentityCache;
use crate::l7::tls::{
    CertCache, ProxyTlsState, SandboxCa, build_upstream_client_config, read_system_ca_bundle,
    write_ca_files,
};
use crate::opa::OpaEngine;
use crate::proxy::ProxyHandle;
#[cfg(target_os = "linux")]
use crate::sandbox::linux::netns::NetworkNamespace;
use openshell_core::policy::{NetworkMode, NetworkPolicy, ProxyPolicy, SandboxPolicy};
pub use process::{ProcessHandle, ProcessStatus};
pub use sandbox::apply_supervisor_startup_hardening;

/// Default interval (seconds) for re-fetching the inference route bundle from
/// the gateway in cluster mode. Override at runtime with the
/// `OPENSHELL_ROUTE_REFRESH_INTERVAL_SECS` environment variable.
/// File-based routes (`--inference-routes`) are loaded once at startup and never
/// refreshed.
const DEFAULT_ROUTE_REFRESH_INTERVAL_SECS: u64 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferenceRouteSource {
    File,
    Cluster,
    None,
}

fn infer_route_source(
    sandbox_id: Option<&str>,
    has_gateway: bool,
    inference_routes: Option<&str>,
) -> InferenceRouteSource {
    if inference_routes.is_some() {
        InferenceRouteSource::File
    } else if sandbox_id.is_some() && has_gateway {
        InferenceRouteSource::Cluster
    } else {
        InferenceRouteSource::None
    }
}

fn disable_inference_on_empty_routes(source: InferenceRouteSource) -> bool {
    !matches!(source, InferenceRouteSource::Cluster)
}

fn route_refresh_interval_secs() -> u64 {
    let Ok(value) = std::env::var("OPENSHELL_ROUTE_REFRESH_INTERVAL_SECS") else {
        return DEFAULT_ROUTE_REFRESH_INTERVAL_SECS;
    };
    match value.parse::<u64>() {
        Ok(interval) if interval > 0 => interval,
        Ok(_) => {
            warn!(
                default_interval_secs = DEFAULT_ROUTE_REFRESH_INTERVAL_SECS,
                "Ignoring zero route refresh interval"
            );
            DEFAULT_ROUTE_REFRESH_INTERVAL_SECS
        }
        Err(error) => {
            warn!(
                interval = %value,
                error = %error,
                default_interval_secs = DEFAULT_ROUTE_REFRESH_INTERVAL_SECS,
                "Ignoring invalid route refresh interval"
            );
            DEFAULT_ROUTE_REFRESH_INTERVAL_SECS
        }
    }
}

#[cfg(target_os = "linux")]
fn is_managed_child(pid: i32) -> bool {
    openshell_supervisor_process::managed_children::is_managed_child(pid)
}

/// One of the supervisor's component subsystems.
///
/// `Network` covers the egress proxy, OPA engine, TLS termination, bypass
/// monitor, and denial aggregator. `Process` covers the entrypoint child,
/// SSH server, supervisor session, zombie reaper, filesystem prep, and
/// seccomp prelude. Shared infrastructure — OCSF context, policy load,
/// provider env fetch, network namespace creation, and the policy poll
/// loop — runs regardless of which components are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SupervisorComponent {
    Network,
    Process,
}

impl SupervisorComponent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::Process => "process",
        }
    }
}

/// The set of supervisor components running in this process.
///
/// Constructed from the `--mode` flag (CSV of component names). The default
/// — `network,process` — preserves today's combined `openshell-sandbox`
/// behavior; restricting to a single component lets the network-side proxy
/// and the process-side child supervisor run as independent processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisorMode {
    components: std::collections::BTreeSet<SupervisorComponent>,
}

impl SupervisorMode {
    pub const DEFAULT_STR: &'static str = "network,process";

    pub fn has(&self, component: SupervisorComponent) -> bool {
        self.components.contains(&component)
    }
}

impl std::fmt::Display for SupervisorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for component in &self.components {
            if !first {
                f.write_str(",")?;
            }
            f.write_str(component.as_str())?;
            first = false;
        }
        Ok(())
    }
}

impl std::str::FromStr for SupervisorMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let mut components = std::collections::BTreeSet::new();
        let mut empty = true;
        for part in s.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            empty = false;
            match trimmed {
                "network" => {
                    components.insert(SupervisorComponent::Network);
                }
                "process" => {
                    components.insert(SupervisorComponent::Process);
                }
                other => {
                    return Err(format!(
                        "invalid supervisor mode component {other:?}; expected \"network\" or \"process\""
                    ));
                }
            }
        }
        if empty {
            return Err("supervisor mode must not be empty".into());
        }
        Ok(Self { components })
    }
}

/// Block until a typical container shutdown signal arrives.
///
/// Network-only mode has no entrypoint child to wait on, so the supervisor's
/// foreground task must instead keep running — and keep the proxy / poll
/// loop / aggregator alive — until the orchestrator asks it to stop.
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        // Either signal terminates a typical container PID; await whichever
        // arrives first. Errors registering the handler degrade gracefully
        // by waiting on whatever channel did succeed.
        let term = signal(SignalKind::terminate()).ok();
        let int = signal(SignalKind::interrupt()).ok();
        match (term, int) {
            (Some(mut term), Some(mut int)) => {
                tokio::select! {
                    _ = term.recv() => {}
                    _ = int.recv() => {}
                }
            }
            (Some(mut term), None) => {
                term.recv().await;
            }
            (None, Some(mut int)) => {
                int.recv().await;
            }
            (None, None) => {
                // Last resort: park the task so the runtime keeps the
                // background components alive instead of returning.
                std::future::pending::<()>().await;
            }
        }
    }

    #[cfg(not(unix))]
    {
        // Non-unix builds use ctrl_c only.
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Run a command in the sandbox.
///
/// # Errors
///
/// Returns an error if the command fails to start or encounters a fatal error.
/// Filesystem paths to the ephemeral CA cert and combined trust bundle.
type CaFilePaths = (std::path::PathBuf, std::path::PathBuf);

/// State produced by `setup_shared` and consumed by every downstream phase.
struct SharedContext {
    policy: SandboxPolicy,
    opa_engine: Option<Arc<OpaEngine>>,
    retained_proto: Option<openshell_core::proto::SandboxPolicy>,
    policy_local_ctx: Arc<policy_local::PolicyLocalContext>,
    provider_credentials: provider_credentials::ProviderCredentialState,
    #[cfg(target_os = "linux")]
    netns: Option<NetworkNamespace>,
    entrypoint_pid: Arc<AtomicU32>,
    sandbox_name_for_agg: Option<String>,
    sandbox_id: Option<String>,
    gateway_channel: Option<AuthedChannel>,
}

/// Handles produced by `setup_network`. Held by `run_sandbox` for the
/// supervisor's lifetime so the proxy and bypass monitor stay alive.
struct NetworkContext {
    _proxy: Option<ProxyHandle>,
    #[cfg(target_os = "linux")]
    _bypass_monitor: Option<tokio::task::JoinHandle<()>>,
    denial_rx: Option<tokio::sync::mpsc::UnboundedReceiver<openshell_core::DenialEvent>>,
    ssh_proxy_url: Option<String>,
    ca_file_paths: Option<CaFilePaths>,
    #[cfg(target_os = "linux")]
    ssh_netns_fd: Option<i32>,
}

impl NetworkContext {
    fn take_denial_rx(
        &mut self,
    ) -> Option<tokio::sync::mpsc::UnboundedReceiver<openshell_core::DenialEvent>> {
        self.denial_rx.take()
    }
}

/// Handle to the entrypoint child process. Held by `run_sandbox` so it can
/// `wait()` (or `kill()` on timeout) before returning the exit code.
struct ProcessContext {
    handle: ProcessHandle,
}

/// Run a command in the sandbox.
///
/// # Errors
///
/// Returns an error if the command fails to start or encounters a fatal error.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub async fn run_sandbox(
    command: Vec<String>,
    workdir: Option<String>,
    timeout_secs: u64,
    interactive: bool,
    sandbox_id: Option<String>,
    sandbox: Option<String>,
    policy_rules: Option<String>,
    policy_data: Option<String>,
    ssh_socket_path: Option<String>,
    _health_check: bool,
    _health_port: u16,
    inference_routes: Option<String>,
    mode: SupervisorMode,
    ocsf_enabled: Arc<std::sync::atomic::AtomicBool>,
    gateway_channel: Option<AuthedChannel>,
) -> Result<i32> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| miette::miette!("No command specified"))?;

    info!(mode = %mode, "Supervisor mode");

    let shared = setup_shared(
        sandbox_id,
        sandbox,
        gateway_channel,
        policy_rules,
        policy_data,
    )
    .await?;

    let mut network = if mode.has(SupervisorComponent::Network) {
        Some(setup_network(&shared, inference_routes.as_deref()).await?)
    } else {
        None
    };

    let process = if mode.has(SupervisorComponent::Process) {
        let ssh_proxy_url = network.as_ref().and_then(|n| n.ssh_proxy_url.clone());
        let ca_paths = network.as_ref().and_then(|n| n.ca_file_paths.clone());
        #[cfg(target_os = "linux")]
        let ssh_netns_fd = network.as_ref().and_then(|n| n.ssh_netns_fd);
        #[cfg(not(target_os = "linux"))]
        let ssh_netns_fd: Option<i32> = None;
        Some(
            setup_process(
                &shared,
                program,
                args,
                workdir.as_deref(),
                interactive,
                ssh_proxy_url,
                ca_paths,
                ssh_socket_path.map(std::path::PathBuf::from),
                ssh_netns_fd,
            )
            .await?,
        )
    } else {
        None
    };

    let denial_rx = network.as_mut().and_then(NetworkContext::take_denial_rx);
    spawn_policy_poll_loop(&shared, ocsf_enabled, denial_rx);

    run_until_shutdown(process, timeout_secs).await
}

/// Initialize OCSF context, load policy, fetch provider credentials, create
/// the network namespace, and prepare the proposals flag. The returned
/// `SharedContext` drives every downstream phase.
async fn setup_shared(
    sandbox_id: Option<String>,
    sandbox: Option<String>,
    gateway_channel: Option<AuthedChannel>,
    policy_rules: Option<String>,
    policy_data: Option<String>,
) -> Result<SharedContext> {
    // Initialize the process-wide OCSF context early so that events emitted
    // during policy loading (filesystem config, validation) have a context.
    // Proxy IP/port use defaults here; they are only significant for network
    // events which happen after the netns is created.
    {
        let hostname = std::fs::read_to_string("/etc/hostname").map_or_else(
            |_| "openshell-sandbox".to_string(),
            |s| s.trim().to_string(),
        );

        if openshell_core::ocsf_ctx::init_ocsf_ctx(SandboxContext {
            sandbox_id: sandbox_id.clone().unwrap_or_default(),
            sandbox_name: sandbox.as_deref().unwrap_or_default().to_string(),
            container_image: std::env::var("OPENSHELL_CONTAINER_IMAGE").unwrap_or_default(),
            hostname,
            product_version: openshell_core::VERSION.to_string(),
            proxy_ip: std::net::IpAddr::from([127, 0, 0, 1]),
            proxy_port: 3128,
        })
        .is_err()
        {
            debug!("OCSF context already initialized, keeping existing");
        }
    }

    // The single authenticated gateway channel comes from `main` so log push,
    // run_sandbox, and every spawned subsystem share one TCP/TLS connection,
    // one token slot, and one renewal loop.

    // Load policy and initialize OPA engine
    let sandbox_name_for_agg = sandbox.clone();
    let (policy, opa_engine, retained_proto) = load_policy(
        sandbox_id.clone(),
        sandbox,
        gateway_channel.clone(),
        policy_rules,
        policy_data,
    )
    .await?;
    let policy_local_ctx = Arc::new(policy_local::PolicyLocalContext::new(
        retained_proto.clone(),
        gateway_channel.clone(),
        sandbox_name_for_agg.clone().or_else(|| sandbox_id.clone()),
    ));

    // Fetch provider environment variables from the server.
    // This is done after loading the policy so the sandbox can still start
    // even if provider env fetch fails (graceful degradation).
    let (provider_env_revision, provider_env, provider_credential_expires_at_ms) =
        if let (Some(id), Some(channel)) = (&sandbox_id, gateway_channel.as_ref()) {
            let client = grpc_client::CachedOpenShellClient::new(channel.clone());
            match client.fetch_provider_environment(id).await {
                Ok(result) => {
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Informational)
                            .status(StatusId::Success)
                            .state(StateId::Enabled, "loaded")
                            .message(format!(
                                "Fetched provider environment [env_count:{}]",
                                result.environment.len()
                            ))
                            .build()
                    );
                    (
                        result.provider_env_revision,
                        result.environment,
                        result.credential_expires_at_ms,
                    )
                }
                Err(e) => {
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Medium)
                            .status(StatusId::Failure)
                            .state(StateId::Other, "degraded")
                            .message(format!(
                                "Failed to fetch provider environment, continuing without: {e}"
                            ))
                            .build()
                    );
                    (
                        0,
                        std::collections::HashMap::new(),
                        std::collections::HashMap::new(),
                    )
                }
            }
        } else {
            (
                0,
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
            )
        };

    let provider_credentials = provider_credentials::ProviderCredentialState::from_environment(
        provider_env_revision,
        provider_env,
        provider_credential_expires_at_ms,
    );
    // NOTE: This snapshot is taken once at supervisor startup. SSH-spawned
    // children see this frozen view for the entire supervisor lifetime and
    // will NOT observe credential rotations from the settings poll loop.
    // The proxy still sees rotations live via resolver(); only direct env
    // reads inside SSH children are stale.
    // Alternative is a shared RWLock around the state.

    // Initialize the agent-proposals feature flag. Default false until the
    // initial settings fetch (or the poll loop) tells us otherwise. The flag
    // gates the skill install, the policy.local route handler, and the L7
    // deny body's `next_steps` field — see `agent_proposals_enabled()`.
    let proposals_enabled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    if openshell_core::ocsf_ctx::init_agent_proposals_enabled(proposals_enabled.clone()).is_err() {
        debug!("agent proposals flag already initialized, keeping existing");
    }

    // Eagerly fetch the initial settings so skill install can honor the flag
    // at startup rather than waiting for the poll loop's first tick. In
    // offline/file-mode there is no gateway, so the flag stays false.
    if let (Some(id), Some(channel)) = (&sandbox_id, gateway_channel.as_ref()) {
        let client = grpc_client::CachedOpenShellClient::new(channel.clone());
        if let Ok(result) = client.poll_settings(id).await {
            let initial = extract_bool_setting(
                &result.settings,
                openshell_core::settings::AGENT_POLICY_PROPOSALS_ENABLED_KEY,
            )
            .unwrap_or(false);
            proposals_enabled.store(initial, Ordering::Relaxed);
        }
    }

    // Create network namespace for proxy mode (Linux only)
    // This must be created before the proxy AND SSH server so that SSH
    // sessions can enter the namespace for network isolation.
    #[cfg(target_os = "linux")]
    let netns = if matches!(policy.network.mode, NetworkMode::Proxy) {
        match NetworkNamespace::create() {
            Ok(ns) => {
                // Install bypass detection rules (nftables log + reject).
                // This provides fast-fail UX and diagnostic logging for direct
                // connection attempts that bypass the HTTP CONNECT proxy.
                let proxy_port = policy
                    .network
                    .proxy
                    .as_ref()
                    .and_then(|p| p.http_addr)
                    .map_or(3128, |addr| addr.port());
                if let Err(e) = ns.install_bypass_rules(proxy_port) {
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Medium)
                            .status(StatusId::Failure)
                            .state(StateId::Disabled, "degraded")
                            .message(format!(
                                "Failed to install bypass detection rules (non-fatal): {e}"
                            ))
                            .build()
                    );
                }
                Some(ns)
            }
            Err(e) => {
                return Err(miette::miette!(
                    "Network namespace creation failed and proxy mode requires isolation. \
                     Ensure CAP_NET_ADMIN and CAP_SYS_ADMIN are available and iproute2 is installed. \
                     Error: {e}"
                ));
            }
        }
    } else {
        None
    };

    // Shared PID: set after process spawn so the proxy can look up
    // the entrypoint process's /proc/net/tcp for identity binding.
    let entrypoint_pid = Arc::new(AtomicU32::new(0));

    Ok(SharedContext {
        policy,
        opa_engine,
        retained_proto,
        policy_local_ctx,
        provider_credentials,
        #[cfg(target_os = "linux")]
        netns,
        entrypoint_pid,
        sandbox_name_for_agg,
        sandbox_id,
        gateway_channel,
    })
}

/// Start the egress proxy stack: identity cache, TLS state, the CONNECT
/// proxy itself, the bypass monitor (Linux only), and the SSH-side
/// rendezvous values (proxy URL, CA paths, netns fd). Skipped entirely when
/// the policy is not in proxy mode.
async fn setup_network(
    shared: &SharedContext,
    inference_routes: Option<&str>,
) -> Result<NetworkContext> {
    if !matches!(shared.policy.network.mode, NetworkMode::Proxy) {
        // Non-proxy network mode still satisfies the `--mode=network`
        // selector but has no proxy/TLS/bypass plumbing to spin up.
        return Ok(NetworkContext {
            _proxy: None,
            #[cfg(target_os = "linux")]
            _bypass_monitor: None,
            denial_rx: None,
            ssh_proxy_url: None,
            ca_file_paths: None,
            #[cfg(target_os = "linux")]
            ssh_netns_fd: None,
        });
    }

    // Create identity cache for SHA256 TOFU when OPA is active. Only the
    // proxy consumes the cache, so we only allocate it here.
    let identity_cache = shared
        .opa_engine
        .as_ref()
        .map(|_| Arc::new(BinaryIdentityCache::new()));

    // Generate ephemeral CA and TLS state for HTTPS L7 inspection.
    // The CA cert is written to disk so sandbox processes can trust it.
    let (tls_state, ca_file_paths) = match SandboxCa::generate() {
        Ok(ca) => {
            let tls_dir = std::path::Path::new("/etc/openshell-tls");
            let system_ca_bundle = read_system_ca_bundle();
            match write_ca_files(&ca, tls_dir, &system_ca_bundle) {
                Ok(paths) => {
                    // /etc/openshell-tls is subsumed by the /etc baseline
                    // path injected by enrich_*_baseline_paths(), so no
                    // explicit Landlock entry is needed here.
                    let upstream_config = build_upstream_client_config(&system_ca_bundle);
                    let cert_cache = CertCache::new(ca);
                    let state = Arc::new(ProxyTlsState::new(cert_cache, upstream_config));
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Informational)
                            .status(StatusId::Success)
                            .state(StateId::Enabled, "enabled")
                            .message("TLS termination enabled: ephemeral CA generated")
                            .build()
                    );
                    (Some(state), Some(paths))
                }
                Err(e) => {
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Medium)
                            .status(StatusId::Failure)
                            .state(StateId::Disabled, "disabled")
                            .message(format!(
                                "Failed to write CA files, TLS termination disabled: {e}"
                            ))
                            .build()
                    );
                    (None, None)
                }
            }
        }
        Err(e) => {
            ocsf_emit!(
                ConfigStateChangeBuilder::new(ocsf_ctx())
                    .severity(SeverityId::Medium)
                    .status(StatusId::Failure)
                    .state(StateId::Disabled, "disabled")
                    .message(format!(
                        "Failed to generate ephemeral CA, TLS termination disabled: {e}"
                    ))
                    .build()
            );
            (None, None)
        }
    };

    let proxy_policy = shared.policy.network.proxy.as_ref().ok_or_else(|| {
        miette::miette!("Network mode is set to proxy but no proxy configuration was provided")
    })?;

    let engine = shared.opa_engine.clone().ok_or_else(|| {
        miette::miette!("Proxy mode requires an OPA engine (--rego-policy and --rego-data)")
    })?;

    let cache = identity_cache.ok_or_else(|| {
        miette::miette!("Proxy mode requires an identity cache (OPA engine must be configured)")
    })?;

    // If we have a network namespace, bind to the veth host IP so sandboxed
    // processes can reach the proxy via TCP.
    #[cfg(target_os = "linux")]
    let bind_addr = shared.netns.as_ref().map(|ns| {
        let port = proxy_policy.http_addr.map_or(3128, |addr| addr.port());
        SocketAddr::new(ns.host_ip(), port)
    });

    #[cfg(not(target_os = "linux"))]
    let bind_addr: Option<SocketAddr> = None;

    // Build inference context for local routing of intercepted inference calls.
    let inference_ctx = build_inference_context(
        shared.sandbox_id.as_deref(),
        shared.gateway_channel.as_ref(),
        inference_routes,
    )
    .await?;

    // Create denial aggregator channel if in gRPC mode (sandbox_id present).
    // Clone the sender for the bypass monitor before passing to the proxy.
    let (denial_tx, denial_rx, bypass_denial_tx) = if shared.sandbox_id.is_some() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let bypass_tx = tx.clone();
        (Some(tx), Some(rx), Some(bypass_tx))
    } else {
        (None, None, None)
    };

    let proxy_handle = ProxyHandle::start_with_bind_addr(
        proxy_policy,
        bind_addr,
        engine,
        cache,
        shared.entrypoint_pid.clone(),
        tls_state,
        inference_ctx,
        shared.provider_credentials.resolver(),
        Some(shared.policy_local_ctx.clone()),
        denial_tx,
    )
    .await?;

    // Spawn bypass detection monitor (Linux only). Reads /dev/kmsg for
    // nftables log entries and emits structured tracing events for direct
    // connection attempts that bypass the proxy. Bypass denials feed the
    // denial aggregator, so this only runs alongside the proxy.
    #[cfg(target_os = "linux")]
    let bypass_monitor = shared.netns.as_ref().and_then(|ns| {
        bypass_monitor::spawn(
            ns.name().to_string(),
            shared.entrypoint_pid.clone(),
            bypass_denial_tx,
        )
    });

    // On non-Linux, bypass_denial_tx is unused (no /dev/kmsg).
    #[cfg(not(target_os = "linux"))]
    drop(bypass_denial_tx);

    // Compute the proxy URL and netns fd for SSH sessions.
    // SSH shell processes need both to enforce network policy:
    // - netns_fd: enter the network namespace via setns() so all traffic
    //   goes through the veth pair (hard enforcement, non-bypassable)
    // - proxy_url: set proxy env vars so cooperative tools route through the
    //   CONNECT proxy; this also opts Node.js into honoring those vars
    #[cfg(target_os = "linux")]
    let ssh_netns_fd = shared.netns.as_ref().and_then(NetworkNamespace::ns_fd);

    #[cfg(target_os = "linux")]
    let ssh_proxy_url = shared.netns.as_ref().map(|ns| {
        let port = shared
            .policy
            .network
            .proxy
            .as_ref()
            .and_then(|p| p.http_addr)
            .map_or(3128, |addr| addr.port());
        format!("http://{}:{port}", ns.host_ip())
    });

    #[cfg(not(target_os = "linux"))]
    let ssh_proxy_url = shared
        .policy
        .network
        .proxy
        .as_ref()
        .and_then(|p| p.http_addr)
        .map(|addr| format!("http://{addr}"));

    Ok(NetworkContext {
        _proxy: Some(proxy_handle),
        #[cfg(target_os = "linux")]
        _bypass_monitor: bypass_monitor,
        denial_rx,
        ssh_proxy_url,
        ca_file_paths,
        #[cfg(target_os = "linux")]
        ssh_netns_fd,
    })
}

/// Bring up the entrypoint side: validate the sandbox user, prepare the
/// filesystem, install agent skills, apply seccomp, spawn the zombie reaper
/// and SSH server, start the supervisor session, and finally fork the
/// entrypoint child.
#[allow(clippy::too_many_arguments)]
async fn setup_process(
    shared: &SharedContext,
    program: &str,
    args: &[String],
    workdir: Option<&str>,
    interactive: bool,
    ssh_proxy_url: Option<String>,
    ca_file_paths: Option<CaFilePaths>,
    ssh_socket_path: Option<std::path::PathBuf>,
    #[cfg_attr(not(target_os = "linux"), allow(unused_variables))] ssh_netns_fd: Option<i32>,
) -> Result<ProcessContext> {
    // Validate that the required "sandbox" user exists in this image.
    // All sandbox images must include this user for privilege dropping.
    #[cfg(unix)]
    validate_sandbox_user(&shared.policy)?;

    // Prepare filesystem: create and chown read_write directories.
    prepare_filesystem(&shared.policy)?;

    // The skill files are read by the entrypoint child, so install them
    // before the child is spawned.
    if agent_proposals_enabled() {
        match skills::install_static_skills() {
            Ok(installed) => {
                info!(
                    path = %installed.policy_advisor.display(),
                    "Installed sandbox agent skill"
                );
            }
            Err(error) => {
                warn!(error = %error, "Failed to install sandbox agent skill");
            }
        }
    } else {
        debug!("agent_policy_proposals_enabled is false at startup; skipping skill install");
    }

    // Install the supervisor seccomp prelude after privileged startup helpers
    // (network namespace setup, nftables probes) complete, but before the SSH
    // listener and workload process are exposed.
    apply_supervisor_startup_hardening()?;

    // Zombie reaper — openshell-sandbox may run as PID 1 in containers and
    // must reap orphaned grandchildren (e.g. background daemons started by
    // coding agents) to prevent zombie accumulation.
    //
    // Use waitid(..., WNOWAIT) so we can inspect exited children before
    // actually reaping them. This avoids racing explicit `child.wait()` calls
    // for managed children (entrypoint and SSH session processes).
    #[cfg(target_os = "linux")]
    tokio::spawn(async {
        use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid, waitpid};
        use tokio::signal::unix::{SignalKind, signal};
        use tokio::time::MissedTickBehavior;

        let mut sigchld = match signal(SignalKind::child()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to register SIGCHLD handler for zombie reaping");
                return;
            }
        };
        let mut retry = tokio::time::interval(Duration::from_secs(5));
        retry.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = sigchld.recv() => {}
                _ = retry.tick() => {}
            }

            loop {
                let status = match waitid(
                    Id::All,
                    WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
                ) {
                    Ok(WaitStatus::StillAlive) | Err(nix::errno::Errno::ECHILD) => break,
                    Ok(status) => status,
                    Err(nix::errno::Errno::EINTR) => continue,
                    Err(e) => {
                        tracing::debug!(error = %e, "waitid error during zombie reaping");
                        break;
                    }
                };

                let Some(pid) = status.pid() else {
                    break;
                };

                if is_managed_child(pid.as_raw()) {
                    // Let the explicit waiter own this child status.
                    break;
                }

                match waitpid(pid, Some(WaitPidFlag::WNOHANG)) {
                    Ok(WaitStatus::StillAlive)
                    | Err(nix::errno::Errno::ECHILD | nix::errno::Errno::EINTR) => {}
                    Ok(reaped) => {
                        tracing::debug!(?reaped, "Reaped orphaned child process");
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, "waitpid error during orphan reap");
                        break;
                    }
                }
            }
        }
    });

    if let Some(listen_path) = ssh_socket_path.clone() {
        let policy_clone = shared.policy.clone();
        let workdir_clone = workdir.map(str::to_owned);
        let proxy_url = ssh_proxy_url.clone();
        #[cfg(target_os = "linux")]
        let netns_fd = ssh_netns_fd;
        #[cfg(not(target_os = "linux"))]
        let netns_fd: Option<i32> = None;
        let ca_paths = ca_file_paths.clone();
        let ssh_provider_env = shared.provider_credentials.snapshot().child_env.clone();

        let (ssh_ready_tx, ssh_ready_rx) = tokio::sync::oneshot::channel();

        tokio::spawn(async move {
            if let Err(err) = ssh::run_ssh_server(
                listen_path,
                ssh_ready_tx,
                policy_clone,
                workdir_clone,
                netns_fd,
                proxy_url,
                ca_paths,
                ssh_provider_env,
            )
            .await
            {
                ocsf_emit!(
                    AppLifecycleBuilder::new(ocsf_ctx())
                        .activity(ActivityId::Fail)
                        .severity(SeverityId::Critical)
                        .status(StatusId::Failure)
                        .message(format!("SSH server failed: {err}"))
                        .build()
                );
            }
        });

        // Wait for the SSH server to bind its socket before spawning the
        // entrypoint process. This prevents exec requests from racing against
        // SSH server startup when Kubernetes marks the pod Ready.
        match timeout(Duration::from_secs(10), ssh_ready_rx).await {
            Ok(Ok(Ok(()))) => {
                ocsf_emit!(
                    AppLifecycleBuilder::new(ocsf_ctx())
                        .activity(ActivityId::Open)
                        .severity(SeverityId::Informational)
                        .status(StatusId::Success)
                        .message("SSH server is ready to accept connections")
                        .build()
                );
            }
            Ok(Ok(Err(err))) => {
                return Err(err.context("SSH server failed during startup"));
            }
            Ok(Err(_)) => {
                return Err(miette::miette!(
                    "SSH server task panicked before signaling ready"
                ));
            }
            Err(_) => {
                return Err(miette::miette!(
                    "SSH server did not start within 10 seconds"
                ));
            }
        }
    }

    // Spawn the persistent supervisor session if we have a gateway endpoint
    // and sandbox identity. The session provides relay channels for SSH
    // connect and ExecSandbox through the gateway.
    if let (Some(channel), Some(id), Some(socket)) = (
        shared.gateway_channel.as_ref(),
        shared.sandbox_id.as_ref(),
        ssh_socket_path.as_ref(),
    ) {
        #[cfg(target_os = "linux")]
        let netns_fd = ssh_netns_fd;
        #[cfg(not(target_os = "linux"))]
        let netns_fd: Option<i32> = None;
        supervisor_session::spawn(channel.clone(), id.clone(), socket.clone(), netns_fd);
        info!("supervisor session task spawned");
    }

    // Spawn the entrypoint child.
    let provider_env = shared.provider_credentials.snapshot().child_env.clone();
    #[cfg(target_os = "linux")]
    let h = ProcessHandle::spawn(
        program,
        args,
        workdir,
        interactive,
        &shared.policy,
        shared.netns.as_ref(),
        ca_file_paths.as_ref(),
        &provider_env,
    )?;

    #[cfg(not(target_os = "linux"))]
    let h = ProcessHandle::spawn(
        program,
        args,
        workdir,
        interactive,
        &shared.policy,
        ca_file_paths.as_ref(),
        &provider_env,
    )?;

    // Store the entrypoint PID so the proxy can resolve TCP peer identity.
    shared.entrypoint_pid.store(h.pid(), Ordering::Release);
    ocsf_emit!(
        ProcessActivityBuilder::new(ocsf_ctx())
            .activity(ActivityId::Open)
            .action(ActionId::Allowed)
            .disposition(DispositionId::Allowed)
            .severity(SeverityId::Informational)
            .status(StatusId::Success)
            .launch_type(LaunchTypeId::Spawn)
            .process(OcsfProcess::new(program, i64::from(h.pid())))
            .message(format!("Process started: pid={}", h.pid()))
            .build()
    );

    // Spawn a task to resolve policy binary symlinks after the container
    // filesystem becomes accessible via /proc/<pid>/root/. This expands
    // symlinks like /usr/bin/python3 → /usr/bin/python3.11 in the OPA
    // policy data so that either path matches at evaluation time.
    //
    // We cannot do this synchronously here because the child process has
    // just been spawned and its mount namespace / procfs entries may not
    // be fully populated yet. Instead, we probe with retries until
    // /proc/<pid>/root/ is accessible or we exhaust attempts.
    if let (Some(engine), Some(proto)) = (&shared.opa_engine, &shared.retained_proto) {
        let resolve_engine = engine.clone();
        let resolve_proto = proto.clone();
        let resolve_pid = shared.entrypoint_pid.clone();
        tokio::spawn(async move {
            let pid = resolve_pid.load(Ordering::Acquire);
            let probe_path = format!("/proc/{pid}/root/");
            // Retry up to 10 times with 500ms intervals (5s total).
            // The child's mount namespace is typically ready within a
            // few hundred ms of spawn.
            for attempt in 1..=10 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if std::fs::metadata(&probe_path).is_ok() {
                    info!(
                        pid = pid,
                        attempt = attempt,
                        "Container filesystem accessible, resolving policy binary symlinks"
                    );
                    match resolve_engine.reload_from_proto_with_pid(&resolve_proto, pid) {
                        Ok(()) => {
                            info!(
                                pid = pid,
                                "Policy binary symlink resolution complete \
                                 (check logs above for per-binary results)"
                            );
                        }
                        Err(e) => {
                            warn!(
                                "Failed to rebuild OPA engine with symlink resolution \
                                 (non-fatal, falling back to literal path matching): {e}"
                            );
                        }
                    }
                    return;
                }
                debug!(
                    pid = pid,
                    attempt = attempt,
                    probe_path = %probe_path,
                    "Container filesystem not yet accessible, retrying symlink resolution"
                );
            }
            warn!(
                "Container filesystem /proc/{pid}/root/ not accessible after 10 attempts (5s); \
                 binary symlink resolution skipped. Policy binary paths will be matched literally. \
                 If binaries are symlinks, use canonical paths in your policy \
                 (run 'readlink -f <path>' inside the sandbox)"
            );
        });
    }

    Ok(ProcessContext { handle: h })
}

/// Spawn the background policy poll task, and — when a denial channel from
/// the proxy is available — the denial aggregator that flushes summaries to
/// the gateway. Both run in any mode whose inputs (sandbox id, gateway
/// channel, OPA engine) are present.
#[allow(clippy::similar_names)]
fn spawn_policy_poll_loop(
    shared: &SharedContext,
    ocsf_enabled: Arc<std::sync::atomic::AtomicBool>,
    denial_rx: Option<tokio::sync::mpsc::UnboundedReceiver<openshell_core::DenialEvent>>,
) {
    let Some(((id, channel), engine)) = shared
        .sandbox_id
        .as_ref()
        .zip(shared.gateway_channel.as_ref())
        .zip(shared.opa_engine.as_ref())
    else {
        return;
    };

    let poll_id = id.clone();
    let poll_channel = channel.clone();
    let poll_engine = engine.clone();
    let poll_ocsf_enabled = ocsf_enabled;
    let poll_pid = shared.entrypoint_pid.clone();
    let poll_provider_credentials = shared.provider_credentials.clone();
    let poll_policy_local = shared.policy_local_ctx.clone();
    let poll_interval_secs: u64 = std::env::var("OPENSHELL_POLICY_POLL_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let poll_ctx = PolicyPollLoopContext {
        channel: poll_channel,
        sandbox_id: poll_id,
        opa_engine: poll_engine,
        entrypoint_pid: poll_pid,
        interval_secs: poll_interval_secs,
        ocsf_enabled: poll_ocsf_enabled,
        provider_credentials: poll_provider_credentials,
        policy_local_ctx: Some(poll_policy_local),
    };

    tokio::spawn(async move {
        if let Err(e) = run_policy_poll_loop(poll_ctx).await {
            ocsf_emit!(
                AppLifecycleBuilder::new(ocsf_ctx())
                    .activity(ActivityId::Fail)
                    .severity(SeverityId::Medium)
                    .status(StatusId::Failure)
                    .message(format!("Policy poll loop exited with error: {e}"))
                    .build()
            );
        }
    });

    // Spawn denial aggregator only when the proxy created a denial channel.
    // The aggregator drains denial events from the proxy, so it only makes
    // sense alongside the proxy itself.
    if let Some(rx) = denial_rx {
        // SubmitPolicyAnalysis resolves by sandbox *name*, not UUID.
        let agg_name = shared
            .sandbox_name_for_agg
            .clone()
            .unwrap_or_else(|| id.clone());
        let agg_channel = channel.clone();
        let flush_interval_secs: u64 = std::env::var("OPENSHELL_DENIAL_FLUSH_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        let aggregator = denial_aggregator::DenialAggregator::new(rx, flush_interval_secs);

        tokio::spawn(async move {
            aggregator
                .run(|summaries| {
                    let channel = agg_channel.clone();
                    let sandbox_name = agg_name.clone();
                    async move {
                        if let Err(e) =
                            flush_proposals_to_gateway(channel, &sandbox_name, summaries).await
                        {
                            warn!(error = %e, "Failed to flush denial summaries to gateway");
                        }
                    }
                })
                .await;
        });
    }
}

/// Wait for the entrypoint to exit (with optional timeout) and return its
/// exit code, or — when no entrypoint was spawned (Network-only mode) —
/// block on a shutdown signal so the proxy and poll loop stay alive.
async fn run_until_shutdown(process: Option<ProcessContext>, timeout_secs: u64) -> Result<i32> {
    let Some(ProcessContext { handle: mut h }) = process else {
        wait_for_shutdown_signal().await;
        return Ok(0);
    };

    let result = if timeout_secs > 0 {
        if let Ok(result) = timeout(Duration::from_secs(timeout_secs), h.wait()).await {
            result
        } else {
            ocsf_emit!(
                ProcessActivityBuilder::new(ocsf_ctx())
                    .activity(ActivityId::Close)
                    .action(ActionId::Denied)
                    .disposition(DispositionId::Blocked)
                    .severity(SeverityId::Critical)
                    .status(StatusId::Failure)
                    .message("Process timed out, killing")
                    .build()
            );
            h.kill()?;
            return Ok(124); // Standard timeout exit code
        }
    } else {
        h.wait().await
    };

    let status = result.into_diagnostic()?;

    ocsf_emit!(
        ProcessActivityBuilder::new(ocsf_ctx())
            .activity(ActivityId::Close)
            .action(ActionId::Allowed)
            .disposition(DispositionId::Allowed)
            .severity(SeverityId::Informational)
            .status(StatusId::Success)
            .exit_code(status.code())
            .message(format!("Process exited with code {}", status.code()))
            .build()
    );

    Ok(status.code())
}

/// Build an inference context for local routing, if route sources are available.
///
/// Route sources (in priority order):
/// 1. Inference routes file (standalone mode) — always takes precedence
/// 2. Cluster bundle (fetched from gateway via gRPC)
///
/// If both a routes file and cluster credentials are provided, the routes file
/// wins and the cluster bundle is not fetched.
///
/// Returns `None` if neither source is configured (inference routing disabled).
// `routes`/`router` are intentionally distinct nouns (the route list vs the
// router that consumes them); both names are clearer than alternatives.
#[allow(clippy::similar_names)]
async fn build_inference_context(
    sandbox_id: Option<&str>,
    gateway_channel: Option<&AuthedChannel>,
    inference_routes: Option<&str>,
) -> Result<Option<Arc<proxy::InferenceContext>>> {
    use openshell_router::Router;
    use openshell_router::config::RouterConfig;

    let source = infer_route_source(sandbox_id, gateway_channel.is_some(), inference_routes);

    // Captured during the initial cluster bundle fetch so the background refresh
    // loop can skip no-op updates from the very first tick.
    let mut initial_revision: Option<String> = None;

    let routes = match source {
        InferenceRouteSource::File => {
            let Some(path) = inference_routes else {
                return Ok(None);
            };

            // Standalone mode: load routes from file (fail-fast on errors)
            if sandbox_id.is_some() {
                ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                    .severity(SeverityId::Informational)
                    .status(StatusId::Success)
                    .state(StateId::Enabled, "loaded")
                    .unmapped("inference_routes", serde_json::json!(path))
                    .message(format!(
                        "Inference routes file takes precedence over cluster bundle [path:{path}]"
                    ))
                    .build());
            }
            ocsf_emit!(
                ConfigStateChangeBuilder::new(ocsf_ctx())
                    .severity(SeverityId::Informational)
                    .status(StatusId::Success)
                    .state(StateId::Other, "loading")
                    .unmapped("inference_routes", serde_json::json!(path))
                    .message(format!("Loading inference routes from file [path:{path}]"))
                    .build()
            );
            let config = RouterConfig::load_from_file(std::path::Path::new(path))
                .map_err(|e| miette::miette!("failed to load inference routes {path}: {e}"))?;
            config
                .resolve_routes()
                .map_err(|e| miette::miette!("failed to resolve routes from {path}: {e}"))?
        }
        InferenceRouteSource::Cluster => {
            let (Some(_id), Some(channel)) = (sandbox_id, gateway_channel) else {
                return Ok(None);
            };

            // Cluster mode: fetch bundle from gateway
            info!("Fetching inference route bundle from gateway");
            let client = grpc_client::CachedOpenShellClient::new(channel.clone());
            match client.fetch_inference_bundle().await {
                Ok(bundle) => {
                    initial_revision = Some(bundle.revision.clone());
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Informational)
                            .status(StatusId::Success)
                            .state(StateId::Enabled, "loaded")
                            .unmapped("route_count", serde_json::json!(bundle.routes.len()))
                            .unmapped("revision", serde_json::json!(&bundle.revision))
                            .message(format!(
                                "Loaded inference route bundle [route_count:{} revision:{}]",
                                bundle.routes.len(),
                                bundle.revision
                            ))
                            .build()
                    );
                    bundle_to_resolved_routes(&bundle)
                }
                Err(e) => {
                    // Distinguish expected "not configured" states from server errors.
                    // gRPC PermissionDenied/NotFound means inference bundle is unavailable
                    // for this sandbox — skip gracefully. Other errors are unexpected.
                    let msg = e.to_string();
                    if msg.contains("permission denied") || msg.contains("not found") {
                        ocsf_emit!(
                            ConfigStateChangeBuilder::new(ocsf_ctx())
                                .severity(SeverityId::Informational)
                                .status(StatusId::Success)
                                .state(StateId::Disabled, "disabled")
                                .unmapped("error", serde_json::json!(e.to_string()))
                                .message(format!(
                                    "Inference bundle unavailable, routing disabled [error:{e}]"
                                ))
                                .build()
                        );
                        return Ok(None);
                    }
                    ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                        .severity(SeverityId::Medium)
                        .status(StatusId::Failure)
                        .state(StateId::Disabled, "disabled")
                        .unmapped("error", serde_json::json!(e.to_string()))
                        .message(format!(
                            "Failed to fetch inference bundle, inference routing disabled [error:{e}]"
                        ))
                        .build());
                    return Ok(None);
                }
            }
        }
        InferenceRouteSource::None => {
            // No route source — inference routing is not configured
            return Ok(None);
        }
    };

    if routes.is_empty() && disable_inference_on_empty_routes(source) {
        ocsf_emit!(
            ConfigStateChangeBuilder::new(ocsf_ctx())
                .severity(SeverityId::Informational)
                .status(StatusId::Success)
                .state(StateId::Disabled, "disabled")
                .message("No usable inference routes, inference routing disabled")
                .build()
        );
        return Ok(None);
    }

    if routes.is_empty() {
        ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
            .severity(SeverityId::Informational)
            .status(StatusId::Success)
            .state(StateId::Other, "waiting")
            .message("Inference route bundle is empty; keeping routing enabled and waiting for refresh")
            .build());
    }

    ocsf_emit!(
        ConfigStateChangeBuilder::new(ocsf_ctx())
            .severity(SeverityId::Informational)
            .status(StatusId::Success)
            .state(StateId::Enabled, "enabled")
            .unmapped("route_count", serde_json::json!(routes.len()))
            .message(format!(
                "Inference routing enabled with local execution [route_count:{}]",
                routes.len()
            ))
            .build()
    );

    // Partition routes by name into user-facing and system caches.
    let (user_routes, system_routes) = partition_routes(routes);

    let router =
        Router::new().map_err(|e| miette::miette!("failed to initialize inference router: {e}"))?;
    let patterns = l7::inference::default_patterns();

    let ctx = Arc::new(proxy::InferenceContext::new(
        patterns,
        router,
        user_routes,
        system_routes,
    ));

    // Spawn background route cache refresh for cluster mode at startup so
    // request handling never depends on control-plane latency.
    if matches!(source, InferenceRouteSource::Cluster)
        && let (Some(_id), Some(channel)) = (sandbox_id, gateway_channel)
    {
        spawn_route_refresh(
            ctx.route_cache(),
            ctx.system_route_cache(),
            channel.clone(),
            route_refresh_interval_secs(),
            initial_revision,
        );
    }

    Ok(Some(ctx))
}

/// Route name for the sandbox system inference route.
const SANDBOX_SYSTEM_ROUTE_NAME: &str = "sandbox-system";

/// Split resolved routes into user-facing and system caches by route name.
///
/// Routes named `"sandbox-system"` go to the system cache; everything else
/// (including `"inference.local"` and empty names) goes to the user cache.
fn partition_routes(
    routes: Vec<openshell_router::config::ResolvedRoute>,
) -> (
    Vec<openshell_router::config::ResolvedRoute>,
    Vec<openshell_router::config::ResolvedRoute>,
) {
    let mut user = Vec::new();
    let mut system = Vec::new();
    for r in routes {
        if r.name == SANDBOX_SYSTEM_ROUTE_NAME {
            system.push(r);
        } else {
            user.push(r);
        }
    }
    (user, system)
}

/// Convert a proto bundle response into resolved routes for the router.
pub(crate) fn bundle_to_resolved_routes(
    bundle: &openshell_core::proto::GetInferenceBundleResponse,
) -> Vec<openshell_router::config::ResolvedRoute> {
    bundle
        .routes
        .iter()
        .map(|r| {
            let (auth, default_headers, passthrough_headers) =
                openshell_core::inference::route_headers_for_provider_type(&r.provider_type);
            let timeout = if r.timeout_secs == 0 {
                openshell_router::config::DEFAULT_ROUTE_TIMEOUT
            } else {
                Duration::from_secs(r.timeout_secs)
            };
            openshell_router::config::ResolvedRoute {
                name: r.name.clone(),
                endpoint: r.base_url.clone(),
                model: r.model_id.clone(),
                api_key: r.api_key.clone(),
                protocols: r.protocols.clone(),
                auth,
                default_headers,
                passthrough_headers,
                timeout,
            }
        })
        .collect()
}

/// Spawn a background task that periodically refreshes both route caches from the gateway.
///
/// The loop uses the bundle `revision` hash to avoid unnecessary cache writes
/// when routes haven't changed. `initial_revision` is the revision captured
/// during the startup fetch in [`build_inference_context`] so the first refresh
/// cycle can already skip a no-op update.
pub(crate) fn spawn_route_refresh(
    user_cache: Arc<tokio::sync::RwLock<Vec<openshell_router::config::ResolvedRoute>>>,
    system_cache: Arc<tokio::sync::RwLock<Vec<openshell_router::config::ResolvedRoute>>>,
    channel: AuthedChannel,
    interval_secs: u64,
    initial_revision: Option<String>,
) {
    tokio::spawn(async move {
        use tokio::time::{MissedTickBehavior, interval};

        let mut current_revision = initial_revision;
        let client = grpc_client::CachedOpenShellClient::new(channel);

        let mut tick = interval(Duration::from_secs(interval_secs));
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tick.tick().await;

            match client.fetch_inference_bundle().await {
                Ok(bundle) => {
                    if current_revision.as_deref() == Some(&bundle.revision) {
                        trace!(revision = %bundle.revision, "Inference bundle unchanged");
                        continue;
                    }

                    let routes = bundle_to_resolved_routes(&bundle);
                    let (user_routes, system_routes) = partition_routes(routes);
                    ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                        .severity(SeverityId::Informational)
                        .status(StatusId::Success)
                        .state(StateId::Enabled, "updated")
                        .unmapped("user_route_count", serde_json::json!(user_routes.len()))
                        .unmapped("system_route_count", serde_json::json!(system_routes.len()))
                        .unmapped("revision", serde_json::json!(&bundle.revision))
                        .message(format!(
                            "Inference routes updated [user_route_count:{} system_route_count:{} revision:{}]",
                            user_routes.len(),
                            system_routes.len(),
                            bundle.revision
                        ))
                        .build());
                    current_revision = Some(bundle.revision);
                    *user_cache.write().await = user_routes;
                    *system_cache.write().await = system_routes;
                }
                Err(e) => {
                    ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                        .severity(SeverityId::Medium)
                        .status(StatusId::Failure)
                        .state(StateId::Other, "stale")
                        .unmapped("error", serde_json::json!(e.to_string()))
                        .message(format!(
                            "Failed to refresh inference route cache, keeping stale routes [error:{e}]"
                        ))
                        .build());
                }
            }
        }
    });
}

// ============================================================================
// Baseline filesystem path enrichment
// ============================================================================

/// Minimum read-only paths required for a proxy-mode sandbox child process to
/// function: dynamic linker, shared libraries, DNS resolution, CA certs,
/// Python venv, openshell logs, process info, and random bytes.
///
/// `/proc` and `/dev/urandom` are included here for the same reasons they
/// appear in `restrictive_default_policy()`: virtually every process needs
/// them.  Before the Landlock per-path fix (#677) these were effectively free
/// because a missing path silently disabled the entire ruleset; now they must
/// be explicit.
const PROXY_BASELINE_READ_ONLY: &[&str] = &[
    "/usr",
    "/lib",
    "/etc",
    "/app",
    "/var/log",
    "/proc",
    "/dev/urandom",
];

/// Minimum read-write paths required for a proxy-mode sandbox child process:
/// user working directory and temporary files.
const PROXY_BASELINE_READ_WRITE: &[&str] = &["/sandbox", "/tmp"];

/// GPU read-only paths.
///
/// `/run/nvidia-persistenced`: NVML tries to connect to the persistenced
/// socket at init time.  If the directory exists but Landlock denies traversal
/// (EACCES vs ECONNREFUSED), NVML returns `NVML_ERROR_INSUFFICIENT_PERMISSIONS`
/// even though the daemon is optional.  Only read/traversal access is needed.
///
/// `/usr/lib/wsl`: On WSL2, CDI bind-mounts GPU libraries (libdxcore.so,
/// libcuda.so.1.1, etc.) into paths under `/usr/lib/wsl/`.  Although `/usr`
/// is already in `PROXY_BASELINE_READ_ONLY`, individual file bind-mounts may
/// not be covered by the parent-directory Landlock rule when the mount crosses
/// a filesystem boundary.  Listing `/usr/lib/wsl` explicitly ensures traversal
/// is permitted regardless of Landlock's cross-mount behaviour.
const GPU_BASELINE_READ_ONLY: &[&str] = &[
    "/run/nvidia-persistenced",
    "/usr/lib/wsl", // WSL2: CDI-injected GPU library directory
];

/// GPU read-write paths (static).
///
/// `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`,
/// `/dev/nvidia-modeset`: control and UVM devices injected by CDI on native
/// Linux.  Landlock restricts `open(2)` on device files even when DAC allows
/// it; these need read-write because NVML/CUDA opens them with `O_RDWR`.
/// These devices do not exist on WSL2 and will be skipped by the existence
/// check in `enrich_proto_baseline_paths()`.
///
/// `/dev/dxg`: On WSL2, NVIDIA GPUs are exposed through the DXG kernel driver
/// (DirectX Graphics) rather than the native nvidia* devices.  CDI injects
/// `/dev/dxg` as the sole GPU device node; it does not exist on native Linux
/// and will be skipped there by the existence check.
///
/// `/proc`: CUDA writes to `/proc/<pid>/task/<tid>/comm` during `cuInit()`
/// to set thread names.  Without write access, `cuInit()` returns error 304.
/// Must use `/proc` (not `/proc/self/task`) because Landlock rules bind to
/// inodes and child processes have different procfs inodes than the parent.
///
/// Per-GPU device files (`/dev/nvidia0`, …) are enumerated at runtime by
/// `enumerate_gpu_device_nodes()` since the count varies.
const GPU_BASELINE_READ_WRITE: &[&str] = &[
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
    "/dev/dxg", // WSL2: DXG device (GPU via DirectX kernel driver, injected by CDI)
    "/proc",
];

/// Returns true if GPU devices are present in the container.
///
/// Checks both the native Linux NVIDIA control device (`/dev/nvidiactl`) and
/// the WSL2 DXG device (`/dev/dxg`).  CDI injects exactly one of these
/// depending on the host kernel; the other will not exist.
fn has_gpu_devices() -> bool {
    std::path::Path::new("/dev/nvidiactl").exists() || std::path::Path::new("/dev/dxg").exists()
}

/// Enumerate per-GPU device nodes (`/dev/nvidia0`, `/dev/nvidia1`, …).
fn enumerate_gpu_device_nodes() -> Vec<String> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/dev") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(suffix) = name.strip_prefix("nvidia") {
                if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
                    continue;
                }
                paths.push(entry.path().to_string_lossy().into_owned());
            }
        }
    }
    paths
}

fn push_unique(paths: &mut Vec<String>, path: String) {
    if !paths.iter().any(|p| p == &path) {
        paths.push(path);
    }
}

fn collect_baseline_enrichment_paths(
    include_proxy: bool,
    include_gpu: bool,
    gpu_device_nodes: Vec<String>,
) -> (Vec<String>, Vec<String>) {
    let mut ro = Vec::new();
    let mut rw = Vec::new();

    if include_proxy {
        for &path in PROXY_BASELINE_READ_ONLY {
            push_unique(&mut ro, path.to_string());
        }
        for &path in PROXY_BASELINE_READ_WRITE {
            push_unique(&mut rw, path.to_string());
        }
    }

    if include_gpu {
        for &path in GPU_BASELINE_READ_ONLY {
            push_unique(&mut ro, path.to_string());
        }
        for &path in GPU_BASELINE_READ_WRITE {
            push_unique(&mut rw, path.to_string());
        }
        for path in gpu_device_nodes {
            push_unique(&mut rw, path);
        }
    }

    // A path promoted to read_write (e.g. /proc for GPU) should not also
    // appear in read_only — Landlock handles the overlap correctly but the
    // duplicate is confusing when inspecting the effective policy.
    ro.retain(|p| !rw.contains(p));

    (ro, rw)
}

fn active_baseline_enrichment_paths(include_proxy: bool) -> (Vec<String>, Vec<String>) {
    let include_gpu = has_gpu_devices();
    let gpu_device_nodes = if include_gpu {
        enumerate_gpu_device_nodes()
    } else {
        Vec::new()
    };
    collect_baseline_enrichment_paths(include_proxy, include_gpu, gpu_device_nodes)
}

/// Collect all active baseline paths for tests and diagnostics.
/// Returns `(read_only, read_write)` as owned `String` vecs.
#[cfg(test)]
fn baseline_enrichment_paths() -> (Vec<String>, Vec<String>) {
    active_baseline_enrichment_paths(true)
}

fn enrich_proto_baseline_paths_with<F>(
    proto: &mut openshell_core::proto::SandboxPolicy,
    ro: &[String],
    rw: &[String],
    path_exists: F,
) -> bool
where
    F: Fn(&str) -> bool,
{
    if ro.is_empty() && rw.is_empty() {
        return false;
    }

    let fs = proto
        .filesystem
        .get_or_insert_with(|| openshell_core::proto::FilesystemPolicy {
            include_workdir: true,
            ..Default::default()
        });

    let mut modified = false;
    for path in ro {
        if !fs.read_only.iter().any(|p| p == path) && !fs.read_write.iter().any(|p| p == path) {
            if !path_exists(path) {
                debug!(
                    path,
                    "Baseline read-only path does not exist, skipping enrichment"
                );
                continue;
            }
            fs.read_only.push(path.clone());
            modified = true;
        }
    }
    for path in rw {
        if fs.read_only.iter().any(|p| p == path) || fs.read_write.iter().any(|p| p == path) {
            continue;
        }
        if !path_exists(path) {
            debug!(
                path,
                "Baseline read-write path does not exist, skipping enrichment"
            );
            continue;
        }
        fs.read_write.push(path.clone());
        modified = true;
    }

    modified
}

/// Ensure a proto `SandboxPolicy` includes the baseline filesystem paths
/// required by proxy-mode sandboxes and GPU runtimes. Paths are only added if
/// missing; user-specified paths are never removed.
///
/// Returns `true` if the policy was modified (caller may want to sync back).
fn enrich_proto_baseline_paths(proto: &mut openshell_core::proto::SandboxPolicy) -> bool {
    let (ro, rw) = active_baseline_enrichment_paths(!proto.network_policies.is_empty());

    // Baseline paths are system-injected, not user-specified.  Skip paths
    // that do not exist in this container image to avoid noisy warnings from
    // Landlock and, more critically, to prevent a single missing baseline
    // path from abandoning the entire Landlock ruleset under best-effort
    // mode (see issue #664).
    let modified = enrich_proto_baseline_paths_with(proto, &ro, &rw, |path| {
        std::path::Path::new(path).exists()
    });

    if modified {
        ocsf_emit!(
            ConfigStateChangeBuilder::new(ocsf_ctx())
                .severity(SeverityId::Informational)
                .status(StatusId::Success)
                .state(StateId::Enabled, "enriched")
                .message("Enriched policy with baseline filesystem paths for proxy mode")
                .build()
        );
    }

    modified
}

/// Ensure a `SandboxPolicy` (Rust type) includes the baseline filesystem
/// paths required by proxy-mode sandboxes and GPU runtimes. Used for the
/// local-file code path where no proto is available.
fn enrich_sandbox_baseline_paths(policy: &mut SandboxPolicy) {
    let (ro, rw) =
        active_baseline_enrichment_paths(matches!(policy.network.mode, NetworkMode::Proxy));
    if ro.is_empty() && rw.is_empty() {
        return;
    }

    let mut modified = false;
    for path in &ro {
        let p = std::path::PathBuf::from(path);
        if !policy.filesystem.read_only.contains(&p) && !policy.filesystem.read_write.contains(&p) {
            if !p.exists() {
                debug!(
                    path,
                    "Baseline read-only path does not exist, skipping enrichment"
                );
                continue;
            }
            policy.filesystem.read_only.push(p);
            modified = true;
        }
    }
    for path in &rw {
        let p = std::path::PathBuf::from(path);
        if policy.filesystem.read_only.contains(&p) || policy.filesystem.read_write.contains(&p) {
            continue;
        }
        if !p.exists() {
            debug!(
                path,
                "Baseline read-write path does not exist, skipping enrichment"
            );
            continue;
        }
        policy.filesystem.read_write.push(p);
        modified = true;
    }

    if modified {
        ocsf_emit!(
            ConfigStateChangeBuilder::new(ocsf_ctx())
                .severity(SeverityId::Informational)
                .status(StatusId::Success)
                .state(StateId::Enabled, "enriched")
                .message("Enriched policy with baseline filesystem paths for proxy mode")
                .build()
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::needless_raw_string_hashes,
    clippy::iter_on_single_items,
    clippy::similar_names,
    clippy::manual_string_new,
    clippy::doc_markdown,
    reason = "Test code: test fixtures often use idiomatic forms not flagged in production."
)]
mod baseline_tests {
    use super::*;
    use openshell_core::policy::{FilesystemPolicy, LandlockPolicy, ProcessPolicy};

    #[test]
    fn proc_not_in_both_read_only_and_read_write_when_gpu_present() {
        // When GPU devices are present, /proc is promoted to read_write
        // (CUDA needs to write /proc/<pid>/task/<tid>/comm). It should
        // NOT also appear in read_only.
        if !has_gpu_devices() {
            // Can't test GPU dedup without GPU devices; skip silently.
            return;
        }
        let (ro, rw) = baseline_enrichment_paths();
        assert!(
            rw.contains(&"/proc".to_string()),
            "/proc should be in read_write when GPU is present"
        );
        assert!(
            !ro.contains(&"/proc".to_string()),
            "/proc should NOT be in read_only when it is already in read_write"
        );
    }

    #[test]
    fn proc_in_read_only_without_gpu() {
        if has_gpu_devices() {
            // On a GPU host we can't test the non-GPU path; skip silently.
            return;
        }
        let (ro, _rw) = baseline_enrichment_paths();
        assert!(
            ro.contains(&"/proc".to_string()),
            "/proc should be in read_only when GPU is not present"
        );
    }

    #[test]
    fn baseline_read_write_always_includes_sandbox_and_tmp() {
        let (_ro, rw) = baseline_enrichment_paths();
        assert!(rw.contains(&"/sandbox".to_string()));
        assert!(rw.contains(&"/tmp".to_string()));
    }

    #[test]
    fn enumerate_gpu_device_nodes_skips_bare_nvidia() {
        // "nvidia" (without a trailing digit) is a valid /dev entry on some
        // systems but is not a per-GPU device node.  The enumerator must
        // not match it.
        let nodes = enumerate_gpu_device_nodes();
        assert!(
            !nodes.contains(&"/dev/nvidia".to_string()),
            "bare /dev/nvidia should not be enumerated: {nodes:?}"
        );
    }

    #[test]
    fn no_duplicate_paths_in_baseline() {
        let (ro, rw) = baseline_enrichment_paths();
        // No path should appear in both lists.
        for path in &ro {
            assert!(
                !rw.contains(path),
                "path {path} appears in both read_only and read_write"
            );
        }
    }

    #[test]
    fn proto_enrichment_preserves_explicit_read_only_for_baseline_read_write_paths() {
        let mut policy = openshell_policy::restrictive_default_policy();
        policy.filesystem = Some(openshell_core::proto::FilesystemPolicy {
            read_only: vec!["/tmp".to_string()],
            read_write: vec![],
            include_workdir: false,
        });
        policy.network_policies.insert(
            "test".into(),
            openshell_core::proto::NetworkPolicyRule {
                name: "test-rule".into(),
                endpoints: vec![openshell_core::proto::NetworkEndpoint {
                    host: "example.com".into(),
                    port: 443,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );

        enrich_proto_baseline_paths(&mut policy);

        let filesystem = policy.filesystem.expect("filesystem policy");
        assert!(
            filesystem.read_only.contains(&"/tmp".to_string()),
            "explicit read_only baseline path should be preserved"
        );
        assert!(
            !filesystem.read_write.contains(&"/tmp".to_string()),
            "baseline enrichment must not promote explicit read_only /tmp to read_write"
        );
    }

    #[test]
    fn proto_gpu_enrichment_adds_devices_without_network_policy() {
        let mut policy = openshell_policy::restrictive_default_policy();
        assert!(
            policy.network_policies.is_empty(),
            "regression setup must exercise the no-network default path"
        );
        let (ro, rw) =
            collect_baseline_enrichment_paths(false, true, vec!["/dev/nvidia0".to_string()]);

        let enriched = enrich_proto_baseline_paths_with(&mut policy, &ro, &rw, |path| {
            matches!(path, "/proc" | "/dev/nvidia0")
        });

        let filesystem = policy.filesystem.expect("filesystem policy");
        assert!(
            enriched,
            "GPU enrichment should not require network policies"
        );
        assert!(
            filesystem.read_write.contains(&"/dev/nvidia0".to_string()),
            "GPU enrichment should add enumerated device nodes without network policies"
        );
    }

    #[test]
    fn gpu_baseline_read_write_contains_dxg() {
        // /dev/dxg must be present so WSL2 sandboxes get the Landlock
        // read-write rule for the CDI-injected DXG device.  The existence
        // check in enrich_proto_baseline_paths() skips it on native Linux.
        assert!(
            GPU_BASELINE_READ_WRITE.contains(&"/dev/dxg"),
            "/dev/dxg must be in GPU_BASELINE_READ_WRITE for WSL2 support"
        );
    }

    #[test]
    fn local_enrichment_preserves_explicit_read_only_for_baseline_read_write_paths() {
        let mut policy = SandboxPolicy {
            version: 1,
            filesystem: FilesystemPolicy {
                read_only: vec![std::path::PathBuf::from("/tmp")],
                read_write: vec![],
                include_workdir: false,
            },
            network: NetworkPolicy {
                mode: NetworkMode::Proxy,
                proxy: Some(ProxyPolicy { http_addr: None }),
            },
            landlock: LandlockPolicy::default(),
            process: ProcessPolicy::default(),
        };

        enrich_sandbox_baseline_paths(&mut policy);

        assert!(
            policy
                .filesystem
                .read_only
                .contains(&std::path::PathBuf::from("/tmp")),
            "explicit read_only baseline path should be preserved"
        );
        assert!(
            !policy
                .filesystem
                .read_write
                .contains(&std::path::PathBuf::from("/tmp")),
            "baseline enrichment must not promote explicit read_only /tmp to read_write"
        );
    }

    #[test]
    fn gpu_baseline_read_only_contains_usr_lib_wsl() {
        // /usr/lib/wsl must be present so CDI-injected WSL2 GPU library
        // bind-mounts are accessible under Landlock.  Skipped on native Linux.
        assert!(
            GPU_BASELINE_READ_ONLY.contains(&"/usr/lib/wsl"),
            "/usr/lib/wsl must be in GPU_BASELINE_READ_ONLY for WSL2 CDI library paths"
        );
    }

    #[test]
    fn has_gpu_devices_reflects_dxg_or_nvidiactl() {
        // Verify the OR logic: result must match the manual disjunction of
        // the two path checks.  Passes in all environments.
        let nvidiactl = std::path::Path::new("/dev/nvidiactl").exists();
        let dxg = std::path::Path::new("/dev/dxg").exists();
        assert_eq!(
            has_gpu_devices(),
            nvidiactl || dxg,
            "has_gpu_devices() should be true iff /dev/nvidiactl or /dev/dxg exists"
        );
    }
}

/// Re-export shared gRPC retry helpers from core. The retry helpers (and the
/// `tonic::Status` classification they encode) are used by the proxy and
/// supervisor poll loops alike, so they live in `openshell-core`.
use openshell_core::grpc_retry::grpc_retry;

/// Load sandbox policy from local files or gRPC.
///
/// Priority:
/// 1. If `policy_rules` and `policy_data` are provided, load OPA engine from local files
/// 2. If `sandbox_id` and `openshell_endpoint` are provided, fetch via gRPC
/// 3. If the server returns no policy, discover from disk or use restrictive default
/// 4. Otherwise, return an error
///
/// Returns the policy, the OPA engine, and (for gRPC mode) the original proto
/// policy. The proto is retained so the OPA engine can be rebuilt with symlink
/// resolution after the container entrypoint starts.
async fn load_policy(
    sandbox_id: Option<String>,
    sandbox: Option<String>,
    gateway_channel: Option<AuthedChannel>,
    policy_rules: Option<String>,
    policy_data: Option<String>,
) -> Result<(
    SandboxPolicy,
    Option<Arc<OpaEngine>>,
    Option<openshell_core::proto::SandboxPolicy>,
)> {
    // File mode: load OPA engine from rego rules + YAML data (dev override)
    if let (Some(policy_file), Some(data_file)) = (&policy_rules, &policy_data) {
        ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
            .severity(SeverityId::Informational)
            .status(StatusId::Success)
            .state(StateId::Other, "loading")
            .unmapped("policy_rules", serde_json::json!(policy_file))
            .unmapped("policy_data", serde_json::json!(data_file))
            .message(format!(
                "Loading OPA policy engine from local files [rules:{policy_file} data:{data_file}]"
            ))
            .build());
        let engine = OpaEngine::from_files(
            std::path::Path::new(policy_file),
            std::path::Path::new(data_file),
        )?;
        let config = engine.query_sandbox_config()?;
        let mut policy = SandboxPolicy {
            version: 1,
            filesystem: config.filesystem,
            network: NetworkPolicy {
                mode: NetworkMode::Proxy,
                proxy: Some(ProxyPolicy { http_addr: None }),
            },
            landlock: config.landlock,
            process: config.process,
        };
        enrich_sandbox_baseline_paths(&mut policy);
        return Ok((policy, Some(Arc::new(engine)), None));
    }

    // gRPC mode: fetch typed proto policy, construct OPA engine from baked rules + proto data
    if let (Some(id), Some(channel)) = (&sandbox_id, &gateway_channel) {
        info!(
            sandbox_id = %id,
            "Fetching sandbox policy via gRPC"
        );
        let client = grpc_client::CachedOpenShellClient::new(channel.clone());
        let proto_policy = grpc_retry("Policy fetch", || client.fetch_policy(id)).await?;

        let mut proto_policy = if let Some(p) = proto_policy {
            p
        } else {
            // No policy configured on the server. Discover from disk or
            // fall back to the restrictive default, then sync to the
            // gateway so it becomes the authoritative baseline.
            ocsf_emit!(
                ConfigStateChangeBuilder::new(ocsf_ctx())
                    .severity(SeverityId::Informational)
                    .status(StatusId::Success)
                    .state(StateId::Other, "discovery")
                    .message("Server returned no policy; attempting local discovery")
                    .build()
            );
            let mut discovered = discover_policy_from_disk_or_default();
            // Enrich before syncing so the gateway baseline includes
            // baseline paths from the start.
            enrich_proto_baseline_paths(&mut discovered);
            let sandbox = sandbox.as_deref().ok_or_else(|| {
                miette::miette!(
                    "Cannot sync discovered policy: sandbox not available.\n\
                     Set OPENSHELL_SANDBOX or --sandbox to enable policy sync."
                )
            })?;

            // Sync and re-fetch over a single connection to avoid extra
            // TLS handshakes.
            grpc_retry("Policy discovery sync", || {
                client.discover_and_sync_policy(id, sandbox, &discovered)
            })
            .await?
        };

        // Ensure baseline filesystem paths are present for proxy-mode
        // sandboxes.  If the policy was enriched, sync the updated version
        // back to the gateway so users can see the effective policy.
        let enriched = enrich_proto_baseline_paths(&mut proto_policy);
        if enriched
            && let Some(sandbox_name) = sandbox.as_deref()
            && let Err(e) = client.sync_policy(sandbox_name, &proto_policy).await
        {
            warn!(
                error = %e,
                "Failed to sync enriched policy back to gateway (non-fatal)"
            );
        }

        // Build OPA engine from baked-in rules + typed proto data.
        // In cluster mode, proxy networking is always enabled so OPA is
        // always required for allow/deny decisions.
        // The initial load uses pid=0 (no symlink resolution) because the
        // container hasn't started yet. After the entrypoint spawns, the
        // engine is rebuilt with the real PID for symlink resolution.
        info!("Creating OPA engine from proto policy data");
        let opa_engine = Some(Arc::new(OpaEngine::from_proto(&proto_policy)?));

        let policy = SandboxPolicy::try_from(proto_policy.clone())?;
        return Ok((policy, opa_engine, Some(proto_policy)));
    }

    // No policy source available
    Err(miette::miette!(
        "Sandbox policy required. Provide one of:\n\
         - --policy-rules and --policy-data (or OPENSHELL_POLICY_RULES and OPENSHELL_POLICY_DATA env vars)\n\
         - --sandbox-id and --openshell-endpoint (or OPENSHELL_SANDBOX_ID and OPENSHELL_ENDPOINT env vars)"
    ))
}

/// Try to discover a sandbox policy from the well-known disk path, falling
/// back to the legacy path, then to the hardcoded restrictive default.
fn discover_policy_from_disk_or_default() -> openshell_core::proto::SandboxPolicy {
    let primary = std::path::Path::new(openshell_policy::CONTAINER_POLICY_PATH);
    if primary.exists() {
        return discover_policy_from_path(primary);
    }
    let legacy = std::path::Path::new(openshell_policy::LEGACY_CONTAINER_POLICY_PATH);
    if legacy.exists() {
        ocsf_emit!(
            ConfigStateChangeBuilder::new(ocsf_ctx())
                .severity(SeverityId::Informational)
                .status(StatusId::Success)
                .state(StateId::Enabled, "loaded")
                .unmapped(
                    "legacy_path",
                    serde_json::json!(legacy.display().to_string())
                )
                .unmapped("new_path", serde_json::json!(primary.display().to_string()))
                .message(format!(
                    "Policy found at legacy path; consider moving [legacy_path:{} new_path:{}]",
                    legacy.display(),
                    primary.display()
                ))
                .build()
        );
        return discover_policy_from_path(legacy);
    }
    discover_policy_from_path(primary)
}

/// Try to read a sandbox policy YAML from `path`, falling back to the
/// hardcoded restrictive default if the file is missing or invalid.
fn discover_policy_from_path(path: &std::path::Path) -> openshell_core::proto::SandboxPolicy {
    use openshell_policy::{
        parse_sandbox_policy, restrictive_default_policy, validate_sandbox_policy,
    };

    let Ok(yaml) = std::fs::read_to_string(path) else {
        ocsf_emit!(
            ConfigStateChangeBuilder::new(ocsf_ctx())
                .severity(SeverityId::Informational)
                .status(StatusId::Success)
                .state(StateId::Enabled, "default")
                .message(format!(
                    "No policy file on disk, using restrictive default [path:{}]",
                    path.display()
                ))
                .build()
        );
        return restrictive_default_policy();
    };
    ocsf_emit!(
        ConfigStateChangeBuilder::new(ocsf_ctx())
            .severity(SeverityId::Informational)
            .status(StatusId::Success)
            .state(StateId::Enabled, "loaded")
            .message(format!(
                "Loaded sandbox policy from container disk [path:{}]",
                path.display()
            ))
            .build()
    );
    match parse_sandbox_policy(&yaml) {
        Ok(policy) => {
            // Validate the disk-loaded policy for safety.
            if let Err(violations) = validate_sandbox_policy(&policy) {
                let messages: Vec<String> = violations.iter().map(ToString::to_string).collect();
                ocsf_emit!(DetectionFindingBuilder::new(ocsf_ctx())
                    .activity(ActivityId::Open)
                    .severity(SeverityId::Medium)
                    .action(ActionId::Denied)
                    .disposition(DispositionId::Blocked)
                    .finding_info(
                        FindingInfo::new(
                            "unsafe-disk-policy",
                            "Unsafe Disk Policy Content",
                        )
                        .with_desc(&format!(
                            "Disk policy at {} contains unsafe content: {}",
                            path.display(),
                            messages.join("; "),
                        )),
                    )
                    .message(format!(
                        "Disk policy contains unsafe content, using restrictive default [path:{}]",
                        path.display()
                    ))
                    .build());
                return restrictive_default_policy();
            }
            policy
        }
        Err(e) => {
            ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                .severity(SeverityId::Medium)
                .status(StatusId::Failure)
                .state(StateId::Other, "fallback")
                .message(format!(
                    "Failed to parse disk policy, using restrictive default [path:{} error:{e}]",
                    path.display()
                ))
                .build());
            restrictive_default_policy()
        }
    }
}

/// Validate that the `sandbox` user exists in this image.
///
/// All sandbox images must include a `sandbox` user for privilege dropping.
/// This check runs at supervisor startup (inside the container) where we can
/// inspect `/etc/passwd`. If the user is missing, the sandbox fails fast
/// with a clear error instead of silently running child processes as root.
#[cfg(unix)]
fn validate_sandbox_user(policy: &SandboxPolicy) -> Result<()> {
    use nix::unistd::User;

    let user_name = policy.process.run_as_user.as_deref().unwrap_or("sandbox");

    if user_name.is_empty() || user_name == "sandbox" {
        match User::from_name("sandbox") {
            Ok(Some(_)) => {
                ocsf_emit!(
                    ConfigStateChangeBuilder::new(ocsf_ctx())
                        .severity(SeverityId::Informational)
                        .status(StatusId::Success)
                        .state(StateId::Enabled, "validated")
                        .message("Validated 'sandbox' user exists in image")
                        .build()
                );
            }
            Ok(None) => {
                return Err(miette::miette!(
                    "sandbox user 'sandbox' not found in image; \
                     all sandbox images must include a 'sandbox' user and group"
                ));
            }
            Err(e) => {
                return Err(miette::miette!("failed to look up 'sandbox' user: {e}"));
            }
        }
    }

    Ok(())
}

/// Prepare a `read_write` path for the sandboxed process.
///
/// Returns `true` when the path was created by the supervisor and therefore
/// still needs to be chowned to the sandbox user/group. Existing paths keep
/// their image-defined ownership.
#[cfg(unix)]
fn prepare_read_write_path(path: &std::path::Path) -> Result<bool> {
    // SECURITY: use symlink_metadata (lstat) to inspect each path *before*
    // calling chown. chown follows symlinks, so a malicious container image
    // could place a symlink (e.g. /sandbox -> /etc/shadow) to trick the
    // root supervisor into transferring ownership of arbitrary files.
    // The TOCTOU window between lstat and chown is not exploitable because
    // no untrusted process is running yet (the child has not been forked).
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            return Err(miette::miette!(
                "read_write path '{}' is a symlink — refusing to chown (potential privilege escalation)",
                path.display()
            ));
        }

        debug!(
            path = %path.display(),
            "Preserving ownership for existing read_write path"
        );
        Ok(false)
    } else {
        debug!(path = %path.display(), "Creating read_write directory");
        std::fs::create_dir_all(path).into_diagnostic()?;
        Ok(true)
    }
}

/// Prepare filesystem for the sandboxed process.
///
/// Creates `read_write` directories if they don't exist and sets ownership
/// on newly-created paths to the configured sandbox user/group. This runs as
/// the supervisor (root) before forking the child process.
#[cfg(unix)]
fn prepare_filesystem(policy: &SandboxPolicy) -> Result<()> {
    use nix::unistd::{Group, User, chown};

    let user_name = match policy.process.run_as_user.as_deref() {
        Some(name) if !name.is_empty() => Some(name),
        _ => None,
    };
    let group_name = match policy.process.run_as_group.as_deref() {
        Some(name) if !name.is_empty() => Some(name),
        _ => None,
    };

    // If no user/group configured, nothing to do
    if user_name.is_none() && group_name.is_none() {
        return Ok(());
    }

    // Resolve user and group
    let uid = if let Some(name) = user_name {
        Some(
            User::from_name(name)
                .into_diagnostic()?
                .ok_or_else(|| miette::miette!("Sandbox user not found: {name}"))?
                .uid,
        )
    } else {
        None
    };

    let gid = if let Some(name) = group_name {
        Some(
            Group::from_name(name)
                .into_diagnostic()?
                .ok_or_else(|| miette::miette!("Sandbox group not found: {name}"))?
                .gid,
        )
    } else {
        None
    };

    // Create missing read_write paths and only chown the ones we created.
    for path in &policy.filesystem.read_write {
        if prepare_read_write_path(path)? {
            debug!(
                path = %path.display(),
                ?uid,
                ?gid,
                "Setting ownership on newly created read_write path"
            );
            chown(path, uid, gid).into_diagnostic()?;
        }
    }

    Ok(())
}

#[cfg(not(unix))]
fn prepare_filesystem(_policy: &SandboxPolicy) -> Result<()> {
    Ok(())
}

/// Background loop that polls the server for policy updates.
///
/// When a new version is detected, attempts to reload the OPA engine via
/// Flush aggregated denial summaries to the gateway via `SubmitPolicyAnalysis`.
async fn flush_proposals_to_gateway(
    channel: AuthedChannel,
    sandbox_name: &str,
    summaries: Vec<denial_aggregator::FlushableDenialSummary>,
) -> Result<()> {
    use crate::grpc_client::CachedOpenShellClient;
    use openshell_core::proto::{DenialSummary, L7RequestSample};

    let client = CachedOpenShellClient::new(channel);

    // Convert FlushableDenialSummary to proto DenialSummary.
    let proto_summaries: Vec<DenialSummary> = summaries
        .into_iter()
        .map(|s| DenialSummary {
            sandbox_id: String::new(),
            host: s.host,
            port: u32::from(s.port),
            binary: s.binary,
            ancestors: s.ancestors,
            deny_reason: s.deny_reason,
            first_seen_ms: s.first_seen_ms,
            last_seen_ms: s.last_seen_ms,
            count: s.count,
            suppressed_count: 0,
            total_count: s.count,
            sample_cmdlines: s.sample_cmdlines,
            binary_sha256: String::new(),
            persistent: false,
            denial_stage: s.denial_stage,
            l7_request_samples: s
                .l7_samples
                .into_iter()
                .map(|l| L7RequestSample {
                    method: l.method,
                    path: l.path,
                    decision: "deny".to_string(),
                    count: l.count,
                })
                .collect(),
            l7_inspection_active: false,
        })
        .collect();

    // Run the mechanistic mapper sandbox-side to generate proposals.
    // The gateway is a thin persistence + validation layer — it never
    // generates proposals itself.
    let proposals = mechanistic_mapper::generate_proposals(&proto_summaries);

    info!(
        sandbox_name = %sandbox_name,
        summaries = proto_summaries.len(),
        proposals = proposals.len(),
        "Flushed denial analysis to gateway"
    );

    client
        .submit_policy_analysis(sandbox_name, proto_summaries, proposals, "mechanistic")
        .await?;

    Ok(())
}

/// `reload_from_proto_with_pid()`. Reports load success/failure back to the
/// server. On failure, the previous engine is untouched (LKG behavior).
///
/// When the entrypoint PID is available, policy reloads include symlink
/// resolution for binary paths via the container filesystem.
struct PolicyPollLoopContext {
    channel: AuthedChannel,
    sandbox_id: String,
    opa_engine: Arc<OpaEngine>,
    entrypoint_pid: Arc<AtomicU32>,
    interval_secs: u64,
    ocsf_enabled: Arc<std::sync::atomic::AtomicBool>,
    provider_credentials: provider_credentials::ProviderCredentialState,
    policy_local_ctx: Option<Arc<policy_local::PolicyLocalContext>>,
}

async fn run_policy_poll_loop(ctx: PolicyPollLoopContext) -> Result<()> {
    use crate::grpc_client::CachedOpenShellClient;
    use openshell_core::proto::PolicySource;
    use std::sync::atomic::Ordering;

    let client = CachedOpenShellClient::new(ctx.channel.clone());
    let mut current_config_revision: u64 = 0;
    let mut current_provider_env_revision: u64 = ctx.provider_credentials.snapshot().revision;
    let mut current_policy_hash = String::new();
    let mut current_settings: std::collections::HashMap<
        String,
        openshell_core::proto::EffectiveSetting,
    > = std::collections::HashMap::new();

    // Initialize revision from the first poll.
    match client.poll_settings(&ctx.sandbox_id).await {
        Ok(result) => {
            current_config_revision = result.config_revision;
            current_policy_hash = result.policy_hash.clone();
            current_settings = result.settings;
            debug!(
                config_revision = current_config_revision,
                "Settings poll: initial config revision"
            );
        }
        Err(e) => {
            warn!(error = %e, "Settings poll: failed to fetch initial version, will retry");
        }
    }

    let interval = Duration::from_secs(ctx.interval_secs);
    loop {
        tokio::time::sleep(interval).await;

        let result = match client.poll_settings(&ctx.sandbox_id).await {
            Ok(r) => r,
            Err(e) => {
                debug!(error = %e, "Settings poll: server unreachable, will retry");
                continue;
            }
        };

        let provider_env_changed = result.provider_env_revision != current_provider_env_revision;
        if result.config_revision == current_config_revision && !provider_env_changed {
            continue;
        }

        let policy_changed = result.policy_hash != current_policy_hash;

        // Log which settings changed.
        log_setting_changes(&current_settings, &result.settings);

        ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
            .severity(SeverityId::Informational)
            .status(StatusId::Success)
            .state(StateId::Other, "detected")
            .unmapped("old_config_revision", serde_json::json!(current_config_revision))
            .unmapped("new_config_revision", serde_json::json!(result.config_revision))
            .unmapped("policy_changed", serde_json::json!(policy_changed))
            .unmapped("provider_env_changed", serde_json::json!(provider_env_changed))
            .message(format!(
                "Settings poll: config change detected [old_revision:{current_config_revision} new_revision:{} policy_changed:{policy_changed} provider_env_changed:{provider_env_changed}]",
                result.config_revision
            ))
            .build());

        if provider_env_changed {
            match client.fetch_provider_environment(&ctx.sandbox_id).await {
                Ok(env_result) => {
                    let env_count = ctx.provider_credentials.install_environment(
                        env_result.provider_env_revision,
                        env_result.environment,
                        env_result.credential_expires_at_ms,
                    );
                    current_provider_env_revision = env_result.provider_env_revision;
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Informational)
                            .status(StatusId::Success)
                            .state(StateId::Enabled, "loaded")
                            .unmapped(
                                "provider_env_revision",
                                serde_json::json!(current_provider_env_revision)
                            )
                            .message(format!(
                                "Provider environment refreshed [revision:{current_provider_env_revision} env_count:{env_count}]"
                            ))
                            .build()
                    );
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        provider_env_revision = result.provider_env_revision,
                        "Settings poll: failed to refresh provider environment"
                    );
                }
            }
        }

        // Only reload OPA when the policy payload actually changed.
        if policy_changed {
            let Some(policy) = result.policy.as_ref() else {
                ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                    .severity(SeverityId::Medium)
                    .status(StatusId::Failure)
                    .state(StateId::Other, "skipped")
                    .message("Settings poll: policy hash changed but no policy payload present; skipping reload")
                    .build());
                current_config_revision = result.config_revision;
                current_policy_hash = result.policy_hash;
                current_settings = result.settings;
                continue;
            };

            let pid = ctx.entrypoint_pid.load(Ordering::Acquire);
            match ctx.opa_engine.reload_from_proto_with_pid(policy, pid) {
                Ok(()) => {
                    if let Some(policy_local_ctx) = ctx.policy_local_ctx.as_ref() {
                        policy_local_ctx.set_current_policy(policy.clone()).await;
                    }
                    if result.global_policy_version > 0 {
                        ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Informational)
                            .status(StatusId::Success)
                            .state(StateId::Enabled, "loaded")
                            .unmapped("policy_hash", serde_json::json!(&result.policy_hash))
                            .unmapped("global_version", serde_json::json!(result.global_policy_version))
                            .message(format!(
                                "Policy reloaded successfully (global) [policy_hash:{} global_version:{}]",
                                result.policy_hash,
                                result.global_policy_version
                            ))
                            .build());
                    } else {
                        ocsf_emit!(
                            ConfigStateChangeBuilder::new(ocsf_ctx())
                                .severity(SeverityId::Informational)
                                .status(StatusId::Success)
                                .state(StateId::Enabled, "loaded")
                                .unmapped("policy_hash", serde_json::json!(&result.policy_hash))
                                .message(format!(
                                    "Policy reloaded successfully [policy_hash:{}]",
                                    result.policy_hash
                                ))
                                .build()
                        );
                    }
                    if result.version > 0
                        && result.policy_source == PolicySource::Sandbox
                        && let Err(e) = client
                            .report_policy_status(&ctx.sandbox_id, result.version, true, "")
                            .await
                    {
                        warn!(error = %e, "Failed to report policy load success");
                    }
                }
                Err(e) => {
                    ocsf_emit!(ConfigStateChangeBuilder::new(ocsf_ctx())
                        .severity(SeverityId::Medium)
                        .status(StatusId::Failure)
                        .state(StateId::Other, "failed")
                        .unmapped("version", serde_json::json!(result.version))
                        .unmapped("error", serde_json::json!(e.to_string()))
                        .message(format!(
                            "Policy reload failed, keeping last-known-good policy [version:{} error:{e}]",
                            result.version
                        ))
                        .build());
                    if result.version > 0
                        && result.policy_source == PolicySource::Sandbox
                        && let Err(report_err) = client
                            .report_policy_status(
                                &ctx.sandbox_id,
                                result.version,
                                false,
                                &e.to_string(),
                            )
                            .await
                    {
                        warn!(error = %report_err, "Failed to report policy load failure");
                    }
                }
            }
        }

        // Apply OCSF JSON toggle from the `ocsf_json_enabled` setting.
        let new_ocsf = extract_bool_setting(&result.settings, "ocsf_json_enabled").unwrap_or(false);
        let prev_ocsf = ctx.ocsf_enabled.swap(new_ocsf, Ordering::Relaxed);
        if new_ocsf != prev_ocsf {
            info!(ocsf_json_enabled = new_ocsf, "OCSF JSONL logging toggled");
        }

        // Apply the agent-proposals feature toggle. On a false→true transition
        // we lazily install the skill so a sandbox that started with the flag
        // off picks up the surface without a recreate. We never uninstall on
        // a true→false transition: stale skill content on disk is harmless
        // because route_request and agent_next_steps both gate on the live
        // atomic, so the agent that reads the skill will see 404s and an
        // empty `next_steps` array regardless.
        if let Some(flag) = openshell_core::ocsf_ctx::agent_proposals_enabled_flag() {
            let new_proposals = extract_bool_setting(
                &result.settings,
                openshell_core::settings::AGENT_POLICY_PROPOSALS_ENABLED_KEY,
            )
            .unwrap_or(false);
            let prev_proposals = flag.swap(new_proposals, Ordering::Relaxed);
            if new_proposals != prev_proposals {
                info!(
                    agent_policy_proposals_enabled = new_proposals,
                    "agent-driven policy proposals toggled"
                );
                if new_proposals && !prev_proposals {
                    match skills::install_static_skills() {
                        Ok(installed) => info!(
                            path = %installed.policy_advisor.display(),
                            "Installed sandbox agent skill on toggle-on"
                        ),
                        Err(error) => warn!(
                            error = %error,
                            "Failed to install sandbox agent skill on toggle-on"
                        ),
                    }
                }
            }
        }

        current_config_revision = result.config_revision;
        current_policy_hash = result.policy_hash;
        current_settings = result.settings;
    }
}

/// Extract a bool value from an effective setting, if present.
fn extract_bool_setting(
    settings: &std::collections::HashMap<String, openshell_core::proto::EffectiveSetting>,
    key: &str,
) -> Option<bool> {
    use openshell_core::proto::setting_value;
    settings
        .get(key)
        .and_then(|es| es.value.as_ref())
        .and_then(|sv| sv.value.as_ref())
        .and_then(|v| match v {
            setting_value::Value::BoolValue(b) => Some(*b),
            _ => None,
        })
}

/// Log individual setting changes between two snapshots.
fn log_setting_changes(
    old: &std::collections::HashMap<String, openshell_core::proto::EffectiveSetting>,
    new: &std::collections::HashMap<String, openshell_core::proto::EffectiveSetting>,
) {
    for (key, new_es) in new {
        let new_val = format_setting_value(new_es);
        match old.get(key) {
            Some(old_es) => {
                let old_val = format_setting_value(old_es);
                if old_val != new_val {
                    ocsf_emit!(
                        ConfigStateChangeBuilder::new(ocsf_ctx())
                            .severity(SeverityId::Informational)
                            .status(StatusId::Success)
                            .state(StateId::Enabled, "updated")
                            .unmapped("key", serde_json::json!(key))
                            .unmapped("old", serde_json::json!(old_val.clone()))
                            .unmapped("new", serde_json::json!(new_val.clone()))
                            .message(format!(
                                "Setting changed [key:{key} old:{old_val} new:{new_val}]"
                            ))
                            .build()
                    );
                }
            }
            None => {
                ocsf_emit!(
                    ConfigStateChangeBuilder::new(ocsf_ctx())
                        .severity(SeverityId::Informational)
                        .status(StatusId::Success)
                        .state(StateId::Enabled, "enabled")
                        .unmapped("key", serde_json::json!(key))
                        .unmapped("value", serde_json::json!(new_val.clone()))
                        .message(format!("Setting added [key:{key} value:{new_val}]"))
                        .build()
                );
            }
        }
    }
    for key in old.keys() {
        if !new.contains_key(key) {
            ocsf_emit!(
                ConfigStateChangeBuilder::new(ocsf_ctx())
                    .severity(SeverityId::Informational)
                    .status(StatusId::Success)
                    .state(StateId::Disabled, "disabled")
                    .unmapped("key", serde_json::json!(key))
                    .message(format!("Setting removed [key:{key}]"))
                    .build()
            );
        }
    }
}

/// Format an `EffectiveSetting` value for log display.
fn format_setting_value(es: &openshell_core::proto::EffectiveSetting) -> String {
    use openshell_core::proto::setting_value;
    match es.value.as_ref().and_then(|sv| sv.value.as_ref()) {
        None => "<unset>".to_string(),
        Some(setting_value::Value::StringValue(v)) => v.clone(),
        Some(setting_value::Value::BoolValue(v)) => v.to_string(),
        Some(setting_value::Value::IntValue(v)) => v.to_string(),
        Some(setting_value::Value::BytesValue(_)) => "<bytes>".to_string(),
    }
}

#[cfg(test)]
#[allow(
    clippy::needless_raw_string_hashes,
    clippy::iter_on_single_items,
    clippy::similar_names,
    clippy::manual_string_new,
    clippy::doc_markdown,
    reason = "Test code: test fixtures often use idiomatic forms not flagged in production."
)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use nix::unistd::{Group, User};
    use openshell_core::policy::{FilesystemPolicy, LandlockPolicy, ProcessPolicy};
    #[cfg(unix)]
    use std::os::unix::fs::{MetadataExt, symlink};
    use temp_env::with_vars;

    use std::sync::{LazyLock, Mutex};

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn bundle_to_resolved_routes_converts_all_fields() {
        let bundle = openshell_core::proto::GetInferenceBundleResponse {
            routes: vec![
                openshell_core::proto::ResolvedRoute {
                    name: "frontier".to_string(),
                    base_url: "https://api.example.com/v1".to_string(),
                    api_key: "sk-test-key".to_string(),
                    model_id: "gpt-4".to_string(),
                    protocols: vec![
                        "openai_chat_completions".to_string(),
                        "openai_responses".to_string(),
                    ],
                    provider_type: "openai".to_string(),
                    timeout_secs: 0,
                },
                openshell_core::proto::ResolvedRoute {
                    name: "local".to_string(),
                    base_url: "http://vllm:8000/v1".to_string(),
                    api_key: "local-key".to_string(),
                    model_id: "llama-3".to_string(),
                    protocols: vec!["openai_chat_completions".to_string()],
                    provider_type: String::new(),
                    timeout_secs: 120,
                },
            ],
            revision: "abc123".to_string(),
            generated_at_ms: 1000,
        };

        let routes = bundle_to_resolved_routes(&bundle);

        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].endpoint, "https://api.example.com/v1");
        assert_eq!(routes[0].model, "gpt-4");
        assert_eq!(routes[0].api_key, "sk-test-key");
        assert_eq!(
            routes[0].auth,
            openshell_core::inference::AuthHeader::Bearer
        );
        assert_eq!(
            routes[0].protocols,
            vec!["openai_chat_completions", "openai_responses"]
        );
        assert_eq!(
            routes[0].timeout,
            openshell_router::config::DEFAULT_ROUTE_TIMEOUT,
            "timeout_secs=0 should map to default"
        );
        assert_eq!(routes[1].endpoint, "http://vllm:8000/v1");
        assert_eq!(
            routes[1].auth,
            openshell_core::inference::AuthHeader::Bearer
        );
        assert_eq!(
            routes[1].timeout,
            Duration::from_secs(120),
            "timeout_secs=120 should map to 120s"
        );
    }

    #[test]
    fn bundle_to_resolved_routes_handles_empty_bundle() {
        let bundle = openshell_core::proto::GetInferenceBundleResponse {
            routes: vec![],
            revision: "empty".to_string(),
            generated_at_ms: 0,
        };

        let routes = bundle_to_resolved_routes(&bundle);
        assert!(routes.is_empty());
    }

    #[test]
    fn bundle_to_resolved_routes_preserves_name_field() {
        let bundle = openshell_core::proto::GetInferenceBundleResponse {
            routes: vec![openshell_core::proto::ResolvedRoute {
                name: "sandbox-system".to_string(),
                base_url: "https://api.example.com/v1".to_string(),
                api_key: "key".to_string(),
                model_id: "model".to_string(),
                protocols: vec!["openai_chat_completions".to_string()],
                provider_type: "openai".to_string(),
                timeout_secs: 0,
            }],
            revision: "rev".to_string(),
            generated_at_ms: 0,
        };

        let routes = bundle_to_resolved_routes(&bundle);
        assert_eq!(routes[0].name, "sandbox-system");
    }

    #[test]
    fn routes_segregated_by_name() {
        let routes = vec![
            openshell_router::config::ResolvedRoute {
                name: "inference.local".to_string(),
                endpoint: "https://api.openai.com/v1".to_string(),
                model: "gpt-4o".to_string(),
                api_key: "key1".to_string(),
                protocols: vec!["openai_chat_completions".to_string()],
                auth: openshell_core::inference::AuthHeader::Bearer,
                default_headers: vec![],
                passthrough_headers: vec![],
                timeout: openshell_router::config::DEFAULT_ROUTE_TIMEOUT,
            },
            openshell_router::config::ResolvedRoute {
                name: "sandbox-system".to_string(),
                endpoint: "https://api.anthropic.com/v1".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                api_key: "key2".to_string(),
                protocols: vec!["anthropic_messages".to_string()],
                auth: openshell_core::inference::AuthHeader::Custom("x-api-key"),
                default_headers: vec![],
                passthrough_headers: vec![],
                timeout: openshell_router::config::DEFAULT_ROUTE_TIMEOUT,
            },
        ];

        let (user, system) = partition_routes(routes);
        assert_eq!(user.len(), 1);
        assert_eq!(user[0].name, "inference.local");
        assert_eq!(system.len(), 1);
        assert_eq!(system[0].name, "sandbox-system");
    }

    // -- build_inference_context tests --

    #[tokio::test]
    async fn build_inference_context_route_file_loads_routes() {
        use std::io::Write;

        let yaml = r#"
routes:
  - name: inference.local
    endpoint: http://localhost:8000/v1
    model: llama-3
    protocols: [openai_chat_completions]
    api_key: test-key
"#;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();
        let path = f.path().to_str().unwrap();

        let ctx = build_inference_context(None, None, Some(path))
            .await
            .expect("should load routes from file");

        let ctx = ctx.expect("context should be Some");
        let cache = ctx.route_cache();
        let routes = cache.read().await;
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].endpoint, "http://localhost:8000/v1");
    }

    #[tokio::test]
    async fn build_inference_context_empty_route_file_returns_none() {
        use std::io::Write;

        // Route file with empty routes list → inference routing disabled (not an error)
        let yaml = "routes: []\n";
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();
        let path = f.path().to_str().unwrap();

        let ctx = build_inference_context(None, None, Some(path))
            .await
            .expect("empty routes file should not error");
        assert!(
            ctx.is_none(),
            "empty routes should disable inference routing"
        );
    }

    #[tokio::test]
    async fn build_inference_context_no_sources_returns_none() {
        let ctx = build_inference_context(None, None, None)
            .await
            .expect("should succeed with None");

        assert!(ctx.is_none(), "no sources should return None");
    }

    #[tokio::test]
    async fn build_inference_context_route_file_overrides_cluster() {
        use std::io::Write;

        let yaml = r#"
routes:
  - name: inference.local
    endpoint: http://localhost:9999/v1
    model: file-model
    protocols: [openai_chat_completions]
    api_key: file-key
"#;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();
        let path = f.path().to_str().unwrap();

        // Even with sandbox_id and endpoint, route_file takes precedence
        let ctx = build_inference_context(Some("sb-1"), None, Some(path))
            .await
            .expect("should load from file");

        let ctx = ctx.expect("context should be Some");
        let cache = ctx.route_cache();
        let routes = cache.read().await;
        assert_eq!(routes[0].endpoint, "http://localhost:9999/v1");
    }

    #[test]
    fn infer_route_source_prefers_file_mode() {
        assert_eq!(
            infer_route_source(Some("sb-1"), true, Some("routes.yaml")),
            InferenceRouteSource::File
        );
    }

    #[test]
    fn infer_route_source_cluster_requires_id_and_endpoint() {
        assert_eq!(
            infer_route_source(Some("sb-1"), true, None),
            InferenceRouteSource::Cluster
        );
        assert_eq!(
            infer_route_source(Some("sb-1"), false, None),
            InferenceRouteSource::None
        );
        assert_eq!(
            infer_route_source(None, true, None),
            InferenceRouteSource::None
        );
    }

    #[test]
    fn disable_inference_on_empty_routes_depends_on_source() {
        assert!(disable_inference_on_empty_routes(
            InferenceRouteSource::File
        ));
        assert!(!disable_inference_on_empty_routes(
            InferenceRouteSource::Cluster
        ));
        assert!(disable_inference_on_empty_routes(
            InferenceRouteSource::None
        ));
    }

    // ---- Policy disk discovery tests ----

    #[test]
    fn discover_policy_from_nonexistent_path_returns_restrictive_default() {
        let path = std::path::Path::new("/nonexistent/policy.yaml");
        let policy = discover_policy_from_path(path);
        // Restrictive default has no network policies.
        assert!(policy.network_policies.is_empty());
        // But does have filesystem and process policies.
        assert!(policy.filesystem.is_some());
        assert!(policy.process.is_some());
    }

    #[test]
    fn discover_policy_from_valid_yaml_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("policy.yaml");
        std::fs::write(
            &path,
            r#"
version: 1
filesystem_policy:
  include_workdir: false
  read_only:
    - /usr
  read_write:
    - /tmp
network_policies:
  test:
    name: test
    endpoints:
      - { host: example.com, port: 443 }
    binaries:
      - { path: /usr/bin/curl }
"#,
        )
        .unwrap();

        let policy = discover_policy_from_path(&path);
        assert_eq!(policy.network_policies.len(), 1);
        assert!(policy.network_policies.contains_key("test"));
        let fs = policy.filesystem.unwrap();
        assert!(!fs.include_workdir);
    }

    #[test]
    fn discover_policy_from_invalid_yaml_returns_restrictive_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("policy.yaml");
        std::fs::write(&path, "this is not valid yaml: [[[").unwrap();

        let policy = discover_policy_from_path(&path);
        // Falls back to restrictive default.
        assert!(policy.network_policies.is_empty());
        assert!(policy.filesystem.is_some());
    }

    #[test]
    fn discover_policy_from_unsafe_yaml_falls_back_to_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("policy.yaml");
        std::fs::write(
            &path,
            r#"
version: 1
process:
  run_as_user: root
  run_as_group: root
filesystem_policy:
  include_workdir: true
  read_only:
    - /usr
  read_write:
    - /tmp
"#,
        )
        .unwrap();

        let policy = discover_policy_from_path(&path);
        // Falls back to restrictive default because of root user.
        let proc = policy.process.unwrap();
        assert_eq!(proc.run_as_user, "sandbox");
        assert_eq!(proc.run_as_group, "sandbox");
    }

    #[test]
    fn discover_policy_restrictive_default_blocks_network() {
        // In cluster mode we keep proxy mode enabled so `inference.local`
        // can always be routed through proxy/OPA controls.
        let proto = openshell_policy::restrictive_default_policy();
        let local_policy = SandboxPolicy::try_from(proto).expect("conversion should succeed");
        assert!(matches!(local_policy.network.mode, NetworkMode::Proxy));
    }

    // ---- Route refresh interval + revision tests ----

    #[test]
    fn default_route_refresh_interval_is_five_seconds() {
        assert_eq!(DEFAULT_ROUTE_REFRESH_INTERVAL_SECS, 5);
    }

    #[test]
    fn route_refresh_interval_uses_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        with_vars(
            [("OPENSHELL_ROUTE_REFRESH_INTERVAL_SECS", Some("9"))],
            || {
                assert_eq!(route_refresh_interval_secs(), 9);
            },
        );
    }

    #[test]
    fn route_refresh_interval_rejects_zero() {
        let _guard = ENV_LOCK.lock().unwrap();
        with_vars(
            [("OPENSHELL_ROUTE_REFRESH_INTERVAL_SECS", Some("0"))],
            || {
                assert_eq!(
                    route_refresh_interval_secs(),
                    DEFAULT_ROUTE_REFRESH_INTERVAL_SECS
                );
            },
        );
    }

    #[test]
    fn route_refresh_interval_rejects_invalid_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        with_vars(
            [("OPENSHELL_ROUTE_REFRESH_INTERVAL_SECS", Some("abc"))],
            || {
                assert_eq!(
                    route_refresh_interval_secs(),
                    DEFAULT_ROUTE_REFRESH_INTERVAL_SECS
                );
            },
        );
    }

    #[tokio::test]
    async fn route_cache_preserves_content_when_not_written() {
        use std::sync::Arc;
        use tokio::sync::RwLock;

        let routes = vec![openshell_router::config::ResolvedRoute {
            name: "inference.local".to_string(),
            endpoint: "http://original:8000/v1".to_string(),
            model: "original-model".to_string(),
            api_key: "key".to_string(),
            auth: openshell_core::inference::AuthHeader::Bearer,
            protocols: vec!["openai_chat_completions".to_string()],
            default_headers: vec![],
            passthrough_headers: vec![],
            timeout: openshell_router::config::DEFAULT_ROUTE_TIMEOUT,
        }];

        let cache = Arc::new(RwLock::new(routes));

        // Verify the cache preserves its content — the revision-based skip
        // logic in spawn_route_refresh ensures the cache is only written
        // when the revision actually changes.
        let read = cache.read().await;
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].model, "original-model");
    }

    #[cfg(unix)]
    fn sandbox_policy_with_read_write(
        path: std::path::PathBuf,
        run_as_user: Option<String>,
        run_as_group: Option<String>,
    ) -> SandboxPolicy {
        SandboxPolicy {
            version: 1,
            filesystem: FilesystemPolicy {
                read_only: vec![],
                read_write: vec![path],
                include_workdir: false,
            },
            network: NetworkPolicy::default(),
            landlock: LandlockPolicy::default(),
            process: ProcessPolicy {
                run_as_user,
                run_as_group,
            },
        }
    }

    #[cfg(unix)]
    #[test]
    fn prepare_read_write_path_creates_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing").join("nested");

        assert!(prepare_read_write_path(&missing).unwrap());
        assert!(missing.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn prepare_read_write_path_preserves_existing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("existing");
        std::fs::create_dir(&existing).unwrap();

        assert!(!prepare_read_write_path(&existing).unwrap());
        assert!(existing.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn prepare_read_write_path_rejects_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        let link = dir.path().join("link");
        std::fs::create_dir(&target).unwrap();
        symlink(&target, &link).unwrap();

        let error = prepare_read_write_path(&link).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("is a symlink — refusing to chown"),
            "unexpected error: {error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn prepare_filesystem_skips_chown_for_existing_read_write_paths() {
        if nix::unistd::geteuid().is_root() {
            return;
        }

        let current_user = User::from_uid(nix::unistd::geteuid())
            .unwrap()
            .expect("current user entry");
        let restricted_group = Group::from_gid(nix::unistd::Gid::from_raw(0))
            .unwrap()
            .expect("gid 0 group entry");
        if restricted_group.gid == nix::unistd::getegid() {
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("existing");
        std::fs::create_dir(&existing).unwrap();
        let before = std::fs::metadata(&existing).unwrap();

        let policy = sandbox_policy_with_read_write(
            existing.clone(),
            Some(current_user.name),
            Some(restricted_group.name),
        );

        prepare_filesystem(&policy).expect("existing path should not be re-owned");

        let after = std::fs::metadata(&existing).unwrap();
        assert_eq!(after.uid(), before.uid());
        assert_eq!(after.gid(), before.gid());
    }

    #[test]
    fn supervisor_mode_default_string_parses_with_both_components() {
        let parsed: SupervisorMode = SupervisorMode::DEFAULT_STR.parse().unwrap();
        assert!(parsed.has(SupervisorComponent::Network));
        assert!(parsed.has(SupervisorComponent::Process));
    }

    #[test]
    fn supervisor_mode_accepts_individual_components() {
        let network: SupervisorMode = "network".parse().unwrap();
        assert!(network.has(SupervisorComponent::Network));
        assert!(!network.has(SupervisorComponent::Process));

        let process: SupervisorMode = "process".parse().unwrap();
        assert!(!process.has(SupervisorComponent::Network));
        assert!(process.has(SupervisorComponent::Process));
    }

    #[test]
    fn supervisor_mode_canonicalises_reversed_pair() {
        let parsed: SupervisorMode = "process,network".parse().unwrap();
        assert_eq!(parsed.to_string(), "network,process");
    }

    #[test]
    fn supervisor_mode_tolerates_whitespace_and_duplicates() {
        let pair: SupervisorMode = " network , process ".parse().unwrap();
        assert!(pair.has(SupervisorComponent::Network));
        assert!(pair.has(SupervisorComponent::Process));

        let dupe: SupervisorMode = "network,network".parse().unwrap();
        assert!(dupe.has(SupervisorComponent::Network));
        assert!(!dupe.has(SupervisorComponent::Process));
    }

    #[test]
    fn supervisor_mode_rejects_empty_and_unknown() {
        assert!("".parse::<SupervisorMode>().is_err());
        assert!(",".parse::<SupervisorMode>().is_err());
        assert!("netw0rk".parse::<SupervisorMode>().is_err());
        assert!("network,bogus".parse::<SupervisorMode>().is_err());
    }

    #[test]
    fn supervisor_mode_display_round_trips_through_from_str() {
        for input in ["network", "process", "network,process"] {
            let parsed: SupervisorMode = input.parse().unwrap();
            assert_eq!(parsed.to_string(), input);
        }
    }
}
