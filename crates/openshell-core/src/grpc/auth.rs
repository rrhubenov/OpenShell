// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bearer-token auth machinery for the gateway gRPC channel.
//!
//! Every request carries a gateway-minted JWT in the `Authorization` header.
//! The token is resolved at startup from one of three sources:
//!
//! 1. `OPENSHELL_SANDBOX_TOKEN` — raw JWT in the env (test harness path).
//! 2. `OPENSHELL_SANDBOX_TOKEN_FILE` — file containing the JWT (Docker /
//!    Podman / VM drivers write this to a bundle file at sandbox-create
//!    time).
//! 3. `OPENSHELL_K8S_SA_TOKEN_FILE` — projected `ServiceAccount` JWT; the
//!    supervisor exchanges it for a gateway JWT via `IssueSandboxToken`
//!    once at startup.
//!
//! The resolved gateway JWT is held in process memory thereafter and
//! injected on every outbound call by [`AuthInterceptor`].

use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use miette::{IntoDiagnostic, Result, WrapErr};
use tonic::Status;
use tonic::metadata::AsciiMetadataValue;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Channel;
use tracing::{debug, info, warn};

use crate::proto::{
    IssueSandboxTokenRequest, RefreshSandboxTokenRequest, open_shell_client::OpenShellClient,
};
use crate::sandbox_env;

/// Channel type after the [`AuthInterceptor`] is applied. Aliased so the
/// generated client type signatures stay readable.
pub type AuthedChannel = InterceptedService<Channel, AuthInterceptor>;

/// Shared, refreshable Bearer header. All [`AuthInterceptor`] clones read
/// the same slot, so the renewal task can replace the token in place without
/// rebuilding the channel.
pub type TokenSlot = Arc<RwLock<AsciiMetadataValue>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenSource {
    Env,
    File,
    K8sServiceAccount,
}

#[derive(Debug)]
struct AcquiredToken {
    token: String,
    source: TokenSource,
}

/// Process-wide token slot. Initialized by the first
/// [`connect_authed_channel`](super::connect_authed_channel) call and shared
/// with every subsequent client and the renewal loop.
static TOKEN_SLOT: OnceLock<TokenSlot> = OnceLock::new();

/// Source used to acquire the process-wide token slot.
static TOKEN_SOURCE: OnceLock<TokenSource> = OnceLock::new();

/// Serializes the first token acquisition. Several supervisor subsystems
/// connect during startup; without this guard they can all observe an empty
/// [`TOKEN_SLOT`] and perform duplicate K8s bootstrap exchanges.
static TOKEN_INIT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// One-shot guard so the renewal loop spawns at most once per process.
pub static REFRESH_SPAWNED: OnceLock<()> = OnceLock::new();

fn install_token_slot(token: &str) -> Result<TokenSlot> {
    let bearer = AsciiMetadataValue::try_from(format!("Bearer {token}"))
        .into_diagnostic()
        .wrap_err("sandbox JWT contained characters not valid for a header value")?;
    if let Some(existing) = TOKEN_SLOT.get() {
        *existing.write().expect("token slot poisoned") = bearer;
        return Ok(existing.clone());
    }
    let slot: TokenSlot = Arc::new(RwLock::new(bearer));
    let _ = TOKEN_SLOT.set(slot.clone());
    Ok(TOKEN_SLOT.get().cloned().unwrap_or(slot))
}

/// gRPC interceptor that injects `authorization: Bearer <token>` on every
/// outbound request. The token lives in a shared [`TokenSlot`] so the renewal
/// task can replace it without rebuilding clients.
#[derive(Clone)]
pub struct AuthInterceptor {
    bearer: TokenSlot,
}

impl AuthInterceptor {
    pub(crate) fn new(bearer: TokenSlot) -> Self {
        Self { bearer }
    }
}

impl tonic::service::Interceptor for AuthInterceptor {
    fn call(
        &mut self,
        mut req: tonic::Request<()>,
    ) -> std::result::Result<tonic::Request<()>, Status> {
        let bearer = self
            .bearer
            .read()
            .expect("auth interceptor token slot poisoned")
            .clone();
        req.metadata_mut().insert("authorization", bearer);
        Ok(req)
    }
}

pub async fn token_slot(
    endpoint: &str,
    plain_channel: &Channel,
) -> Result<(TokenSlot, TokenSource)> {
    if let Some(existing) = TOKEN_SLOT.get() {
        let source = TOKEN_SOURCE.get().copied().unwrap_or(TokenSource::Env);
        return Ok((existing.clone(), source));
    }

    let _guard = TOKEN_INIT_LOCK.lock().await;

    if let Some(existing) = TOKEN_SLOT.get() {
        let source = TOKEN_SOURCE.get().copied().unwrap_or(TokenSource::Env);
        return Ok((existing.clone(), source));
    }

    let acquired = acquire_sandbox_token(endpoint, plain_channel).await?;
    let slot = install_token_slot(&acquired.token)?;
    let _ = TOKEN_SOURCE.set(acquired.source);
    Ok((slot, acquired.source))
}

/// Resolve the sandbox JWT used to authenticate every outbound RPC.
///
/// `endpoint` is logged on errors but never used for transport here; the
/// actual network call lives inside this function only on the K8s
/// bootstrap path, which uses `plain_channel` to call `IssueSandboxToken`
/// once before the steady-state Bearer-authenticated channel is built.
async fn acquire_sandbox_token(endpoint: &str, plain_channel: &Channel) -> Result<AcquiredToken> {
    if let Ok(t) = std::env::var(sandbox_env::SANDBOX_TOKEN)
        && !t.is_empty()
    {
        debug!(source = "env", "loaded sandbox token");
        return Ok(AcquiredToken {
            token: t,
            source: TokenSource::Env,
        });
    }

    if let Ok(path) = std::env::var(sandbox_env::SANDBOX_TOKEN_FILE)
        && !path.is_empty()
    {
        let contents = std::fs::read_to_string(&path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to read sandbox token from {path}"))?;
        debug!(source = "file", path = %path, "loaded sandbox token");
        return Ok(AcquiredToken {
            token: contents.trim().to_string(),
            source: TokenSource::File,
        });
    }

    if let Ok(sa_path) = std::env::var(sandbox_env::K8S_SA_TOKEN_FILE)
        && !sa_path.is_empty()
    {
        return Ok(AcquiredToken {
            token: acquire_k8s_sandbox_token(endpoint, plain_channel, &sa_path).await?,
            source: TokenSource::K8sServiceAccount,
        });
    }

    Err(miette::miette!(
        "no sandbox token source available — set one of {}, {}, or {}",
        sandbox_env::SANDBOX_TOKEN,
        sandbox_env::SANDBOX_TOKEN_FILE,
        sandbox_env::K8S_SA_TOKEN_FILE,
    ))
}

async fn acquire_k8s_sandbox_token(
    endpoint: &str,
    plain_channel: &Channel,
    sa_path: &str,
) -> Result<String> {
    let sa_token = std::fs::read_to_string(sa_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("failed to read K8s SA token from {sa_path}"))?
        .trim()
        .to_string();
    info!(endpoint = %endpoint, "exchanging K8s ServiceAccount token for sandbox JWT");
    // The bootstrap exchange uses a one-off interceptor pinned to the
    // SA token; the resulting gateway JWT becomes the value in the
    // shared `TOKEN_SLOT` once `connect_authed_channel` returns.
    let bootstrap_slot: TokenSlot = Arc::new(RwLock::new(
        AsciiMetadataValue::try_from(format!("Bearer {sa_token}"))
            .into_diagnostic()
            .wrap_err("SA token contained characters not valid for a header value")?,
    ));
    let interceptor = AuthInterceptor::new(bootstrap_slot);
    let bootstrap = InterceptedService::new(plain_channel.clone(), interceptor);
    let mut client = OpenShellClient::new(bootstrap);
    let resp = client
        .issue_sandbox_token(IssueSandboxTokenRequest {})
        .await
        .into_diagnostic()
        .wrap_err("IssueSandboxToken bootstrap exchange failed")?;
    Ok(resp.into_inner().token)
}

/// Background task that renews the sandbox JWT at ~80% of its remaining
/// lifetime. The new token replaces the value in [`TOKEN_SLOT`], so all
/// in-flight and future clients pick it up on their next request. The
/// loop never panics: every failure is logged and re-attempted after a
/// bounded backoff.
pub async fn refresh_token_loop(
    channel: AuthedChannel,
    slot: TokenSlot,
    source: TokenSource,
    endpoint: String,
    plain_channel: Channel,
) {
    let mut client = OpenShellClient::new(channel);
    loop {
        let sleep = compute_refresh_delay(&slot);
        tokio::time::sleep(sleep).await;
        match client
            .refresh_sandbox_token(RefreshSandboxTokenRequest {})
            .await
        {
            Ok(resp) => {
                let new_token = resp.into_inner().token;
                match AsciiMetadataValue::try_from(format!("Bearer {new_token}")) {
                    Ok(value) => {
                        if let Ok(mut guard) = slot.write() {
                            *guard = value;
                            info!("renewed gateway sandbox JWT in-place");
                        }
                    }
                    Err(e) => warn!(error = %e, "refreshed JWT contained invalid header bytes"),
                }
            }
            Err(status) => {
                if status.code() == tonic::Code::Unauthenticated
                    && source == TokenSource::K8sServiceAccount
                {
                    if let Some(sa_path) = std::env::var(sandbox_env::K8S_SA_TOKEN_FILE)
                        .ok()
                        .filter(|p| !p.is_empty())
                    {
                        match acquire_k8s_sandbox_token(&endpoint, &plain_channel, &sa_path).await {
                            Ok(new_token) => {
                                match AsciiMetadataValue::try_from(format!("Bearer {new_token}")) {
                                    Ok(value) => {
                                        if let Ok(mut guard) = slot.write() {
                                            *guard = value;
                                            info!(
                                                "rebootstrapped gateway sandbox JWT after refresh authentication failure"
                                            );
                                            continue;
                                        }
                                    }
                                    Err(e) => warn!(
                                        error = %e,
                                        "rebootstrapped JWT contained invalid header bytes"
                                    ),
                                }
                            }
                            Err(e) => warn!(
                                error = %e,
                                "K8s ServiceAccount bootstrap retry failed after refresh authentication failure"
                            ),
                        }
                    } else {
                        warn!(
                            "RefreshSandboxToken returned Unauthenticated and K8s SA token file is unavailable"
                        );
                    }
                } else if status.code() == tonic::Code::Unauthenticated {
                    warn!(
                        source = ?source,
                        "RefreshSandboxToken returned Unauthenticated; static token sources cannot rebootstrap automatically"
                    );
                }
                warn!(error = %status, "RefreshSandboxToken failed; will retry");
                // Backoff so we don't spin against a sustained failure.
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }
    }
}

/// Compute the next refresh delay: 80 % of the time remaining until the
/// current token's `exp`, plus up to 10 % jitter, with a small lower bound
/// for already-expired tokens and capped at 12 h. If the token can't be parsed
/// (legacy/non-JWT bearer)
/// default to 6 h.
fn compute_refresh_delay(slot: &TokenSlot) -> Duration {
    let token = slot
        .read()
        .ok()
        .and_then(|v| v.to_str().ok().map(str::to_string))
        .unwrap_or_default();
    let bearer = token.strip_prefix("Bearer ").unwrap_or(&token);
    let now_ms = i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
    )
    .unwrap_or(i64::MAX);
    let remaining_ms = parse_jwt_exp_ms(bearer).map_or(21_600_000, |exp| exp - now_ms); // 6 h fallback
    let mut delay_ms = if remaining_ms <= 0 {
        1_000
    } else {
        (remaining_ms * 8 / 10).clamp(1_000, 43_200_000)
    };
    // Up to 10 % jitter, derived deterministically from token bytes so
    // unit tests are reproducible without injecting an RNG.
    let jitter_pct = (token.len() % 10) as u64;
    let jitter_ms = (u64::try_from(delay_ms).unwrap_or(0) * jitter_pct) / 100;
    delay_ms = delay_ms.saturating_add(i64::try_from(jitter_ms).unwrap_or(0));
    Duration::from_millis(u64::try_from(delay_ms).unwrap_or(0))
}

/// Decode the `exp` claim from a JWT without verifying its signature.
/// Returns the expiry in milliseconds since the Unix epoch, or `None` if
/// the token is not a parseable JWT.
fn parse_jwt_exp_ms(jwt: &str) -> Option<i64> {
    use base64::Engine;
    let mut parts = jwt.splitn(3, '.');
    let _header = parts.next()?;
    let payload_b64 = parts.next()?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .ok()?;
    let value: serde_json::Value = serde_json::from_slice(&decoded).ok()?;
    let exp_secs = value.get("exp")?.as_i64()?;
    exp_secs.checked_mul(1000)
}

#[cfg(test)]
mod auth_tests {
    use super::*;

    #[test]
    fn parse_jwt_exp_reads_unsigned_payload() {
        use base64::Engine as _;
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(br#"{"exp":1234567890,"sandbox_id":"sb-1"}"#);
        let token = format!("h.{payload}.sig");
        assert_eq!(parse_jwt_exp_ms(&token), Some(1_234_567_890_000));
    }

    #[test]
    fn parse_jwt_exp_returns_none_for_malformed_token() {
        assert!(parse_jwt_exp_ms("not-a-jwt").is_none());
        assert!(parse_jwt_exp_ms("only.two").is_none());
        assert!(parse_jwt_exp_ms("a.!!!.c").is_none());
    }

    #[test]
    fn compute_refresh_delay_uses_80_percent_when_token_present() {
        // Build a JWT whose exp is 1000 seconds in the future. With 0-jitter
        // the delay should be roughly 800 seconds.
        use base64::Engine as _;
        let now_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let exp = now_s + 1000;
        let payload_json = format!(r#"{{"exp":{exp}}}"#);
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload_json);
        let token = format!("h.{payload}.s");
        let bearer = AsciiMetadataValue::try_from(format!("Bearer {token}")).unwrap();
        let slot: TokenSlot = Arc::new(RwLock::new(bearer));
        let delay = compute_refresh_delay(&slot);
        // 800 s baseline + up to 10 % jitter → 800..=880 s, with some slack
        // for the 1-second resolution of the exp claim.
        let secs = delay.as_secs();
        assert!(
            (700..=900).contains(&secs),
            "expected 80%-of-1000s delay, got {secs}s"
        );
    }

    #[test]
    fn compute_refresh_delay_uses_short_delay_for_expired_token() {
        // Already-expired token still produces a small positive delay so the
        // loop doesn't busy-spin.
        use base64::Engine as _;
        let exp = 1; // past
        let payload =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(format!(r#"{{"exp":{exp}}}"#));
        let token = format!("h.{payload}.s");
        let bearer = AsciiMetadataValue::try_from(format!("Bearer {token}")).unwrap();
        let slot: TokenSlot = Arc::new(RwLock::new(bearer));
        let delay = compute_refresh_delay(&slot);
        assert!((1..60).contains(&delay.as_secs()));
    }

    #[test]
    fn compute_refresh_delay_supports_short_token_ttl() {
        use base64::Engine as _;
        let now_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let exp = now_s + 30;
        let payload_json = format!(r#"{{"exp":{exp}}}"#);
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload_json);
        let token = format!("h.{payload}.s");
        let bearer = AsciiMetadataValue::try_from(format!("Bearer {token}")).unwrap();
        let slot: TokenSlot = Arc::new(RwLock::new(bearer));
        let delay = compute_refresh_delay(&slot);
        assert!(
            delay.as_secs() < 30,
            "expected refresh before 30s expiry, got {delay:?}",
        );
    }
}
