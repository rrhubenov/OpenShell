// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Credential placeholder resolution.
//!
//! [`SecretResolver`] maps `openshell:resolve:env:KEY` placeholder strings (and
//! the provider-shaped `OPENSHELL-RESOLVE-ENV-KEY` alias form) back to the
//! real secret values. The supervisor seeds the child process with placeholders
//! instead of raw secrets and uses this resolver at the egress proxy boundary
//! to swap them back in just before bytes leave the sandbox.
//!
//! HTTP-specific rewriting (request line, header block, query parameters) lives
//! in the supervisor crate — this module only owns the placeholder grammar,
//! resolver, and the low-level token-extraction helpers needed by both layers.

use base64::Engine as _;
use std::collections::HashMap;
use std::fmt;

pub const PLACEHOLDER_PREFIX: &str = "openshell:resolve:env:";
pub const PROVIDER_ALIAS_MARKER: &str = "OPENSHELL-RESOLVE-ENV-";

/// Public access to the placeholder prefix for fail-closed scanning in other modules.
pub const PLACEHOLDER_PREFIX_PUBLIC: &str = PLACEHOLDER_PREFIX;
pub const PROVIDER_ALIAS_MARKER_PUBLIC: &str = PROVIDER_ALIAS_MARKER;

/// Characters that are valid in an env var key name (used to extract
/// placeholder boundaries within concatenated strings like path segments).
fn is_env_key_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_alias_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'~')
}

pub fn contains_raw_reserved_marker(value: &str) -> bool {
    value.contains(PLACEHOLDER_PREFIX) || value.contains(PROVIDER_ALIAS_MARKER)
}

pub fn contains_reserved_credential_marker(value: &str) -> bool {
    if contains_raw_reserved_marker(value) {
        return true;
    }
    let decoded = percent_decode(value);
    contains_raw_reserved_marker(&decoded)
}

// ---------------------------------------------------------------------------
// Error and result types
// ---------------------------------------------------------------------------

/// Error returned when a placeholder cannot be resolved or a resolved secret
/// contains prohibited characters.
#[derive(Debug)]
pub struct UnresolvedPlaceholderError {
    pub location: &'static str, // "header", "query_param", "path"
}

impl fmt::Display for UnresolvedPlaceholderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unresolved credential placeholder in {}: detected reserved credential token that could not be resolved",
            self.location
        )
    }
}

/// Result of rewriting an HTTP header block with credential resolution.
#[derive(Debug)]
pub struct RewriteResult {
    /// The rewritten HTTP bytes (headers + body overflow).
    pub rewritten: Vec<u8>,
    /// A redacted version of the request target for logging.
    /// Contains `[CREDENTIAL]` in place of resolved credential values.
    /// `None` if the target was not modified.
    // Kept on the public result struct as part of the API contract; consumed
    // selectively by callers that emit redacted logs.
    #[allow(dead_code)]
    pub redacted_target: Option<String>,
}

/// Result of rewriting a request target for OPA evaluation.
#[derive(Debug)]
pub struct RewriteTargetResult {
    /// The resolved target (real secrets) — for upstream forwarding only.
    pub resolved: String,
    /// The redacted target (`[CREDENTIAL]` in place of secrets) — for OPA + logs.
    pub redacted: String,
}

// ---------------------------------------------------------------------------
// SecretResolver
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct SecretResolver {
    by_placeholder: HashMap<String, Secret>,
}

#[derive(Clone)]
struct Secret {
    value: String,
    expires_at_ms: i64,
}

fn current_time_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

// Manual `Debug` impl: the auto-derived `Debug` would format the
// `by_placeholder` map, exposing both placeholder keys (which reveal which
// provider env var names are configured) and the resolved secret values
// themselves. Any accidental `{:?}` in a tracing call, or a
// derived `Debug` on a containing struct, would write secrets to logs.
//
// We expose only the count of registered placeholders without leaking anything.
impl fmt::Debug for SecretResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecretResolver")
            .field("placeholders", &self.by_placeholder.len())
            .finish_non_exhaustive() // Use to show that the struct is not empty
    }
}

impl SecretResolver {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_provider_env(
        provider_env: HashMap<String, String>,
    ) -> (HashMap<String, String>, Option<Self>) {
        Self::from_provider_env_for_revision(provider_env, HashMap::new(), 0)
    }

    pub fn from_provider_env_for_revision(
        provider_env: HashMap<String, String>,
        credential_expires_at_ms: HashMap<String, i64>,
        revision: u64,
    ) -> (HashMap<String, String>, Option<Self>) {
        Self::from_provider_env_for_revision_with_current_aliases(
            provider_env,
            credential_expires_at_ms,
            revision,
            false,
        )
    }

    pub fn from_provider_env_for_current_revision(
        provider_env: HashMap<String, String>,
        credential_expires_at_ms: HashMap<String, i64>,
        revision: u64,
    ) -> (HashMap<String, String>, Option<Self>, Option<Self>) {
        if revision == 0 {
            let (child_env, current_resolver) =
                Self::from_provider_env_for_revision_with_current_aliases(
                    provider_env,
                    credential_expires_at_ms,
                    0,
                    true,
                );
            return (child_env, None, current_resolver);
        }
        let provider_env_for_current = provider_env.clone();
        let credential_expires_at_ms_for_current = credential_expires_at_ms.clone();
        let (child_env, revision_resolver) =
            Self::from_provider_env_for_revision_with_current_aliases(
                provider_env,
                credential_expires_at_ms,
                revision,
                false,
            );
        let (_, current_resolver) = Self::from_provider_env_for_revision_with_current_aliases(
            provider_env_for_current,
            credential_expires_at_ms_for_current,
            revision,
            true,
        );
        (child_env, revision_resolver, current_resolver)
    }

    fn from_provider_env_for_revision_with_current_aliases(
        provider_env: HashMap<String, String>,
        credential_expires_at_ms: HashMap<String, i64>,
        revision: u64,
        include_current_aliases: bool,
    ) -> (HashMap<String, String>, Option<Self>) {
        if provider_env.is_empty() {
            return (HashMap::new(), None);
        }

        let mut child_env = HashMap::with_capacity(provider_env.len());
        let mut by_placeholder = HashMap::with_capacity(provider_env.len());

        for (key, value) in provider_env {
            let placeholder = placeholder_for_env_key_for_revision(&key, revision);
            let secret = Secret {
                value,
                expires_at_ms: credential_expires_at_ms
                    .get(&key)
                    .copied()
                    .unwrap_or_default(),
            };
            child_env.insert(key.clone(), placeholder.clone());
            by_placeholder.insert(placeholder, secret.clone());
            if include_current_aliases && revision != 0 {
                by_placeholder.insert(placeholder_for_env_key(&key), secret.clone());
            }
        }

        (child_env, Some(Self { by_placeholder }))
    }

    pub fn merge<'a>(resolvers: impl IntoIterator<Item = &'a Self>) -> Option<Self> {
        let mut by_placeholder = HashMap::new();
        for resolver in resolvers {
            by_placeholder.extend(resolver.by_placeholder.clone());
        }
        if by_placeholder.is_empty() {
            None
        } else {
            Some(Self { by_placeholder })
        }
    }

    /// Resolve a placeholder string to the real secret value.
    ///
    /// Returns `None` if the placeholder is unknown or the resolved value
    /// contains prohibited control characters (CRLF, null byte).
    pub fn resolve_placeholder(&self, value: &str) -> Option<&str> {
        let secret = if let Some(secret) = self.by_placeholder.get(value) {
            secret
        } else {
            let key = alias_env_key(value)?;
            let canonical = placeholder_for_env_key(key);
            self.by_placeholder.get(&canonical)?
        };
        if secret.expires_at_ms > 0 && secret.expires_at_ms <= current_time_ms() {
            tracing::warn!(
                location = "resolve_placeholder",
                "credential resolution rejected: credential is expired"
            );
            return None;
        }
        match validate_resolved_secret(&secret.value) {
            Ok(s) => Some(s),
            Err(reason) => {
                tracing::warn!(
                    location = "resolve_placeholder",
                    reason,
                    "credential resolution rejected: resolved value contains prohibited characters"
                );
                None
            }
        }
    }

    pub fn rewrite_header_value(
        &self,
        value: &str,
    ) -> Result<Option<String>, UnresolvedPlaceholderError> {
        // Direct placeholder match: `x-api-key: openshell:resolve:env:KEY`
        if let Some(secret) = self.resolve_placeholder(value.trim()) {
            return Ok(Some(secret.to_string()));
        }

        let trimmed = value.trim();

        // Basic auth decoding: `Basic <base64>` where the decoded content
        // contains a placeholder (e.g. `user:openshell:resolve:env:PASS`).
        if let Some(encoded) = trimmed
            .strip_prefix("Basic ")
            .or_else(|| trimmed.strip_prefix("basic "))
            .map(str::trim)
            && let Some(rewritten) = self.rewrite_basic_auth_token(encoded)?
        {
            return Ok(Some(format!("Basic {rewritten}")));
        }

        // Prefixed placeholder: `Bearer openshell:resolve:env:KEY`
        let Some(split_at) = trimmed.find(char::is_whitespace) else {
            if contains_reserved_credential_marker(trimmed) {
                return Err(UnresolvedPlaceholderError { location: "header" });
            }
            return Ok(None);
        };
        let prefix = &trimmed[..split_at];
        let candidate = trimmed[split_at..].trim();
        if let Some(secret) = self.resolve_placeholder(candidate) {
            return Ok(Some(format!("{prefix} {secret}")));
        }

        if contains_reserved_credential_marker(candidate) {
            return Err(UnresolvedPlaceholderError { location: "header" });
        }

        Ok(None)
    }

    pub fn rewrite_text_placeholders(
        &self,
        text: &mut String,
        location: &'static str,
    ) -> Result<usize, UnresolvedPlaceholderError> {
        if !contains_raw_reserved_marker(text) {
            return Ok(0);
        }

        let mut rewritten = String::with_capacity(text.len());
        let mut pos = 0;
        let mut replacements = 0;

        while pos < text.len() {
            let next_canonical = text[pos..].find(PLACEHOLDER_PREFIX).map(|p| pos + p);
            let next_alias = text[pos..].find(PROVIDER_ALIAS_MARKER).map(|marker_pos| {
                let marker_abs = pos + marker_pos;
                alias_start_for_marker(text, marker_abs)
            });
            let Some(abs_start) = [next_canonical, next_alias].into_iter().flatten().min() else {
                rewritten.push_str(&text[pos..]);
                break;
            };

            rewritten.push_str(&text[pos..abs_start]);

            if text[abs_start..].starts_with(PLACEHOLDER_PREFIX) {
                let Some((token_end, token)) = self.credential_token_at(text, abs_start) else {
                    return Err(UnresolvedPlaceholderError { location });
                };
                let Some(secret) = self.resolve_placeholder(token) else {
                    return Err(UnresolvedPlaceholderError { location });
                };
                rewritten.push_str(secret);
                replacements += 1;
                pos = token_end;
                continue;
            }

            if let Some((token_end, token)) = alias_token_at(text, abs_start) {
                let Some(secret) = self.resolve_placeholder(token) else {
                    return Err(UnresolvedPlaceholderError { location });
                };
                rewritten.push_str(secret);
                replacements += 1;
                pos = token_end;
                continue;
            }

            return Err(UnresolvedPlaceholderError { location });
        }

        if contains_raw_reserved_marker(&rewritten) {
            return Err(UnresolvedPlaceholderError { location });
        }

        *text = rewritten;
        Ok(replacements)
    }

    /// Rewrite credential placeholders inside a WebSocket text message.
    ///
    /// The message is mutated only after all placeholders resolve
    /// successfully. The return value is the number of replacements; callers
    /// must not log the rewritten text.
    pub fn rewrite_websocket_text_placeholders(
        &self,
        text: &mut String,
    ) -> Result<usize, UnresolvedPlaceholderError> {
        self.rewrite_text_placeholders(text, "websocket")
    }

    fn credential_token_at<'a>(
        &'a self,
        text: &'a str,
        abs_start: usize,
    ) -> Option<(usize, &'a str)> {
        self.longest_known_token_match(text, abs_start)
            .or_else(|| canonical_token_at(text, abs_start))
            .or_else(|| alias_token_at(text, abs_start))
    }

    fn longest_known_token_match<'a>(
        &'a self,
        text: &str,
        abs_start: usize,
    ) -> Option<(usize, &'a str)> {
        let suffix = &text[abs_start..];
        self.by_placeholder
            .keys()
            .filter_map(|placeholder| {
                if !suffix.starts_with(placeholder) {
                    return None;
                }
                let key_end = abs_start + placeholder.len();
                let boundary_ok = token_boundary_ok(text, abs_start, key_end, placeholder);
                boundary_ok.then_some((key_end, placeholder.as_str()))
            })
            .max_by_key(|(_, placeholder)| placeholder.len())
    }

    /// Decode a Base64-encoded Basic auth token, resolve any placeholders in
    /// the decoded `username:password` string, and re-encode.
    ///
    /// Returns `None` if decoding fails or no placeholders are found.
    fn rewrite_basic_auth_token(
        &self,
        encoded: &str,
    ) -> Result<Option<String>, UnresolvedPlaceholderError> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let Some(decoded_bytes) = b64.decode(encoded.trim()).ok() else {
            return Ok(None);
        };
        let Some(decoded) = std::str::from_utf8(&decoded_bytes).ok() else {
            return Ok(None);
        };

        if !contains_raw_reserved_marker(decoded) {
            return Ok(None);
        }

        let mut rewritten = decoded.to_string();
        let replacements = self.rewrite_text_placeholders(&mut rewritten, "header")?;

        if replacements == 0 {
            return Ok(None);
        }

        Ok(Some(b64.encode(rewritten.as_bytes())))
    }
}

pub fn alias_start_for_marker(text: &str, marker_abs: usize) -> usize {
    let mut start = marker_abs;
    let bytes = text.as_bytes();
    while start > 0 && is_alias_token_char(bytes[start - 1]) {
        start -= 1;
    }
    start
}

pub fn canonical_token_at(text: &str, abs_start: usize) -> Option<(usize, &str)> {
    if !text[abs_start..].starts_with(PLACEHOLDER_PREFIX) {
        return None;
    }
    let key_start = abs_start + PLACEHOLDER_PREFIX.len();
    let key_end = text[key_start..]
        .bytes()
        .position(|b| !is_env_key_char(b))
        .map_or(text.len(), |p| key_start + p);
    (key_end > key_start).then_some((key_end, &text[abs_start..key_end]))
}

pub fn alias_token_at(text: &str, abs_start: usize) -> Option<(usize, &str)> {
    let suffix = &text[abs_start..];
    let marker_rel = suffix.find(PROVIDER_ALIAS_MARKER)?;
    if marker_rel == 0 {
        return None;
    }
    let key_start = abs_start + marker_rel + PROVIDER_ALIAS_MARKER.len();
    let key_end = text[key_start..]
        .bytes()
        .position(|b| !is_env_key_char(b))
        .map_or(text.len(), |p| key_start + p);
    if key_end == key_start {
        return None;
    }
    let before_ok = abs_start == 0 || !is_alias_token_char(text.as_bytes()[abs_start - 1]);
    let after_ok = key_end == text.len() || !is_alias_token_char(text.as_bytes()[key_end]);
    (before_ok && after_ok).then_some((key_end, &text[abs_start..key_end]))
}

fn alias_env_key(token: &str) -> Option<&str> {
    let marker_start = token.find(PROVIDER_ALIAS_MARKER)?;
    if marker_start == 0 {
        return None;
    }
    if !token[..marker_start].bytes().all(is_alias_token_char) {
        return None;
    }
    let key_start = marker_start + PROVIDER_ALIAS_MARKER.len();
    let key_end = token[key_start..]
        .bytes()
        .position(|b| !is_env_key_char(b))
        .map_or(token.len(), |p| key_start + p);
    (key_end == token.len() && key_end > key_start).then_some(&token[key_start..key_end])
}

fn token_boundary_ok(text: &str, abs_start: usize, token_end: usize, token: &str) -> bool {
    if token.starts_with(PLACEHOLDER_PREFIX) {
        return token_end == text.len()
            || !is_env_key_char(text.as_bytes()[token_end])
            || text[token_end..].starts_with(PLACEHOLDER_PREFIX);
    }
    let before_ok = abs_start == 0 || !is_alias_token_char(text.as_bytes()[abs_start - 1]);
    let after_ok = token_end == text.len() || !is_alias_token_char(text.as_bytes()[token_end]);
    before_ok && after_ok
}

pub fn placeholder_for_env_key(key: &str) -> String {
    format!("{PLACEHOLDER_PREFIX}{key}")
}

pub fn placeholder_for_env_key_for_revision(key: &str, revision: u64) -> String {
    if revision == 0 {
        placeholder_for_env_key(key)
    } else {
        format!("{PLACEHOLDER_PREFIX}v{revision}_{key}")
    }
}

// ---------------------------------------------------------------------------
// Secret validation (F1 — CWE-113)
// ---------------------------------------------------------------------------

/// Validate that a resolved secret value does not contain characters that
/// could enable HTTP header injection or request splitting.
fn validate_resolved_secret(value: &str) -> Result<&str, &'static str> {
    if value
        .bytes()
        .any(|b| b == b'\r' || b == b'\n' || b == b'\0')
    {
        return Err("resolved secret contains prohibited control characters (CR, LF, or NUL)");
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Percent decoding (RFC 3986)
// ---------------------------------------------------------------------------

/// Percent-decode a URL-encoded string.
pub fn percent_decode(input: &str) -> String {
    let mut decoded = Vec::with_capacity(input.len());
    let mut bytes = input.bytes();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let hi = bytes.next();
            let lo = bytes.next();
            if let (Some(h), Some(l)) = (hi, lo) {
                let hex = [h, l];
                if let Ok(s) = std::str::from_utf8(&hex)
                    && let Ok(val) = u8::from_str_radix(s, 16)
                {
                    decoded.push(val);
                    continue;
                }
                // Invalid percent encoding — preserve verbatim
                decoded.push(b'%');
                decoded.push(h);
                decoded.push(l);
            } else {
                decoded.push(b'%');
                if let Some(h) = hi {
                    decoded.push(h);
                }
            }
        } else {
            decoded.push(b);
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

#[cfg(test)]
#[allow(
    clippy::iter_on_single_items,
    reason = "Test code: single-key fixtures are clearer as array literals than std::iter::once."
)]
mod tests {
    use super::*;

    #[test]
    fn provider_env_is_replaced_with_placeholders() {
        let (child_env, resolver) = SecretResolver::from_provider_env(
            [("ANTHROPIC_API_KEY".to_string(), "sk-test".to_string())]
                .into_iter()
                .collect(),
        );

        assert_eq!(
            child_env.get("ANTHROPIC_API_KEY"),
            Some(&"openshell:resolve:env:ANTHROPIC_API_KEY".to_string())
        );
        assert_eq!(
            resolver
                .as_ref()
                .and_then(|resolver| resolver
                    .resolve_placeholder("openshell:resolve:env:ANTHROPIC_API_KEY")),
            Some("sk-test")
        );
    }

    #[test]
    fn empty_provider_env_produces_no_resolver() {
        let (child_env, resolver) = SecretResolver::from_provider_env(HashMap::new());
        assert!(child_env.is_empty());
        assert!(resolver.is_none());
    }

    #[test]
    fn resolver_rejects_secret_with_crlf() {
        let (_, resolver) = SecretResolver::from_provider_env(
            [("INJECTED".to_string(), "value\r\nX-Header: leak".to_string())]
                .into_iter()
                .collect(),
        );
        let resolver = resolver.expect("resolver");
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:INJECTED"),
            None,
            "CRLF in secret value must be rejected"
        );
    }

    #[test]
    fn placeholder_for_env_key_for_revision_distinguishes_revisions() {
        assert_eq!(
            placeholder_for_env_key_for_revision("KEY", 0),
            "openshell:resolve:env:KEY"
        );
        assert_eq!(
            placeholder_for_env_key_for_revision("KEY", 7),
            "openshell:resolve:env:v7_KEY"
        );
    }

    #[test]
    fn percent_decode_round_trips() {
        let encoded = "hello%20world%21";
        let decoded = percent_decode(encoded);
        assert_eq!(decoded, "hello world!");
    }

    #[test]
    fn contains_reserved_credential_marker_detects_percent_encoded() {
        let encoded = "openshell%3Aresolve%3Aenv%3AKEY";
        assert!(contains_reserved_credential_marker(encoded));
    }

    #[test]
    fn rewrite_websocket_text_replaces_placeholders_and_returns_count() {
        let (child_env, resolver) = SecretResolver::from_provider_env(
            [
                ("DISCORD_BOT_TOKEN".to_string(), "real-token".to_string()),
                ("APP_ID".to_string(), "app-123".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        let resolver = resolver.expect("resolver");
        let token = child_env.get("DISCORD_BOT_TOKEN").unwrap();
        let app_id = child_env.get("APP_ID").unwrap();
        let mut payload =
            format!(r#"{{"op":2,"d":{{"token":"{token}","properties":{{"app":"{app_id}"}}}}}}"#);

        let count = resolver
            .rewrite_websocket_text_placeholders(&mut payload)
            .expect("rewrite should succeed");

        assert_eq!(count, 2);
        assert!(payload.contains(r#""token":"real-token""#));
        assert!(payload.contains(r#""app":"app-123""#));
        assert!(!payload.contains(PLACEHOLDER_PREFIX));
    }

    #[test]
    fn rewrite_websocket_text_replaces_provider_shaped_alias() {
        let (_, resolver) = SecretResolver::from_provider_env(
            [("APP_TOKEN".to_string(), "app-real-token".to_string())]
                .into_iter()
                .collect(),
        );
        let resolver = resolver.expect("resolver");
        let mut payload = r#"{"token":"provider-OPENSHELL-RESOLVE-ENV-APP_TOKEN"}"#.to_string();

        let count = resolver
            .rewrite_websocket_text_placeholders(&mut payload)
            .expect("alias should rewrite");

        assert_eq!(count, 1);
        assert_eq!(payload, r#"{"token":"app-real-token"}"#);
    }

    #[test]
    fn rewrite_websocket_text_without_placeholder_is_unchanged() {
        let (_, resolver) = SecretResolver::from_provider_env(
            [("KEY".to_string(), "secret".to_string())]
                .into_iter()
                .collect(),
        );
        let resolver = resolver.expect("resolver");
        let mut payload = r#"{"op":1,"d":42}"#.to_string();

        let count = resolver
            .rewrite_websocket_text_placeholders(&mut payload)
            .expect("rewrite should succeed");

        assert_eq!(count, 0);
        assert_eq!(payload, r#"{"op":1,"d":42}"#);
    }

    #[test]
    fn rewrite_websocket_text_unknown_placeholder_fails_closed_without_mutating() {
        let (_, resolver) = SecretResolver::from_provider_env(
            [("KEY".to_string(), "secret".to_string())]
                .into_iter()
                .collect(),
        );
        let resolver = resolver.expect("resolver");
        let original = r#"{"token":"openshell:resolve:env:UNKNOWN"}"#.to_string();
        let mut payload = original.clone();

        let err = resolver
            .rewrite_websocket_text_placeholders(&mut payload)
            .expect_err("unknown placeholder should fail");

        assert_eq!(err.location, "websocket");
        assert_eq!(payload, original);
    }

    #[test]
    fn rewrite_websocket_text_handles_repeated_adjacent_and_unicode_placeholders() {
        let (child_env, resolver) = SecretResolver::from_provider_env(
            [
                ("TOKEN".to_string(), "tok".to_string()),
                ("APP".to_string(), "app".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        let resolver = resolver.expect("resolver");
        let token = child_env.get("TOKEN").unwrap();
        let app = child_env.get("APP").unwrap();
        let mut payload = format!("prefix-☃-{token}{app}-{token}-suffix");

        let count = resolver
            .rewrite_websocket_text_placeholders(&mut payload)
            .expect("rewrite should succeed");

        assert_eq!(count, 3);
        assert_eq!(payload, "prefix-☃-tokapp-tok-suffix");
    }

    #[test]
    fn rewrite_websocket_text_placeholder_like_prefix_fails_without_mutating() {
        let (_, resolver) = SecretResolver::from_provider_env(
            [("KEY".to_string(), "secret".to_string())]
                .into_iter()
                .collect(),
        );
        let resolver = resolver.expect("resolver");
        let original = "openshell:resolve:env:-not-a-key".to_string();
        let mut payload = original.clone();

        let err = resolver
            .rewrite_websocket_text_placeholders(&mut payload)
            .expect_err("placeholder-like prefix should fail closed");

        assert_eq!(err.location, "websocket");
        assert_eq!(payload, original);
    }

    #[test]
    fn debug_format_does_not_leak_secret_values() {
        let (_, resolver) = SecretResolver::from_provider_env(
            [
                (
                    "ANTHROPIC_API_KEY".to_string(),
                    "sk-very-secret-value-12345".to_string(),
                ),
                ("DB_PASSWORD".to_string(), "very-secret-value".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        let resolver = resolver.expect("resolver");

        let plain = format!("{resolver:?}");
        let pretty = format!("{resolver:#?}");

        for output in [&plain, &pretty] {
            assert!(
                !output.contains("sk-very-secret-value-12345"),
                "secret value leaked via Debug: {output}"
            );
            assert!(
                !output.contains("very-secret-value"),
                "secret value leaked via Debug: {output}"
            );
            assert!(
                !output.contains("ANTHROPIC_API_KEY"),
                "placeholder key (env var name) leaked via Debug: {output}"
            );
            assert!(
                !output.contains("DB_PASSWORD"),
                "placeholder key (env var name) leaked via Debug: {output}"
            );
            assert!(
                !output.contains(PLACEHOLDER_PREFIX),
                "placeholder prefix leaked via Debug: {output}"
            );
            assert!(
                output.contains("SecretResolver"),
                "Debug output should still identify the type: {output}"
            );
        }

        assert!(
            plain.contains('2'),
            "Debug output should expose the placeholder count: {plain}"
        );
    }

    #[test]
    fn debug_format_of_empty_resolver_is_safe() {
        let resolver = SecretResolver::default();
        let output = format!("{resolver:?}");
        assert!(output.contains("SecretResolver"));
        assert!(output.contains('0'));
        assert!(!output.contains(PLACEHOLDER_PREFIX));
    }
}
