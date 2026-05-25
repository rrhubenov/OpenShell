// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry of PIDs the supervisor has spawned and is responsible for reaping.
//!
//! The reaper consults [`is_managed_child`] to distinguish supervisor-spawned
//! processes from unrelated zombies inherited via PID-1 reparenting.

#[cfg(target_os = "linux")]
use std::collections::HashSet;
#[cfg(target_os = "linux")]
use std::sync::{LazyLock, Mutex};

#[cfg(target_os = "linux")]
static MANAGED_CHILDREN: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Record `pid` as a supervisor-managed child eligible for reaping.
#[cfg(target_os = "linux")]
pub fn register_managed_child(pid: u32) {
    let Ok(pid) = i32::try_from(pid) else {
        return;
    };
    if pid <= 0 {
        return;
    }
    if let Ok(mut children) = MANAGED_CHILDREN.lock() {
        children.insert(pid);
    }
}

/// Stop tracking `pid` once it has exited or been adopted elsewhere.
#[cfg(target_os = "linux")]
pub fn unregister_managed_child(pid: u32) {
    let Ok(pid) = i32::try_from(pid) else {
        return;
    };
    if pid <= 0 {
        return;
    }
    if let Ok(mut children) = MANAGED_CHILDREN.lock() {
        children.remove(&pid);
    }
}

#[cfg(target_os = "linux")]
pub fn is_managed_child(pid: i32) -> bool {
    MANAGED_CHILDREN
        .lock()
        .is_ok_and(|children| children.contains(&pid))
}

#[cfg(not(target_os = "linux"))]
pub fn register_managed_child(_pid: u32) {}

#[cfg(not(target_os = "linux"))]
pub fn unregister_managed_child(_pid: u32) {}
