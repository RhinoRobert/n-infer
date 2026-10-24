//! Process-global llama.cpp backend. It may only be initialised once.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use crate::error::{err, Result};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());
/// Where the compute-backend libraries (`ggml-cpu-*`, `ggml-cuda`) sit next to the addon.
static BACKEND_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Records the folder holding the addon's companion libraries. Only used by builds made with
/// the `dynamic` feature; harmless otherwise. Must be called before the first model is loaded.
pub fn set_backend_dir(dir: PathBuf) {
    let _ = BACKEND_DIR.set(dir);
}

pub fn get(verbose: bool) -> Result<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let _guard = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    crate::hardware::ensure_cpu_supported()?;
    // Left untouched, llama.cpp logs to stderr; otherwise route (and drop) them via tracing.
    if !verbose {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    }
    load_dynamic_backends();
    let backend = LlamaBackend::init().map_err(err)?;
    Ok(BACKEND.get_or_init(|| backend))
}

/// With the `dynamic` feature, ggml discovers the best CPU variant (and any GPU backend) at
/// runtime from separate libraries; point it at the folder they were installed to.
#[cfg(feature = "dynamic")]
fn load_dynamic_backends() {
    if let Some(dir) = BACKEND_DIR.get() {
        llama_cpp_2::llama_backend::load_backends_from_path(dir);
    }
}

#[cfg(not(feature = "dynamic"))]
fn load_dynamic_backends() {}
