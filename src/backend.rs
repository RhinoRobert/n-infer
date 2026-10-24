//! Process-global llama.cpp backend. It may only be initialised once.

use std::sync::{Mutex, OnceLock};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use crate::error::{err, Result};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

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
    let backend = LlamaBackend::init().map_err(err)?;
    Ok(BACKEND.get_or_init(|| backend))
}
