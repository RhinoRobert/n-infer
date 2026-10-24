//! The N-API surface: everything JavaScript can see.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

use crate::engine::Engine;
use crate::error::{err, Result};
use crate::hardware::{self, HardwareInfo};
use crate::options::{GenerateOptions, ModelConfig};

/// Which accelerator this binary was compiled with: "cuda", "vulkan" or "cpu".
#[napi]
pub fn build_backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else {
        "cpu"
    }
}

/// CPU cores and instruction sets, plus every compute device with its free memory.
#[napi]
pub fn hardware_info() -> Result<HardwareInfo> {
    hardware::info()
}

/// Loads a GGUF model on the libuv thread pool; the event loop stays free.
#[napi]
pub fn load_model(config: ModelConfig) -> AsyncTask<LoadTask> {
    AsyncTask::new(LoadTask { config })
}

#[napi]
pub struct NativeModel {
    engine: Option<Arc<Engine>>,
}

#[napi]
impl NativeModel {
    /// Resolves with the full completion once generation finishes.
    #[napi]
    pub fn generate(
        &self,
        prompt: String,
        options: Option<GenerateOptions>,
        cancel: Option<&CancelHandle>,
    ) -> Result<AsyncTask<GenerateTask>> {
        let engine = self.engine()?;
        let cancel_flag = cancel
            .map(|h| h.flag.clone())
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        Ok(AsyncTask::new(GenerateTask {
            engine,
            prompt,
            options: options.unwrap_or_default(),
            cancel: cancel_flag,
        }))
    }

    /// Calls `callback(err, chunk)` per text piece, then `callback(null, null)` when finished.
    #[napi]
    pub fn stream(
        &self,
        prompt: String,
        options: Option<GenerateOptions>,
        callback: ThreadsafeFunction<Option<String>, ErrorStrategy::CalleeHandled>,
    ) -> Result<CancelHandle> {
        let engine = self.engine()?;
        let options = options.unwrap_or_default();
        let flag = Arc::new(AtomicBool::new(false));
        let cancel = flag.clone();
        let mode = ThreadsafeFunctionCallMode::NonBlocking;

        std::thread::spawn(move || {
            let result = engine.generate(&prompt, &options, &cancel, |chunk| {
                if !cancel.load(Ordering::Relaxed) {
                    callback.call(Ok(Some(chunk)), mode);
                }
            });
            callback.call(result.map(|()| None), mode);
        });

        Ok(CancelHandle { flag })
    }

    /// Deterministically releases native model weights and KV-cache VRAM immediately.
    #[napi]
    pub fn close(&mut self) {
        if let Some(engine) = self.engine.take() {
            engine.close();
        }
    }

    fn engine(&self) -> Result<Arc<Engine>> {
        self.engine
            .as_ref()
            .filter(|e| !e.is_closed())
            .cloned()
            .ok_or_else(|| err("LocalModel has been closed"))
    }
}

#[napi]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
}

#[napi]
impl CancelHandle {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    #[napi]
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
}

impl Default for CancelHandle {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LoadTask {
    config: ModelConfig,
}

impl Task for LoadTask {
    type Output = Engine;
    type JsValue = NativeModel;

    fn compute(&mut self) -> Result<Engine> {
        Engine::load(&self.config)
    }

    fn resolve(&mut self, _env: Env, engine: Engine) -> Result<NativeModel> {
        Ok(NativeModel {
            engine: Some(Arc::new(engine)),
        })
    }
}

pub struct GenerateTask {
    engine: Arc<Engine>,
    prompt: String,
    options: GenerateOptions,
    cancel: Arc<AtomicBool>,
}

impl Task for GenerateTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<String> {
        let mut text = String::new();
        self.engine
            .generate(&self.prompt, &self.options, &self.cancel, |chunk| {
                text.push_str(&chunk)
            })?;
        if self.cancel.load(Ordering::Relaxed) {
            return Err(err("generation aborted"));
        }
        Ok(text)
    }

    fn resolve(&mut self, _env: Env, text: String) -> Result<String> {
        Ok(text)
    }
}
