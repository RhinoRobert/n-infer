//! The N-API surface: everything JavaScript can see.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

use crate::engine::Engine;
use crate::error::Result;
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

/// Loads a GGUF model on the libuv thread pool; the event loop stays free.
#[napi]
pub fn load_model(config: ModelConfig) -> AsyncTask<LoadTask> {
    AsyncTask::new(LoadTask { config })
}

#[napi]
pub struct NativeModel {
    engine: Arc<Engine>,
}

#[napi]
impl NativeModel {
    /// Resolves with the full completion once generation finishes.
    #[napi]
    pub fn generate(&self, prompt: String, options: Option<GenerateOptions>) -> AsyncTask<GenerateTask> {
        AsyncTask::new(GenerateTask {
            engine: self.engine.clone(),
            prompt,
            options: options.unwrap_or_default(),
        })
    }

    /// Calls `callback(err, chunk)` per text piece, then `callback(null, null)` when finished.
    #[napi]
    pub fn stream(
        &self,
        prompt: String,
        options: Option<GenerateOptions>,
        callback: ThreadsafeFunction<Option<String>, ErrorStrategy::CalleeHandled>,
    ) -> CancelHandle {
        let engine = self.engine.clone();
        let options = options.unwrap_or_default();
        let flag = Arc::new(AtomicBool::new(false));
        let cancel = flag.clone();
        let mode = ThreadsafeFunctionCallMode::NonBlocking;

        std::thread::spawn(move || {
            let result = engine.generate(&prompt, &options, &cancel, |chunk| {
                callback.call(Ok(Some(chunk)), mode);
            });
            callback.call(result.map(|()| None), mode);
        });

        CancelHandle { flag }
    }
}

#[napi]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
}

#[napi]
impl CancelHandle {
    #[napi]
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
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
        Ok(NativeModel { engine: Arc::new(engine) })
    }
}

pub struct GenerateTask {
    engine: Arc<Engine>,
    prompt: String,
    options: GenerateOptions,
}

impl Task for GenerateTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<String> {
        let mut text = String::new();
        let never_cancelled = AtomicBool::new(false);
        self.engine
            .generate(&self.prompt, &self.options, &never_cancelled, |chunk| text.push_str(&chunk))?;
        Ok(text)
    }

    fn resolve(&mut self, _env: Env, text: String) -> Result<String> {
        Ok(text)
    }
}
