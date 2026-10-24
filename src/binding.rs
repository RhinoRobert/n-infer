//! The N-API surface: everything JavaScript can see. Kept thin: check arguments, call the engine.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

use crate::engine::{Engine, Event, Request};
use crate::error::err;
use crate::hardware::{self, HardwareInfo};
use crate::options::{
    validate_messages, validate_segments, ChatMessage, GenerateOptions, ModelConfig, PromptSegment,
};

/// Called with `(error, chunk)` for each piece of text, then `(null, null)` when finished.
type Callback = ThreadsafeFunction<Option<String>, ErrorStrategy::CalleeHandled>;

/// The crate version, so the JavaScript loader can check it is paired with the right binary.
#[napi]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Which accelerator this binary was compiled with: "cuda", "vulkan", "metal" or "cpu".
#[napi]
pub fn build_backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cpu"
    }
}

/// CPU cores and instruction sets, plus every compute device with its free memory.
#[napi]
pub fn hardware_info() -> Result<HardwareInfo> {
    Ok(hardware::info()?)
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
    /// Starts a generation. Returns at once; text arrives through `callback`.
    #[napi]
    pub fn run(
        &self,
        prompt: Vec<PromptSegment>,
        options: Option<GenerateOptions>,
        callback: Callback,
    ) -> Result<CancelHandle> {
        let engine = self.engine()?;
        validate_segments(&prompt)?;
        let options = options.unwrap_or_default().validate()?;

        let cancel = Arc::new(AtomicBool::new(false));
        let mode = ThreadsafeFunctionCallMode::NonBlocking;
        let request = Request { prompt, options, cancel: Arc::clone(&cancel) };
        engine.submit(request, move |event| {
            match event {
                Event::Text(text) => callback.call(Ok(Some(text)), mode),
                Event::Finished => callback.call(Ok(None), mode),
                Event::Failed(error) => callback.call(Err(error.into()), mode),
            };
        })?;
        Ok(CancelHandle { flag: cancel })
    }

    /// Applies the model's chat template and returns the prompt split into trusted template
    /// text and untrusted message text.
    #[napi]
    pub fn render_chat(&self, messages: Vec<ChatMessage>) -> Result<Vec<PromptSegment>> {
        validate_messages(&messages)?;
        Ok(self.engine()?.render_chat(&messages)?)
    }

    /// Token ids `prompt` becomes, exactly as `run` would tokenize it.
    #[napi]
    pub fn tokenize(&self, prompt: Vec<PromptSegment>) -> Result<Vec<i32>> {
        validate_segments(&prompt)?;
        Ok(self.engine()?.tokenize(&prompt)?)
    }

    /// Stops all work and frees the model and its contexts before returning.
    #[napi]
    pub fn close(&mut self) {
        if let Some(engine) = self.engine.take() {
            engine.close();
        }
    }

    fn engine(&self) -> Result<Arc<Engine>> {
        self.engine.clone().ok_or_else(|| err("LocalModel has been closed").into())
    }
}

/// Stops one running (or queued) generation.
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
        Ok(Engine::load(&self.config)?)
    }

    fn resolve(&mut self, _env: Env, engine: Engine) -> Result<NativeModel> {
        Ok(NativeModel { engine: Some(Arc::new(engine)) })
    }
}
