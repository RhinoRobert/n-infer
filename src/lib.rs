#![deny(clippy::all)]

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

const DEFAULT_CTX: u32 = 4096;
const DEFAULT_MAX_TOKENS: u32 = 256;
const PROMPT_BATCH: usize = 512;

fn err<E: std::fmt::Display>(e: E) -> Error {
    Error::from_reason(e.to_string())
}

// llama.cpp's backend is process-global and may only be initialised once.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static BACKEND_INIT: Mutex<()> = Mutex::new(());

fn backend(verbose: bool) -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _guard = BACKEND_INIT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(verbose));
    let b = LlamaBackend::init().map_err(err)?;
    Ok(BACKEND.get_or_init(|| b))
}

#[napi(object)]
pub struct ModelConfig {
    pub model_path: String,
    /// Layers to offload to the GPU. Ignored by CPU-only builds.
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
    pub threads: Option<i32>,
    pub verbose: Option<bool>,
}

#[napi(object)]
pub struct GenerateOptions {
    pub max_tokens: Option<u32>,
    /// 0 (or less) selects greedy decoding.
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u32>,
}

struct Engine {
    model: LlamaModel,
    n_ctx: u32,
    threads: Option<i32>,
    verbose: bool,
}

/// Decodes `prompt` and samples tokens, handing each decoded text piece to `emit`.
/// A fresh context is created per call so concurrent generations never share KV state.
fn run(
    engine: &Engine,
    prompt: &str,
    opts: &GenerateOptions,
    cancel: &AtomicBool,
    mut emit: impl FnMut(String),
) -> Result<()> {
    let backend = backend(engine.verbose)?;

    let mut cparams = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(engine.n_ctx));
    if let Some(t) = engine.threads {
        cparams = cparams.with_n_threads(t).with_n_threads_batch(t);
    }
    let mut ctx = engine.model.new_context(backend, cparams).map_err(err)?;

    let tokens = engine.model.vocab().tokenize(prompt.as_bytes(), true, true);
    if tokens.is_empty() {
        return Err(err("prompt produced no tokens"));
    }
    let n_ctx = engine.n_ctx as usize;
    if tokens.len() >= n_ctx {
        return Err(err(format!(
            "prompt is {} tokens but the context window is {}",
            tokens.len(),
            n_ctx
        )));
    }
    let max_new = (opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS) as usize).min(n_ctx - tokens.len());

    // Feed the prompt in chunks so prompts longer than one batch still work.
    let mut batch = LlamaBatch::new(PROMPT_BATCH, 1);
    let last = tokens.len() - 1;
    let mut pos: i32 = 0;
    for chunk in tokens.chunks(PROMPT_BATCH) {
        batch.clear();
        for &tok in chunk {
            batch.add(tok, pos, &[0], pos as usize == last).map_err(err)?;
            pos += 1;
        }
        ctx.decode(&mut batch).map_err(err)?;
    }

    let temperature = opts.temperature.unwrap_or(0.8) as f32;
    let mut sampler = if temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        let seed = opts.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(1234)
        });
        LlamaSampler::chain_simple([
            LlamaSampler::top_p(opts.top_p.unwrap_or(0.95) as f32, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(seed),
        ])
    };

    // Streaming decoder so multi-byte UTF-8 characters split across tokens stay intact.
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut logits_idx = batch.n_tokens() - 1;

    for _ in 0..max_new {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let token = sampler.sample(&ctx, logits_idx);
        sampler.accept(token);
        if engine.model.vocab().is_eog(token) {
            break;
        }

        let bytes = engine.model.vocab().token_to_piece(token, true, None);
        let mut text = String::with_capacity(decoder.max_utf8_buffer_length(bytes.len()).unwrap_or(16));
        let _ = decoder.decode_to_string(&bytes, &mut text, false);
        if !text.is_empty() {
            emit(text);
        }

        batch.clear();
        batch.add(token, pos, &[0], true).map_err(err)?;
        pos += 1;
        logits_idx = 0;
        ctx.decode(&mut batch).map_err(err)?;
    }
    Ok(())
}

pub struct LoadTask {
    config: ModelConfig,
}

impl Task for LoadTask {
    type Output = Engine;
    type JsValue = NativeModel;

    fn compute(&mut self) -> Result<Engine> {
        let verbose = self.config.verbose.unwrap_or(false);
        let backend = backend(verbose)?;
        let params = LlamaModelParams::default().with_n_gpu_layers(self.config.gpu_layers.unwrap_or(0));
        let model = LlamaModel::load_from_file(backend, &self.config.model_path, &params)
            .map_err(|e| err(format!("failed to load '{}': {e}", self.config.model_path)))?;
        Ok(Engine {
            model,
            n_ctx: self.config.context_size.unwrap_or(DEFAULT_CTX),
            threads: self.config.threads,
            verbose,
        })
    }

    fn resolve(&mut self, _env: Env, engine: Engine) -> Result<NativeModel> {
        Ok(NativeModel {
            engine: Arc::new(engine),
        })
    }
}

pub struct GenerateTask {
    engine: Arc<Engine>,
    prompt: String,
    opts: GenerateOptions,
}

impl Task for GenerateTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<String> {
        let mut out = String::new();
        let never = AtomicBool::new(false);
        run(&self.engine, &self.prompt, &self.opts, &never, |s| out.push_str(&s))?;
        Ok(out)
    }

    fn resolve(&mut self, _env: Env, out: String) -> Result<String> {
        Ok(out)
    }
}

/// Loads a GGUF model on the libuv thread pool; the event loop stays free.
#[napi]
pub fn load_model(config: ModelConfig) -> AsyncTask<LoadTask> {
    AsyncTask::new(LoadTask { config })
}

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
            opts: options.unwrap_or(GenerateOptions {
                max_tokens: None,
                temperature: None,
                top_p: None,
                seed: None,
            }),
        })
    }

    /// Calls `callback(err, chunk)` for each text piece, then `callback(null, null)` when done.
    #[napi]
    pub fn stream(
        &self,
        prompt: String,
        options: Option<GenerateOptions>,
        callback: ThreadsafeFunction<Option<String>, ErrorStrategy::CalleeHandled>,
    ) -> CancelHandle {
        let engine = self.engine.clone();
        let flag = Arc::new(AtomicBool::new(false));
        let cancel = flag.clone();
        let opts = options.unwrap_or(GenerateOptions {
            max_tokens: None,
            temperature: None,
            top_p: None,
            seed: None,
        });

        std::thread::spawn(move || {
            let result = run(&engine, &prompt, &opts, &cancel, |chunk| {
                callback.call(Ok(Some(chunk)), ThreadsafeFunctionCallMode::NonBlocking);
            });
            match result {
                Ok(()) => callback.call(Ok(None), ThreadsafeFunctionCallMode::NonBlocking),
                Err(e) => callback.call(Err(e), ThreadsafeFunctionCallMode::NonBlocking),
            };
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
