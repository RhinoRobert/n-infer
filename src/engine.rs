//! Model loading, GGUF header verification, persistent KV-cache session management,
//! and the token generation loop. No N-API types in here.

use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::backend;
use crate::error::{err, Result};
use crate::options::{GenerateOptions, ModelConfig, ValidatedGenerateOptions, ValidatedModelConfig};
use crate::tuning::Tuning;

/// VRAM left untouched per device when fitting a model to the GPU.
const FIT_MARGIN_BYTES: usize = 1024 * 1024 * 1024;
/// Context size the fitter may shrink to before it starts moving layers to the CPU.
const FIT_MIN_CONTEXT: u32 = 2048;
/// llama.cpp's fitter mutates global logger state, so only one may run at a time.
static FIT_LOCK: Mutex<()> = Mutex::new(());
/// How many recent tokens the repetition penalty looks back over.
const REPEAT_WINDOW: i32 = 64;
/// Maximum UTF-8 byte length accepted for a single prompt (4 MiB).
const MAX_PROMPT_BYTES: usize = 4 * 1024 * 1024;
/// Maximum concurrent generations permitted on a single model instance.
const MAX_CONCURRENT_GENERATIONS: usize = 16;
/// Minimum valid GGUF file size in bytes.
const MIN_GGUF_BYTES: u64 = 64;

struct Session {
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
    /// Exact sequence of tokens currently resident in `ctx`'s KV cache (seq 0).
    cached_tokens: Vec<LlamaToken>,
    /// Prompt tokens submitted per `decode` call (the batch capacity).
    chunk: usize,
}

pub struct Engine {
    /// Primary reusable session (KV cache + batch buffer).
    /// Declared BEFORE `model` so it is always dropped before `LlamaModel`.
    cached_session: Mutex<Option<Session>>,
    /// Serializes `LlamaModel::new_context` calls (upstream llama.cpp mutates model state
    /// during context creation and is not safe to call concurrently on the same model).
    context_init_lock: Mutex<()>,
    active_generations: AtomicUsize,
    closed: AtomicBool,
    model: Box<LlamaModel>,
    context_size: u32,
    tuning: Tuning,
    verbose: bool,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Explicitly drop the cached LlamaContext before LlamaModel is freed.
        if let Ok(mut slot) = self.cached_session.lock() {
            let _ = slot.take();
        }
    }
}

impl Engine {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let cfg = config.validate()?;
        validate_gguf_file(&cfg.model_path)?;

        let backend = backend::get(cfg.verbose)?;
        let tuning = Tuning::resolve(&cfg);
        let params = model_params(&cfg, &tuning);
        let model = LlamaModel::load_from_file(backend, &cfg.model_path, &params)
            .map_err(|e| err(format!("failed to load '{}': {e}", cfg.model_path)))?;

        let engine = Self {
            cached_session: Mutex::new(None),
            context_init_lock: Mutex::new(()),
            active_generations: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            model: Box::new(model),
            context_size: cfg.context_size,
            tuning,
            verbose: cfg.verbose,
        };

        // Pre-allocate the primary context during init() so KV memory is ready before the first call.
        let warm_session = engine.new_session()?;
        *engine
            .cached_session
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(warm_session);

        Ok(engine)
    }

    /// Marks the engine as closed, signals any in-flight generation to stop,
    /// and immediately releases the cached KV context.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        if let Ok(mut slot) = self.cached_session.lock() {
            let _ = slot.take();
        }
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Decodes `prompt`, then samples until end-of-generation, `max_tokens` or cancellation,
    /// passing each decoded text piece to `emit`.
    ///
    /// Sequential calls reuse the warm `LlamaContext` and automatically reuse the longest
    /// matching prefix already present in the KV cache.
    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
        cancel: &AtomicBool,
        mut emit: impl FnMut(String),
    ) -> Result<()> {
        if self.is_closed() {
            return Err(err("LocalModel has been closed"));
        }
        let opts = options.validate()?;
        if prompt.is_empty() {
            return Err(err("prompt must not be empty"));
        }
        if prompt.len() > MAX_PROMPT_BYTES {
            return Err(err(format!(
                "prompt exceeds maximum byte length ({} > {MAX_PROMPT_BYTES})",
                prompt.len()
            )));
        }

        let _active_guard = ActiveGuard::acquire(&self.active_generations)?;
        if opts.max_tokens == 0 {
            return Ok(());
        }

        let tokens = self.tokenize(prompt)?;
        let max_tokens =
            (opts.max_tokens as usize).min(self.context_size as usize - tokens.len());

        let mut session_guard = self.acquire_session()?;
        let session = session_guard.session_mut();

        let Some(mut pos) = eval_prompt(session, &tokens, cancel, &self.closed)? else {
            if self.closed.load(Ordering::Relaxed) {
                return Err(err("LocalModel was closed during generation"));
            }
            return Ok(());
        };

        let mut sampler = build_sampler(&opts, self.model.n_vocab());
        let penalty_tail = &tokens[tokens.len().saturating_sub(REPEAT_WINDOW as usize)..];
        sampler.accept_many(penalty_tail);

        let mut utf8 = encoding_rs::UTF_8.new_decoder();

        for step in 0..max_tokens {
            if cancel.load(Ordering::Relaxed) || self.closed.load(Ordering::Relaxed) {
                break;
            }
            // Note: `sampler.sample` calls `llama_sampler_accept` internally in llama.cpp.
            let token = sampler.sample(&session.ctx, -1);
            if self.model.vocab().is_eog(token) {
                break;
            }

            let text = self.decode_piece(&mut utf8, token);
            if !text.is_empty() {
                emit(text);
            }

            // Skip decoding the final token when we have reached `max_tokens`.
            if step + 1 < max_tokens
                && !cancel.load(Ordering::Relaxed)
                && !self.closed.load(Ordering::Relaxed)
            {
                session.batch.clear();
                session.batch.add(token, pos, &[0], true).map_err(err)?;
                pos += 1;
                if let Err(e) = session.ctx.decode(&mut session.batch) {
                    session.ctx.clear_kv_cache();
                    session.cached_tokens.clear();
                    return Err(err(e));
                }
                session.cached_tokens.push(token);
            }
        }
        if self.closed.load(Ordering::Relaxed) {
            return Err(err("LocalModel was closed during generation"));
        }
        Ok(())
    }

    fn acquire_session(&self) -> Result<SessionGuard<'_>> {
        let taken = self
            .cached_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let session = match taken {
            Some(s) => s,
            None => self.new_session()?,
        };
        Ok(SessionGuard {
            engine: self,
            session: Some(session),
        })
    }

    fn new_session(&self) -> Result<Session> {
        let _lock = self
            .context_init_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let params = self.tuning.context_params(self.context_size);
        let backend = backend::get(self.verbose)?;
        // SAFETY: `self.model` is heap-allocated (`Box<LlamaModel>`) and owned by `Engine`.
        // Every `Session` is either owned by a `SessionGuard<'_>` borrowing `&self` or stored
        // in `self.cached_session`, which is explicitly dropped before `self.model` in `Drop`.
        let model_ref: &'static LlamaModel = unsafe { &*(&*self.model as *const LlamaModel) };
        let ctx = model_ref.new_context(backend, params).map_err(err)?;
        Ok(Session {
            ctx,
            batch: LlamaBatch::new(self.tuning.n_batch as usize, 1),
            cached_tokens: Vec::with_capacity(self.context_size as usize),
            chunk: self.tuning.n_batch as usize,
        })
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<LlamaToken>> {
        let tokens = self.model.vocab().tokenize(prompt.as_bytes(), true, true);
        if tokens.is_empty() {
            return Err(err("prompt produced no tokens"));
        }
        if tokens.len() >= self.context_size as usize {
            return Err(err(format!(
                "prompt is {} tokens but the context window is {}",
                tokens.len(),
                self.context_size
            )));
        }
        Ok(tokens)
    }

    /// Streaming UTF-8 decode so characters split across tokens stay intact.
    fn decode_piece(&self, utf8: &mut encoding_rs::Decoder, token: LlamaToken) -> String {
        let bytes = self.model.vocab().token_to_piece(token, false, None);
        let mut text =
            String::with_capacity(utf8.max_utf8_buffer_length(bytes.len()).unwrap_or(16));
        let _ = utf8.decode_to_string(&bytes, &mut text, false);
        text
    }
}

/// Model params for the requested offload. With no explicit layer count (the default on GPU
/// builds) llama.cpp's fitter splits the model between VRAM and RAM for this machine.
fn model_params(cfg: &ValidatedModelConfig, tuning: &Tuning) -> Pin<Box<LlamaModelParams>> {
    match cfg.gpu_layers {
        Some(layers) => Box::pin(LlamaModelParams::default().with_n_gpu_layers(layers)),
        None => fit_to_vram(cfg, tuning)
            .unwrap_or_else(|| Box::pin(LlamaModelParams::default().with_n_gpu_layers(0))),
    }
}

fn fit_to_vram(cfg: &ValidatedModelConfig, tuning: &Tuning) -> Option<Pin<Box<LlamaModelParams>>> {
    let path = CString::new(cfg.model_path.as_str()).ok()?;
    let mut params = Box::pin(LlamaModelParams::default());
    let mut cparams = tuning.context_params(cfg.context_size);
    let mut margins = vec![FIT_MARGIN_BYTES; llama_cpp_2::max_devices().max(1)];
    let log_level = if cfg.verbose { 2 } else { 4 };

    let _guard = FIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let fitted = params.as_mut().fit_params(
        &path,
        &mut cparams,
        &mut margins,
        FIT_MIN_CONTEXT.min(cfg.context_size),
        log_level,
    );
    match fitted {
        Ok(_) => Some(params),
        Err(e) => {
            if cfg.verbose {
                eprintln!("n-infer: could not fit the model to VRAM ({e}); using the CPU");
            }
            None
        }
    }
}

/// Evaluates `tokens` into `session.ctx`, reusing any matching prefix already present
/// in `session.cached_tokens`. Returns `Ok(None)` if cancelled mid-prefill.
fn eval_prompt(
    session: &mut Session,
    tokens: &[LlamaToken],
    cancel: &AtomicBool,
    closed: &AtomicBool,
) -> Result<Option<i32>> {
    // Keep at most `tokens.len() - 1` cached tokens so the final prompt token is always
    // decoded to produce fresh logits for sampling.
    let mut keep = session
        .cached_tokens
        .iter()
        .zip(tokens.iter())
        .take_while(|(a, b)| a == b)
        .count()
        .min(tokens.len() - 1);

    if keep == 0 {
        if !session.cached_tokens.is_empty() {
            session.ctx.clear_kv_cache();
            session.cached_tokens.clear();
        }
    } else if session.cached_tokens.len() > keep {
        if session
            .ctx
            .clear_kv_cache_seq(Some(0), Some(keep as u32), None)
            .unwrap_or(false)
        {
            session.cached_tokens.truncate(keep);
        } else {
            // Recurrent/hybrid architectures do not support partial KV truncation; full reset.
            session.ctx.clear_kv_cache();
            session.cached_tokens.clear();
            keep = 0;
        }
    }

    let last = tokens.len() - 1;
    let mut pos = keep as i32;
    for chunk in tokens[keep..].chunks(session.chunk) {
        if cancel.load(Ordering::Relaxed) || closed.load(Ordering::Relaxed) {
            return Ok(None);
        }
        session.batch.clear();
        for &token in chunk {
            session
                .batch
                .add(token, pos, &[0], pos as usize == last)
                .map_err(err)?;
            pos += 1;
        }
        if let Err(e) = session.ctx.decode(&mut session.batch) {
            session.ctx.clear_kv_cache();
            session.cached_tokens.clear();
            return Err(err(e));
        }
        session.cached_tokens.extend_from_slice(chunk);
    }
    Ok(Some(pos))
}

fn build_sampler(options: &ValidatedGenerateOptions, n_vocab: i32) -> LlamaSampler {
    let mut chain = Vec::new();
    if (options.repetition_penalty - 1.0).abs() > f32::EPSILON {
        chain.push(LlamaSampler::penalties(
            n_vocab,
            REPEAT_WINDOW,
            options.repetition_penalty,
            0.0,
            0.0,
        ));
    }
    if options.temperature <= 0.0 {
        chain.push(LlamaSampler::greedy());
    } else {
        let seed = options.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(1234)
        });
        if options.top_k > 0 {
            chain.push(LlamaSampler::top_k(options.top_k));
        }
        chain.push(LlamaSampler::top_p(options.top_p, 1));
        chain.push(LlamaSampler::temp(options.temperature));
        chain.push(LlamaSampler::dist(seed));
    }
    LlamaSampler::chain_simple(chain)
}

/// Verifies `path` points to a regular file with a valid GGUF magic header and version (1..=3)
/// before handing the path to llama.cpp's C/C++ parser.
fn validate_gguf_file(path: &str) -> Result<()> {
    let mut file =
        File::open(path).map_err(|e| err(format!("cannot open model '{}': {e}", path)))?;
    let meta = file
        .metadata()
        .map_err(|e| err(format!("cannot stat model '{}': {e}", path)))?;
    if !meta.is_file() {
        return Err(err(format!("model path '{}' is not a regular file", path)));
    }
    if meta.len() < MIN_GGUF_BYTES {
        return Err(err(format!(
            "model file '{}' is too small ({} bytes) to be a valid GGUF file",
            path,
            meta.len()
        )));
    }
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|e| err(format!("cannot read header of '{}': {e}", path)))?;
    if &header[0..4] != b"GGUF" {
        return Err(err(format!(
            "invalid model file '{}': missing GGUF magic header",
            path
        )));
    }
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if !(1..=3).contains(&version) {
        return Err(err(format!(
            "unsupported GGUF format version ({version}) in '{}'",
            path
        )));
    }
    Ok(())
}

struct ActiveGuard<'a>(&'a AtomicUsize);

impl<'a> ActiveGuard<'a> {
    fn acquire(counter: &'a AtomicUsize) -> Result<Self> {
        let prev = counter.fetch_add(1, Ordering::AcqRel);
        if prev >= MAX_CONCURRENT_GENERATIONS {
            counter.fetch_sub(1, Ordering::AcqRel);
            return Err(err(format!(
                "too many concurrent generations (max {MAX_CONCURRENT_GENERATIONS})"
            )));
        }
        Ok(Self(counter))
    }
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct SessionGuard<'a> {
    engine: &'a Engine,
    session: Option<Session>,
}

impl SessionGuard<'_> {
    fn session_mut(&mut self) -> &mut Session {
        self.session.as_mut().expect("session present")
    }
}

impl Drop for SessionGuard<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            if !self.engine.is_closed() {
                if let Ok(mut slot) = self.engine.cached_session.lock() {
                    if slot.is_none() {
                        *slot = Some(session);
                    }
                }
            }
        }
    }
}
