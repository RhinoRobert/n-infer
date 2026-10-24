//! The inference engine: owns a loaded model and a bounded pool of workers that run generations.
//!
//! ```text
//! Engine ── Workers (bounded queue) ── worker thread ── Session (KV cache) ── llama.cpp
//!    └── Shared (model, tuning, closed flag)
//! ```
//!
//! Nothing here touches the N-API runtime; `binding.rs` is the only file that does.

mod chat;
mod generate;
mod loader;
mod prompt;
mod sampling;
mod session;
mod stop;
mod workers;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use llama_cpp_2::model::LlamaModel;

use self::session::Session;
use self::workers::Workers;
use crate::backend;
use crate::error::{err, guard, Error, Result};
use crate::options::{ChatMessage, ModelConfig, PromptSegment, ValidatedGenerateOptions};
use crate::tuning::Tuning;

/// State shared by the engine and its worker threads.
pub struct Shared {
    pub model: Arc<LlamaModel>,
    pub tuning: Tuning,
    pub context_size: u32,
    pub verbose: bool,
    pub closed: AtomicBool,
}

/// What a generation reports back, in order. Exactly one of `Finished` / `Failed` comes last.
pub enum Event {
    Text(String),
    Finished,
    Failed(Error),
}

/// One generation to run.
pub struct Request {
    pub prompt: Vec<PromptSegment>,
    pub options: ValidatedGenerateOptions,
    /// Set to stop this generation early.
    pub cancel: Arc<AtomicBool>,
}

pub struct Engine {
    shared: Arc<Shared>,
    workers: Workers,
}

impl Engine {
    /// Loads the model and allocates one warm context per worker, so allocation failures show up
    /// here rather than on the first request.
    pub fn load(config: &ModelConfig) -> Result<Self> {
        guard(|| Self::load_checked(config))
    }

    fn load_checked(config: &ModelConfig) -> Result<Self> {
        let cfg = config.validate()?;
        loader::validate_gguf_file(&cfg.model_path)?;
        if let Some(expected) = &cfg.model_sha256 {
            loader::verify_sha256(&cfg.model_path, expected)?;
        }

        let backend = backend::get(cfg.verbose)?;
        let tuning = Tuning::resolve(&cfg);
        let params = loader::model_params(&cfg, &tuning);
        let model = LlamaModel::load_from_file(backend, &cfg.model_path, &params)
            .map_err(|e| err(format!("failed to load '{}': {e}", cfg.model_path)))?;

        let shared = Arc::new(Shared {
            model: Arc::new(model),
            tuning,
            context_size: cfg.context_size,
            verbose: cfg.verbose,
            closed: AtomicBool::new(false),
        });
        let sessions =
            (0..cfg.max_concurrent).map(|_| Session::new(&shared)).collect::<Result<Vec<_>>>()?;
        let workers = Workers::start(&shared, sessions, cfg.max_queue as usize)?;
        Ok(Self { shared, workers })
    }

    /// Queues `request`. Events reach `sink` from a worker thread.
    pub fn submit(&self, request: Request, sink: impl FnMut(Event) + Send + 'static) -> Result<()> {
        self.ensure_open()?;
        self.workers.submit(request, sink)
    }

    /// Applies the model's chat template, keeping message text out of special-token parsing.
    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<Vec<PromptSegment>> {
        self.ensure_open()?;
        chat::render(&self.shared.model, messages)
    }

    /// Token ids `segments` would become as a prompt (no context-window limit applied).
    pub fn tokenize(&self, segments: &[PromptSegment]) -> Result<Vec<i32>> {
        self.ensure_open()?;
        let tokens = prompt::tokenize(&self.shared.model, segments, None)?;
        Ok(tokens.into_iter().map(|token| token.0).collect())
    }

    /// Stops all work and releases the contexts. Safe to call more than once.
    pub fn close(&self) {
        self.shared.closed.store(true, Ordering::Release);
        self.workers.shutdown();
    }

    fn ensure_open(&self) -> Result<()> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(err("LocalModel has been closed"));
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.close();
    }
}
