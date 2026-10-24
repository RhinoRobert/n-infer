//! A llama.cpp context plus the tokens currently in its KV cache.
//!
//! This file holds the crate's only `unsafe` lifetime trick; everything else is safe Rust.

use std::sync::{Arc, Mutex};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use super::Shared;
use crate::backend;
use crate::error::{err, Result};

/// llama.cpp mutates model state while creating a context, so creation is serialised.
static CONTEXT_INIT_LOCK: Mutex<()> = Mutex::new(());

pub struct Session {
    // Declaration order matters: `ctx` borrows the model and must drop before `_model` does.
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
    /// Exact tokens resident in the KV cache (sequence 0).
    cached_tokens: Vec<LlamaToken>,
    /// Prompt tokens submitted per `decode` call (the batch capacity).
    chunk: usize,
    /// Keeps the model alive for as long as `ctx` exists.
    _model: Arc<LlamaModel>,
}

impl Session {
    pub fn new(shared: &Shared) -> Result<Self> {
        let backend = backend::get(shared.verbose)?;
        let params = shared.tuning.context_params(shared.context_size);
        let _guard = CONTEXT_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: `ctx` is stored next to an `Arc` clone of the model and is declared before it,
        // so the context is always dropped first. The model lives on the heap behind the `Arc`
        // and is never moved or mutated, so the extended reference stays valid for the
        // context's entire life.
        let model: &'static LlamaModel = unsafe { &*Arc::as_ptr(&shared.model) };
        let ctx = model.new_context(backend, params).map_err(err)?;

        let chunk = shared.tuning.n_batch as usize;
        Ok(Self {
            ctx,
            batch: LlamaBatch::new(chunk, 1),
            cached_tokens: Vec::with_capacity(shared.context_size as usize),
            chunk,
            _model: Arc::clone(&shared.model),
        })
    }

    /// Forgets everything in the KV cache.
    pub fn clear(&mut self) {
        self.ctx.clear_kv_cache();
        self.cached_tokens.clear();
    }

    /// Evaluates `tokens`, reusing any prefix already in the cache. Returns the position of the
    /// next token, or `None` if `interrupted` became true part-way through.
    pub fn prefill(
        &mut self,
        tokens: &[LlamaToken],
        interrupted: &dyn Fn() -> bool,
    ) -> Result<Option<i32>> {
        let keep = self.reuse_prefix(tokens);
        let last = tokens.len() - 1;
        let mut pos = keep as i32;

        for chunk in tokens[keep..].chunks(self.chunk) {
            if interrupted() {
                return Ok(None);
            }
            self.batch.clear();
            for &token in chunk {
                self.batch.add(token, pos, &[0], pos as usize == last).map_err(err)?;
                pos += 1;
            }
            self.decode()?;
            self.cached_tokens.extend_from_slice(chunk);
        }
        Ok(Some(pos))
    }

    /// Evaluates one generated token at `pos`.
    pub fn advance(&mut self, token: LlamaToken, pos: i32) -> Result<()> {
        self.batch.clear();
        self.batch.add(token, pos, &[0], true).map_err(err)?;
        self.decode()?;
        self.cached_tokens.push(token);
        Ok(())
    }

    /// Samples the next token from the logits of the last evaluated position.
    pub fn sample(&self, sampler: &mut LlamaSampler) -> LlamaToken {
        sampler.sample(&self.ctx, -1)
    }

    /// Trims the cache to the longest prefix shared with `tokens` and returns its length.
    /// The final prompt token is never kept, so it is always re-decoded to produce fresh logits.
    fn reuse_prefix(&mut self, tokens: &[LlamaToken]) -> usize {
        let shared = self
            .cached_tokens
            .iter()
            .zip(tokens)
            .take_while(|(cached, wanted)| cached == wanted)
            .count()
            .min(tokens.len() - 1);

        if shared == 0 {
            if !self.cached_tokens.is_empty() {
                self.clear();
            }
            return 0;
        }
        if self.cached_tokens.len() > shared {
            let truncated =
                self.ctx.clear_kv_cache_seq(Some(0), Some(shared as u32), None).unwrap_or(false);
            if !truncated {
                // Recurrent and hybrid models cannot drop part of their state: start over.
                self.clear();
                return 0;
            }
            self.cached_tokens.truncate(shared);
        }
        shared
    }

    /// Decodes the current batch; a failed decode leaves the cache in an unknown state, so wipe it.
    fn decode(&mut self) -> Result<()> {
        self.ctx.decode(&mut self.batch).map_err(|e| {
            self.ctx.clear_kv_cache();
            self.cached_tokens.clear();
            err(e)
        })
    }
}
