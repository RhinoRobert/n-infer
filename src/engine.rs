//! Model loading and the token generation loop. No N-API types in here.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::backend;
use crate::error::{err, Result};
use crate::options::{GenerateOptions, ModelConfig};

const DEFAULT_CONTEXT: u32 = 4096;
const DEFAULT_MAX_TOKENS: u32 = 256;
/// Prompt tokens decoded per batch; longer prompts are fed in chunks.
const PROMPT_BATCH: usize = 512;
/// How many recent tokens the repetition penalty looks back over.
const REPEAT_WINDOW: i32 = 64;
const DEFAULT_REPEAT_PENALTY: f64 = 1.1;

pub struct Engine {
    model: LlamaModel,
    context_size: u32,
    threads: Option<i32>,
    verbose: bool,
}

impl Engine {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let verbose = config.verbose.unwrap_or(false);
        let backend = backend::get(verbose)?;
        let params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers.unwrap_or(0));
        let model = LlamaModel::load_from_file(backend, &config.model_path, &params)
            .map_err(|e| err(format!("failed to load '{}': {e}", config.model_path)))?;
        Ok(Self {
            model,
            context_size: config.context_size.unwrap_or(DEFAULT_CONTEXT),
            threads: config.threads,
            verbose,
        })
    }

    /// Decodes `prompt`, then samples until end-of-generation, `max_tokens` or cancellation,
    /// passing each decoded text piece to `emit`.
    ///
    /// A fresh context is created per call so concurrent generations never share KV state.
    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
        cancel: &AtomicBool,
        mut emit: impl FnMut(String),
    ) -> Result<()> {
        let mut ctx = self.new_context()?;
        let tokens = self.tokenize(prompt)?;
        let max_tokens = (options.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS) as usize)
            .min(self.context_size as usize - tokens.len());

        let mut batch = LlamaBatch::new(PROMPT_BATCH, 1);
        let mut pos = feed_prompt(&mut ctx, &mut batch, &tokens)?;
        let mut sampler = build_sampler(options, self.model.n_vocab());
        let mut utf8 = encoding_rs::UTF_8.new_decoder();
        let mut logits_idx = batch.n_tokens() - 1;

        for _ in 0..max_tokens {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let token = sampler.sample(&ctx, logits_idx);
            sampler.accept(token);
            if self.model.vocab().is_eog(token) {
                break;
            }

            let text = self.decode_piece(&mut utf8, token);
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

    fn new_context(&self) -> Result<LlamaContext<'_>> {
        let mut params =
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.context_size));
        if let Some(threads) = self.threads {
            params = params.with_n_threads(threads).with_n_threads_batch(threads);
        }
        let backend = backend::get(self.verbose)?;
        self.model.new_context(backend, params).map_err(err)
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
        let bytes = self.model.vocab().token_to_piece(token, true, None);
        let mut text = String::with_capacity(utf8.max_utf8_buffer_length(bytes.len()).unwrap_or(16));
        let _ = utf8.decode_to_string(&bytes, &mut text, false);
        text
    }
}

/// Decodes the prompt in chunks; only the final token requests logits. Returns the next position.
fn feed_prompt(ctx: &mut LlamaContext, batch: &mut LlamaBatch, tokens: &[LlamaToken]) -> Result<i32> {
    let last = tokens.len() - 1;
    let mut pos: i32 = 0;
    for chunk in tokens.chunks(PROMPT_BATCH) {
        batch.clear();
        for &token in chunk {
            batch.add(token, pos, &[0], pos as usize == last).map_err(err)?;
            pos += 1;
        }
        ctx.decode(batch).map_err(err)?;
    }
    Ok(pos)
}

fn build_sampler(options: &GenerateOptions, n_vocab: i32) -> LlamaSampler {
    let temperature = options.temperature.unwrap_or(0.8) as f32;
    let repeat_penalty = options.repetition_penalty.unwrap_or(DEFAULT_REPEAT_PENALTY) as f32;

    let mut chain = Vec::new();
    if repeat_penalty != 1.0 {
        chain.push(LlamaSampler::penalties(n_vocab, REPEAT_WINDOW, repeat_penalty, 0.0, 0.0));
    }
    if temperature <= 0.0 {
        chain.push(LlamaSampler::greedy());
    } else {
        let seed = options.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(1234)
        });
        chain.push(LlamaSampler::top_p(options.top_p.unwrap_or(0.95) as f32, 1));
        chain.push(LlamaSampler::temp(temperature));
        chain.push(LlamaSampler::dist(seed));
    }
    LlamaSampler::chain_simple(chain)
}
