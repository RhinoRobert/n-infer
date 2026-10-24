//! Machine-aware defaults for threads and batch sizes.
//!
//! llama.cpp defaults to 4 threads and a 512-token micro-batch regardless of hardware. We size
//! threads to the *physical* core count (SMT siblings share execution units and make
//! memory-bound decoding slower) and use larger batches for faster prompt processing.

use llama_cpp_2::context::params::LlamaContextParams;
use std::num::NonZeroU32;

use crate::options::ValidatedModelConfig;

/// Tokens submitted to one `decode` call while processing a prompt.
const DEFAULT_BATCH: u32 = 2048;
/// Tokens processed per compute graph pass (GPU prompt processing prefers larger values).
const DEFAULT_UBATCH: u32 = 512;

#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    /// Threads for single-token generation.
    pub threads: i32,
    /// Threads for prompt processing.
    pub threads_batch: i32,
    pub n_batch: u32,
    pub n_ubatch: u32,
}

impl Tuning {
    pub fn resolve(cfg: &ValidatedModelConfig) -> Self {
        let physical = num_cpus::get_physical().max(1) as i32;
        let n_batch = cfg.batch_size.unwrap_or(DEFAULT_BATCH);
        Self {
            threads: cfg.threads.unwrap_or(physical),
            threads_batch: cfg.threads_batch.unwrap_or(physical),
            n_batch,
            n_ubatch: cfg.ubatch_size.unwrap_or(DEFAULT_UBATCH).min(n_batch),
        }
    }

    pub fn context_params(&self, n_ctx: u32) -> LlamaContextParams {
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(self.n_batch)
            .with_n_ubatch(self.n_ubatch)
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads_batch)
    }
}
