//! Plain option objects shared with JavaScript (camelCase on the JS side),
//! plus strict boundary validation before values reach the C/C++ engine.

use napi_derive::napi;

use crate::error::{err, Result};

pub const DEFAULT_CONTEXT: u32 = 4096;
pub const MIN_CONTEXT: u32 = 64;
pub const MAX_CONTEXT: u32 = 131_072;
pub const MAX_GPU_LAYERS: u32 = 1000;
pub const MAX_THREADS: i32 = 256;
pub const MAX_BATCH: u32 = 16_384;

pub const DEFAULT_MAX_TOKENS: u32 = 256;
pub const MAX_TOKENS_LIMIT: u32 = 131_072;
pub const DEFAULT_TEMPERATURE: f32 = 0.8;
pub const MAX_TEMPERATURE: f32 = 100.0;
pub const DEFAULT_TOP_P: f32 = 0.95;
/// Candidates kept before top-p. Without it, top-p sorts the whole vocabulary on every token.
pub const DEFAULT_TOP_K: i32 = 40;
pub const MAX_TOP_K: u32 = 100_000;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub const MAX_REPEAT_PENALTY: f32 = 10.0;

#[napi(object)]
pub struct ModelConfig {
    pub model_path: String,
    /// Layers to offload to the GPU. Ignored by CPU-only builds.
    /// Leave unset (or use `gpuAuto`) to let llama.cpp fit as many as the free VRAM allows.
    pub gpu_layers: Option<u32>,
    /// Pick GPU layers automatically. Default: on for GPU builds unless `gpuLayers` is given.
    pub gpu_auto: Option<bool>,
    pub context_size: Option<u32>,
    /// Threads for token generation (default: physical core count).
    pub threads: Option<i32>,
    /// Threads for prompt processing (default: physical core count).
    pub threads_batch: Option<i32>,
    /// Tokens per decode call while processing a prompt (default 2048).
    pub batch_size: Option<u32>,
    /// Tokens per compute pass; larger can speed up GPU prompt processing (default 512).
    pub ubatch_size: Option<u32>,
    /// Forward llama.cpp's own logging to stderr.
    pub verbose: Option<bool>,
}

pub struct ValidatedModelConfig {
    pub model_path: String,
    /// `None` means "ask llama.cpp to fit the model to the free VRAM".
    pub gpu_layers: Option<u32>,
    pub context_size: u32,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub verbose: bool,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<ValidatedModelConfig> {
        if self.model_path.trim().is_empty() || self.model_path.contains('\0') {
            return Err(err("invalid modelPath: must be a non-empty path without NUL bytes"));
        }
        if let Some(layers) = self.gpu_layers {
            if layers > MAX_GPU_LAYERS {
                return Err(err(format!(
                    "invalid gpuLayers ({layers}): must be <= {MAX_GPU_LAYERS}"
                )));
            }
        }
        let gpu_build = cfg!(any(feature = "cuda", feature = "vulkan"));
        let auto = self.gpu_auto.unwrap_or(self.gpu_layers.is_none()) && gpu_build;
        let gpu_layers = if auto { None } else { Some(self.gpu_layers.unwrap_or(0)) };
        let context_size = self.context_size.unwrap_or(DEFAULT_CONTEXT);
        if !(MIN_CONTEXT..=MAX_CONTEXT).contains(&context_size) {
            return Err(err(format!(
                "invalid contextSize ({context_size}): must be between {MIN_CONTEXT} and {MAX_CONTEXT}"
            )));
        }
        for (name, value) in [("threads", self.threads), ("threadsBatch", self.threads_batch)] {
            if let Some(n) = value {
                if !(1..=MAX_THREADS).contains(&n) {
                    return Err(err(format!(
                        "invalid {name} ({n}): must be between 1 and {MAX_THREADS}"
                    )));
                }
            }
        }
        for (name, value) in [("batchSize", self.batch_size), ("ubatchSize", self.ubatch_size)] {
            if let Some(n) = value {
                if !(1..=MAX_BATCH).contains(&n) {
                    return Err(err(format!("invalid {name} ({n}): must be between 1 and {MAX_BATCH}")));
                }
            }
        }
        Ok(ValidatedModelConfig {
            model_path: self.model_path.clone(),
            gpu_layers,
            context_size,
            threads: self.threads,
            threads_batch: self.threads_batch,
            batch_size: self.batch_size,
            ubatch_size: self.ubatch_size,
            verbose: self.verbose.unwrap_or(false),
        })
    }
}

#[napi(object)]
#[derive(Default)]
pub struct GenerateOptions {
    pub max_tokens: Option<u32>,
    /// 0 selects greedy decoding.
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    /// Candidates kept before top-p; 0 disables the cut. Default 40.
    pub top_k: Option<u32>,
    /// Penalty on recently seen tokens; 1.0 disables it. Default 1.1.
    pub repetition_penalty: Option<f64>,
    pub seed: Option<u32>,
}

pub struct ValidatedGenerateOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub seed: Option<u32>,
}

impl GenerateOptions {
    pub fn validate(&self) -> Result<ValidatedGenerateOptions> {
        let max_tokens = self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        if max_tokens > MAX_TOKENS_LIMIT {
            return Err(err(format!(
                "invalid maxTokens ({max_tokens}): must be <= {MAX_TOKENS_LIMIT}"
            )));
        }

        let temperature = self.temperature.unwrap_or(f64::from(DEFAULT_TEMPERATURE));
        if !temperature.is_finite() || !(0.0..=f64::from(MAX_TEMPERATURE)).contains(&temperature) {
            return Err(err(format!(
                "invalid temperature ({temperature}): must be a finite number between 0 and {MAX_TEMPERATURE}"
            )));
        }

        let top_p = self.top_p.unwrap_or(f64::from(DEFAULT_TOP_P));
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(err(format!(
                "invalid topP ({top_p}): must be a finite number in (0, 1]"
            )));
        }

        let top_k = self.top_k.map_or(DEFAULT_TOP_K, |k| k as i32);
        if self.top_k.is_some_and(|k| k > MAX_TOP_K) {
            return Err(err(format!("invalid topK: must be <= {MAX_TOP_K}")));
        }

        let repetition_penalty = self
            .repetition_penalty
            .unwrap_or(f64::from(DEFAULT_REPEAT_PENALTY));
        if !repetition_penalty.is_finite()
            || repetition_penalty <= 0.0
            || repetition_penalty > f64::from(MAX_REPEAT_PENALTY)
        {
            return Err(err(format!(
                "invalid repetitionPenalty ({repetition_penalty}): must be a finite number in (0, {MAX_REPEAT_PENALTY}]"
            )));
        }

        Ok(ValidatedGenerateOptions {
            max_tokens,
            temperature: temperature as f32,
            top_p: top_p as f32,
            top_k,
            repetition_penalty: repetition_penalty as f32,
            seed: self.seed,
        })
    }
}
