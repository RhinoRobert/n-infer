//! Plain option objects shared with JavaScript (camelCase on the JS side).

use napi_derive::napi;

#[napi(object)]
pub struct ModelConfig {
    pub model_path: String,
    /// Layers to offload to the GPU. Ignored by CPU-only builds.
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
    pub threads: Option<i32>,
    /// Forward llama.cpp's own logging to stderr.
    pub verbose: Option<bool>,
}

#[napi(object)]
#[derive(Default)]
pub struct GenerateOptions {
    pub max_tokens: Option<u32>,
    /// 0 (or less) selects greedy decoding.
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    /// Penalty on recently seen tokens; 1.0 disables it. Default 1.1.
    pub repetition_penalty: Option<f64>,
    pub seed: Option<u32>,
}
