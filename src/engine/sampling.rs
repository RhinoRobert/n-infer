//! Builds the token sampler chain for one generation.

use std::time::{SystemTime, UNIX_EPOCH};

use llama_cpp_2::sampling::LlamaSampler;

use crate::options::ValidatedGenerateOptions;

/// How many recent tokens the repetition penalty looks back over.
pub const REPEAT_WINDOW: i32 = 64;

/// Penalties, then either greedy decoding (temperature 0) or top-k -> top-p -> temperature -> sample.
pub fn build(options: &ValidatedGenerateOptions, n_vocab: i32) -> LlamaSampler {
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
        if options.top_k > 0 {
            chain.push(LlamaSampler::top_k(options.top_k));
        }
        chain.push(LlamaSampler::top_p(options.top_p, 1));
        chain.push(LlamaSampler::temp(options.temperature));
        chain.push(LlamaSampler::dist(options.seed.unwrap_or_else(time_seed)));
    }
    LlamaSampler::chain_simple(chain)
}

fn time_seed() -> u32 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(1234)
}
