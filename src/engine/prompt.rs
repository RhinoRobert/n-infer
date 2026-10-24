//! Turns prompt segments into tokens.

use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

use crate::error::{err, Result};
use crate::options::PromptSegment;

/// Tokenizes `segments` in order. Only segments marked `special` may produce control tokens,
/// so untrusted text can be passed through with `special: false` and stay inert.
///
/// A beginning-of-sequence token is added once, up front, unless the text already starts with one.
/// With `context_size` set, a prompt that cannot fit in the window is rejected.
pub fn tokenize(
    model: &LlamaModel,
    segments: &[PromptSegment],
    context_size: Option<u32>,
) -> Result<Vec<LlamaToken>> {
    let vocab = model.vocab();
    let bos = bos_text(model);
    let mut tokens: Vec<LlamaToken> = Vec::new();

    for segment in segments.iter().filter(|s| !s.text.is_empty()) {
        let starts_with_bos = bos.as_deref().is_some_and(|b| segment.text.starts_with(b));
        let add_bos = tokens.is_empty() && !starts_with_bos;
        tokens.extend(vocab.tokenize(segment.text.as_bytes(), add_bos, segment.special));
    }

    if tokens.is_empty() {
        return Err(err("prompt produced no tokens"));
    }
    if let Some(limit) = context_size.filter(|&limit| tokens.len() >= limit as usize) {
        return Err(err(format!(
            "prompt is {} tokens but the context window is {limit}",
            tokens.len()
        )));
    }
    Ok(tokens)
}

/// The text of the model's BOS token, if it has one.
fn bos_text(model: &LlamaModel) -> Option<String> {
    let vocab = model.vocab();
    let bos = vocab.bos();
    if bos.0 < 0 {
        return None;
    }
    String::from_utf8(vocab.token_to_piece(bos, true, None)).ok().filter(|text| !text.is_empty())
}
