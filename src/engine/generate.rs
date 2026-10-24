//! One generation: tokenize, prefill (reusing the cache), then sample until a stop condition.

use std::sync::atomic::Ordering;

use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::vocab::LlamaVocab;

use super::session::Session;
use super::stop::StopMatcher;
use super::{prompt, sampling, Request, Shared};
use crate::error::{err, Result};

/// Runs `request` on `slot`'s session (creating one if the slot is empty), passing each decoded
/// piece of text to `emit`.
pub fn run(
    shared: &Shared,
    slot: &mut Option<Session>,
    request: &Request,
    emit: &mut dyn FnMut(String),
) -> Result<()> {
    let options = &request.options;
    let interrupted =
        || request.cancel.load(Ordering::Relaxed) || shared.closed.load(Ordering::Relaxed);

    ensure_open(shared)?;
    if options.max_tokens == 0 {
        return Ok(());
    }
    let tokens = prompt::tokenize(&shared.model, &request.prompt, Some(shared.context_size))?;
    let max_tokens = (options.max_tokens as usize).min(shared.context_size as usize - tokens.len());

    if slot.is_none() {
        *slot = Some(Session::new(shared)?);
    }
    let Some(session) = slot.as_mut() else {
        return Err(err("no inference session available"));
    };
    if !options.cache {
        session.clear();
    }

    let Some(mut pos) = session.prefill(&tokens, &interrupted)? else {
        return ensure_open(shared);
    };

    let mut sampler = sampling::build(options, shared.model.n_vocab());
    sampler.accept_many(&tokens[tokens.len().saturating_sub(sampling::REPEAT_WINDOW as usize)..]);

    let vocab = shared.model.vocab();
    let mut utf8 = encoding_rs::UTF_8.new_decoder();
    let mut stops = StopMatcher::new(options.stop.clone());
    let mut hit_stop = false;

    for step in 0..max_tokens {
        if interrupted() {
            break;
        }
        // `sample` also records the token with the sampler, so it must not be accepted again.
        let token = session.sample(&mut sampler);
        if vocab.is_eog(token) {
            break;
        }

        let piece = decode_piece(&vocab, &mut utf8, token);
        if !piece.is_empty() {
            let (text, stopped) = stops.push(&piece);
            if !text.is_empty() {
                emit(text);
            }
            if stopped {
                hit_stop = true;
                break;
            }
        }

        // The last token is never fed back in: nothing will read the logits it would produce.
        if step + 1 < max_tokens && !interrupted() {
            session.advance(token, pos)?;
            pos += 1;
        }
    }

    if !hit_stop {
        let rest = stops.flush();
        if !rest.is_empty() {
            emit(rest);
        }
    }
    if !options.cache {
        session.clear();
    }
    ensure_open(shared)
}

fn ensure_open(shared: &Shared) -> Result<()> {
    if shared.closed.load(Ordering::Relaxed) {
        return Err(err("LocalModel was closed"));
    }
    Ok(())
}

/// Streaming UTF-8 decode, so a character split across two tokens is emitted whole.
fn decode_piece(
    vocab: &LlamaVocab<'_>,
    utf8: &mut encoding_rs::Decoder,
    token: LlamaToken,
) -> String {
    let bytes = vocab.token_to_piece(token, false, None);
    let mut text = String::with_capacity(utf8.max_utf8_buffer_length(bytes.len()).unwrap_or(16));
    let _ = utf8.decode_to_string(&bytes, &mut text, false);
    text
}
