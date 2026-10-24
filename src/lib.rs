//! n-infer: in-process GGUF inference for Node.js.
//!
//! - `options`: option objects shared with JS
//! - `backend`: process-global llama.cpp init
//! - `engine`: model loading + generation loop (pure Rust)
//! - `binding`: the N-API surface

#![deny(clippy::all)]

mod backend;
mod binding;
mod engine;
mod error;
mod options;
