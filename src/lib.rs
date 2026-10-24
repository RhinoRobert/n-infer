//! n-infer: in-process GGUF inference for Node.js.
//!
//! - `options`: option objects shared with JS
//! - `backend`: process-global llama.cpp init
//! - `hardware`: device / CPU feature discovery and the CPU-support guard
//! - `tuning`: machine-aware thread and batch defaults
//! - `engine`: model loading, KV-cache reuse and the generation loop (no N-API types)
//! - `binding`: the N-API surface

#![deny(clippy::all)]

mod backend;
mod binding;
mod engine;
mod error;
mod hardware;
mod options;
mod tuning;
