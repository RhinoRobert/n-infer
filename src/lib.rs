//! n-infer: in-process GGUF inference for Node.js.
//!
//! - `binding`: the N-API surface (the only file that talks to Node)
//! - `engine`: model loading, worker pool, KV-cache sessions and the generation loop
//! - `options`: option objects shared with JS, and their validation
//! - `hardware`, `tuning`: what this machine can do, and defaults sized to it
//! - `backend`, `error`: process-wide llama.cpp setup and the crate's error type

#![deny(clippy::all)]

mod backend;
mod binding;
mod engine;
mod error;
mod hardware;
mod options;
mod tuning;
