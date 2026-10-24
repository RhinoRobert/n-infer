# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/), and the project uses [Semantic Versioning](https://semver.org/) (pre-1.0, minor versions may change the API).

## [Unreleased]

## [0.1.0-beta.0]

First public beta.

### Added

- `LocalModel` with `init()`, `generate()`, `stream()`, `tokenize()`, `countTokens()`, `close()` and `Symbol.dispose`.
- Chat messages applied through the model's own template, with message text kept out of control-token parsing.
- Sampling options: `temperature`, `topP`, `topK`, `repetitionPenalty`, `seed`, `stop` sequences, `maxTokens`.
- `AbortSignal` support for `generate()` and `stream()`.
- Warm KV cache with automatic prefix reuse (`cache: false` to opt out).
- Machine-aware defaults: physical-core threads, VRAM-fitted GPU layers (`gpuLayers: 'auto'`), batch sizes.
- `hardware()` and `backend` exports; `modelSha256` to pin a model file.
- Bounded worker pool (`maxConcurrent`, `maxQueue`) with a fixed memory footprint.
- Prebuilt binaries for Windows x64, Linux x64/arm64 and macOS arm64 (Metal), plus opt-in CUDA add-ons.

### Security

- Input limits, finite-number checks, GGUF header validation, own-property-only option copying.
- Panics are contained and reported; no request can hang after a native failure.
