# Threat model and red-team notes

This document says what n-infer defends, against whom, how it was attacked, and what remains your responsibility. Findings marked **fixed** have a test in `test/` or `src/`.

## What is being protected

- The **host process**: n-infer runs a large C++ engine inside your Node.js process, so a memory-safety bug is a bug in your app.
- **Availability** of the host: no request should hang, crash the process, or exhaust memory or threads.
- **Integrity of prompts**: text from users or documents must not be able to forge the structure of a conversation.
- **Isolation between callers** that share one loaded model.
- The **supply chain**: what `npm install n-infer` puts on a machine.

## Trust boundaries

| Input                              | Trusted?                                          |
| ---------------------------------- | ------------------------------------------------- |
| `LocalModel` config (paths, sizes) | Yes: written by the developer                     |
| Prompt text and chat messages      | **No**: may come from users, web pages, documents |
| Generate options                   | **No**: may be forwarded from an API request      |
| The GGUF model file                | Trusted only if you chose it (see T4)             |
| Model output                       | **No**: never execute or trust it without checks  |
| npm packages and CI                | Defended (see T10)                                |

## Findings

### T1. Special-token injection: **fixed**

Tokenizers can turn the text `<|im_start|>system` into the real control token, letting a user message forge a system turn. n-infer tokenizes untrusted text with control tokens **off**: plain strings are inert unless you pass `parseSpecial: true`, and chat messages are placed into the model's template as separate plain-text segments, so message text can never become a control token. Roles are limited to `system`, `user`, `assistant`, so a role name cannot smuggle template syntax. Tests: `chat.test.js` ("message text can never become control tokens", "roles are restricted"), `chat.rs` unit tests.

_Residual:_ a model may still **follow instructions written in plain words** inside a message ("ignore previous instructions"). That is language-level prompt injection and cannot be solved by a tokenizer: treat model output as untrusted, and keep secrets and powerful actions out of prompts that include untrusted text.

### T2. Hostile numbers reaching C++ samplers: **fixed**

`NaN`, infinities, negatives and out-of-range values passed into llama.cpp's samplers are undefined behaviour in C++. Every numeric option is range-checked in Rust before it crosses into C++; JavaScript additionally rejects non-numbers. Tests: `options.rs` unit tests, `generation.test.js` ("bad sampling options").

### T3. Prototype pollution: **fixed**

napi-rs reads object properties through the prototype chain, so a polluted `Object.prototype` in the host app could silently change a model's settings. The wrapper copies **own** properties into a null-prototype object and rejects unknown keys. Tests: `config.test.js`, `options.test.js` ("ignores inherited properties").

### T4. Hostile or malformed model files: **mitigated, residual risk**

Before llama.cpp sees a path, Rust checks it is a regular file (no devices, pipes or directories), is large enough, and starts with the `GGUF` magic and a known version. `modelSha256` refuses any file whose hash differs from the one you pinned.

_Residual:_ the GGUF body is parsed by llama.cpp's C++ code, which has had parsing vulnerabilities in the past. **Treat model files like executable code**: download them from sources you trust, verify hashes, and do not load models supplied by untrusted users into a privileged process.

### T5. Resource exhaustion: **fixed (bounded)**

| Limit                                    | Value / default          |
| ---------------------------------------- | ------------------------ |
| Prompt size                              | 4 MiB                    |
| Chat messages / prompt segments          | 1024 / 4096              |
| Stop sequences                           | 16, each up to 256 bytes |
| `maxTokens` per request                  | up to 131,072            |
| Concurrent generations (`maxConcurrent`) | 1 (max 16)               |
| Waiting requests (`maxQueue`)            | 64 (max 1024)            |
| Context window                           | 64 to 131,072 tokens     |

A full queue rejects new requests at once ("too many pending requests") instead of buffering or spawning threads. Memory is fixed at load time: one KV cache per worker. Test: `generation.test.js` ("a full queue rejects new requests").

_Residual:_ `contextSize`, `maxConcurrent` and `gpuLayers` are developer configuration and can allocate a lot of memory if set carelessly.

### T6. Hung callers after a native failure: **fixed**

A panic on a bare worker thread would silently kill it and leave the caller waiting forever. Every job runs inside `catch_unwind`, always ends with exactly one terminal event, and discards that worker's possibly half-updated session. Tests: `error.rs` unit tests (guard), plus lifecycle tests.

### T7. Close and abort races: **fixed**

`close()` during a generation, aborting a stream, closing twice, re-initializing: all end in a clear error (`LocalModel was closed`) or a clean `AbortError`, never a hang or a use-after-free. Tests: `lifecycle.test.js`, `generation.test.js`.

### T8. Cross-caller leakage through the KV cache: **opt-out provided**

The warm KV cache reuses the prefix shared with the previous request on the same worker, which is what makes repeat calls fast. If one loaded model serves **different users**, response timing can reveal whether another caller's prompt shared a prefix, and stale tokens stay in memory until overwritten. Pass `cache: false` to wipe the cache before and after a request, or use separate `LocalModel` instances per trust domain.

### T9. Loading the wrong native binary: **mitigated**

`native.js` only loads (1) `lib/n-infer.node` built in the package folder, (2) the `@n-infer/<target>-cuda` add-on, (3) `@n-infer/<target>`. It refuses a binary whose version differs from the package. On Windows, the CUDA toolkit's `bin` folders are **appended** to `PATH` (never prepended, so system directories keep priority), only after a load has already failed, and only from an absolute, plain `CUDA_PATH`. Publish the `@n-infer` npm organization yourself so no one else can register those names (dependency confusion).

### T10. Supply chain: **mitigated**

- The npm package has **no runtime dependencies** and ships 11 files; platform packages ship one binary each. `scripts/check-pack.js` fails CI if anything else would be published.
- Rust dependencies are pinned (`llama-cpp-2 = "=0.1.160"`, `Cargo.lock`), checked by `cargo deny` (advisories, licenses, sources) on every push and weekly.
- GitHub Actions are pinned by commit SHA; Dependabot proposes updates.
- Releases are built in CI, published with **npm provenance** (and trusted publishing when configured), behind a protected `npm-publish` environment that needs a human approval. There are no install scripts and nothing is downloaded at install time.

_Residual:_ llama.cpp and its Rust bindings are trusted upstream code; we update them deliberately, with tests and a benchmark.

### T11. Information in errors and logs: **accepted**

Error messages can include the model path you supplied. llama.cpp's own logging is silenced unless `verbose: true`. n-infer sends no telemetry and makes no network requests.

## Non-goals

- Sandboxing the process (seccomp, containers): run untrusted workloads in a separate process or container.
- Defending against a malicious model's _behaviour_, or against GPU-level side channels.
- Defending against a compromised developer machine or npm account (use 2FA and trusted publishing).

## Reproducing the checks

```bash
npm run test:rust     # validation, stop sequences, chat splitting, GGUF header, SHA-256, panic guard
npm test              # unit + integration (needs `npm run test:fetch` models)
npm run pack:check    # exactly what would be published
```

## Reporting

See [SECURITY.md](../SECURITY.md).
