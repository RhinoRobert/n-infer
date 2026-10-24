# Architecture

n-infer is a thin, careful bridge: JavaScript on one side, llama.cpp on the other, a small amount of Rust in between that owns every rule about memory, threads and trust.

```
 your code
    │  generate() / stream() / tokenize()
    ▼
 lib/  (vanilla ESM, zero dependencies)
    │  config.js / options.js   copy own properties, check types
    │  stream.js                native callbacks  ->  async iteration + AbortSignal
    │  native.js                find and load the right binary
    ▼
 src/binding.rs   N-API surface: the only Rust file that talks to Node
    │  validates ranges (options.rs) and hands a Request to the engine
    ▼
 src/engine/
    ├─ mod.rs        Engine: load model, own the workers, close()
    ├─ workers.rs    bounded queue + N worker threads, panic-proof
    ├─ session.rs    one llama.cpp context + its KV cache (the only `unsafe`)
    ├─ generate.rs   tokenize -> prefill (reuse cache) -> sample loop -> stop
    ├─ prompt.rs     segments -> tokens (control tokens only where trusted)
    ├─ chat.rs       model chat template, with message text kept as plain text
    ├─ sampling.rs   sampler chain
    ├─ stop.rs       stop sequences split across tokens
    └─ loader.rs     GGUF header check, SHA-256, GPU fit
    ▼
 llama.cpp (CPU / CUDA / Metal / Vulkan)
```

## One request, end to end

1. `model.generate(prompt, options)` validates types and copies **own** properties only (a polluted `Object.prototype` cannot reach the native side).
2. A string prompt becomes one **segment** `{ text, special: false }`. Chat messages go through `renderChat`, which asks the model's chat template to render stand-in markers, then cuts the result at the markers: template text becomes `special: true`, message text stays `special: false`.
3. `binding.rs::run` checks ranges and sizes (Rust is the single source of truth) and submits a `Request` to the bounded queue. It returns at once.
4. A worker thread takes the request. Its **own** `Session` already holds a warm context; the prompt is tokenized, the longest prefix shared with the cached tokens is kept, and only the rest is evaluated.
5. Tokens are sampled and decoded to text with a streaming UTF-8 decoder, run through the stop-sequence matcher, and delivered to JavaScript through a thread-safe callback.
6. Exactly one terminal event ends every request, even if generation panics.

## Design decisions

- **Workers own their sessions.** A worker creates its context at load time and keeps it, so warm caches never move between threads, memory use is fixed (`maxConcurrent` contexts), and allocation failures surface in `init()`.
- **Bounded everything.** The queue (`maxQueue`), concurrency (`maxConcurrent`), prompt size, stop sequences, message count and `maxTokens` all have hard limits. A full queue rejects immediately instead of growing.
- **A single `unsafe`.** `Session` stores a `LlamaContext<'static>` next to an `Arc` of the model that owns it, declared in drop order. The SAFETY argument is written next to the code (`session.rs`).
- **Panics are contained** (`error::guard`, `workers::handle`). A panic discards that worker's session and reports an error; it can never leave a caller waiting.
- **Validation lives in Rust.** JavaScript checks only types, so there is one set of ranges and one set of error messages.
- **Close is deterministic.** `close()` flags the engine, drops the queue, and joins the workers, so VRAM is released before it returns.
- **The JavaScript package has no dependencies and ships no binaries.** Binaries come from per-platform packages (`@n-infer/<target>`); GPU builds are separate opt-in packages. `native.js` checks that the binary and the package are the same version.

## Threads

| Thread               | Does                                                             |
| -------------------- | ---------------------------------------------------------------- |
| Node main thread     | Your JavaScript; receives text through thread-safe callbacks     |
| libuv pool (briefly) | Loading a model (`loadModel`)                                    |
| `n-infer-worker-N`   | One per `maxConcurrent`: runs generations, owns a session        |
| llama.cpp / OpenMP   | Compute threads inside a decode call (`threads`, `threadsBatch`) |
