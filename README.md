# n-infer

**Run GGUF language models inside your Node.js process.** llama.cpp, wrapped in Rust, tuned for low latency. No server to start, no Python, no HTTP hop, and zero JavaScript dependencies.

```js
import { LocalModel } from 'n-infer';

const ai = await new LocalModel({ modelPath: './models/model.gguf' }).init();

console.log(await ai.generate('The capital of France is'));

for await (const piece of ai.stream('Write a haiku about Rust:')) process.stdout.write(piece);
```

- **Fast on the machine it runs on.** GPU layers are fitted to your free VRAM, threads match your physical cores, and a warm context reuses the cache of any prompt prefix it has already seen.
- **Small and dependency-free.** The package is 11 files and about 10 kB; native binaries arrive as separate per-platform packages.
- **Safe by default.** Untrusted text can never become control tokens, inputs are bounded, `close()` and abort are deterministic, and a native failure can never leave a caller hanging. See the [threat model](docs/THREAT_MODEL.md).
- **Typed.** Ships TypeScript declarations; `AbortSignal`, `using` / `Symbol.dispose` and async iteration work as you would expect.

> **Status: beta, Windows x64.** The API may still change before 1.0. Prebuilt binaries are published for Windows x64, which is the platform it is built and tested on. The source is portable, so you can build it on Linux or macOS yourself, but that is not tested or supported yet.

## Install

```bash
npm install n-infer@beta
```

While n-infer is in beta, `@beta` selects the latest beta release.

This installs the prebuilt CPU binary for **Windows x64** (any CPU with AVX2, roughly 2013 or newer) and needs Node.js 20.19 or newer. It also needs the Microsoft Visual C++ Redistributable, which most Windows machines already have.

**NVIDIA GPU** is an opt-in add-on, so CPU users do not download it:

```bash
npm install @n-infer/win32-x64-msvc-cuda@beta
```

It needs a recent NVIDIA driver and the CUDA 12 runtime libraries (`cublas`) on your `PATH`; installing the CUDA Toolkit provides them. If the add-on cannot load, n-infer falls back to the CPU build and prints a warning. Check what you got:

```js
import { backend, hardware } from 'n-infer';
console.log(backend); // 'cuda' | 'cpu' (also 'vulkan' | 'metal' in source builds)
console.log(hardware()); // cores, CPU features, every device and its free memory
```

You need a model in GGUF format, for example from [Hugging Face](https://huggingface.co/models?library=gguf).

## Try it in your browser

A small chat app ships in the repository. It loads a model inside the Node process and streams replies to a page with live speed and GPU stats. It finds models you already have (including Ollama's folder, without Ollama running).

```bash
git clone https://github.com/RhinoRobert/n-infer && cd n-infer
npm install && npm run build
npm run demo            # then open http://127.0.0.1:3939
```

Add `--model ./your.gguf` or `--models-dir ./models` to point it at your own files. It listens on 127.0.0.1 only.

## Using it

### Chat

Pass messages and the model's own chat template is applied for you. Message text is always treated as plain text, never as control tokens.

```js
const reply = await ai.generate(
  [
    { role: 'system', content: 'You are a terse assistant.' },
    { role: 'user', content: 'What is the capital of Japan?' },
  ],
  { maxTokens: 64, temperature: 0.2 },
);
```

### Streaming, stopping and aborting

```js
const controller = new AbortController();
setTimeout(() => controller.abort(), 5_000);

try {
  for await (const piece of ai.stream(prompt, { stop: ['\n\n'], signal: controller.signal })) {
    process.stdout.write(piece);
  }
} catch (error) {
  if (error.name !== 'AbortError') throw error;
}
```

Breaking out of the loop also stops generation. `stop` sequences end the output without being included in it, even when they arrive split across tokens.

### Tokens and cleanup

```js
ai.countTokens(prompt); // exactly how generate() would tokenize it
ai.close(); // frees RAM and VRAM before it returns
```

On Node 24+ (or TypeScript 5.2+) `using ai = await new LocalModel({ ... }).init();` closes it for you at the end of the scope.

## Configuration

`new LocalModel(config)`. Only `modelPath` is required.

| Option                     | Default                            | Meaning                                                    |
| -------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| `modelPath`                | -                                  | Path to a `.gguf` file                                     |
| `gpuLayers`                | `'auto'` on GPU builds, `0` on CPU | Layers on the GPU; `'auto'` fits the free VRAM             |
| `contextSize`              | `4096`                             | Context window in tokens                                   |
| `threads` / `threadsBatch` | physical cores                     | Threads for generation / prompt processing                 |
| `batchSize` / `ubatchSize` | `2048` / `512`                     | Prompt batch sizes                                         |
| `maxConcurrent`            | `1`                                | Parallel generations (each holds a full context in memory) |
| `maxQueue`                 | `64`                               | Waiting requests before new ones are rejected              |
| `modelSha256`              | -                                  | Refuse to load unless the file's SHA-256 matches           |
| `verbose`                  | `false`                            | Print llama.cpp's own logs                                 |

`generate()` / `stream()` options: `maxTokens` (256), `temperature` (0.8; 0 is greedy), `topP` (0.95), `topK` (40), `repetitionPenalty` (1.1), `seed`, `stop`, `cache` (true), `parseSpecial` (false), `signal`. Types and ranges are in [`lib/index.d.ts`](lib/index.d.ts) and invalid values are rejected with a clear message.

Unknown option names are errors, so a typo like `gpulayers` cannot silently do nothing.

## Performance

Measured on an RTX 3080 (10 GB) with a 4.4 GB quantized model, greedy decoding, Windows 11:

| Metric                                   | Result               |
| ---------------------------------------- | -------------------- |
| Generation                               | about 100 tokens/s   |
| Prompt processing                        | about 3,400 tokens/s |
| Repeat call with a warm prefix (1 token) | about 10 ms          |
| Model load (auto GPU fit)                | 3 to 4 s             |

On this machine the GPU is at the hardware ceiling (memory bandwidth), and CPU-only decoding (about 5 tokens/s) is limited by RAM speed rather than cores. Details, methodology and how to reproduce them are in [docs/TUNING.md](docs/TUNING.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md). **We do not claim to be faster than other libraries until the benchmark in `bench/` shows it on the same model and hardware.**

## How it compares

[node-llama-cpp](https://github.com/withcatai/node-llama-cpp) is the established, full-featured option. n-infer is younger and narrower, and aims to be the best choice when latency, footprint and safety defaults matter most.

|                                             | n-infer                        | node-llama-cpp                                   |
| ------------------------------------------- | ------------------------------ | ------------------------------------------------ |
| Engine                                      | llama.cpp via Rust (napi-rs)   | llama.cpp via C++                                |
| JavaScript runtime dependencies             | **0**                          | many (CLI and tooling deps)                      |
| Prebuilt binaries                           | Windows x64 (CPU, CUDA add-on) | Windows, Linux, macOS (CPU, CUDA, Vulkan, Metal) |
| Chat templates                              | yes (model's own)              | yes (many wrappers)                              |
| Untrusted text cannot become control tokens | yes, by construction, tested   | yes                                              |
| Warm KV prefix reuse                        | automatic                      | yes                                              |
| Auto GPU fitting / auto thread sizing       | yes                            | partly                                           |
| JSON-schema / grammar-constrained output    | **not yet** (planned)          | yes                                              |
| Function calling                            | **not yet**                    | yes                                              |
| Embeddings / reranking                      | **not yet** (planned)          | yes                                              |
| CLI, model downloader                       | no (the core never downloads)  | yes                                              |
| Vulkan builds                               | planned                        | yes                                              |

If you need structured output, function calling or embeddings today, use node-llama-cpp. If you want a small, fast, auditable local-inference core for a bot or service, use n-infer.

## Security in one paragraph

Prompts and options are treated as untrusted; model files are not (a GGUF is parsed by C++ code, so only load ones you trust, and pin them with `modelSha256`). Limits on sizes, queue depth and concurrency are enforced, panics are contained, and releases are built in CI with npm provenance. Read [SECURITY.md](SECURITY.md) and the [threat model](docs/THREAT_MODEL.md) before exposing a model to untrusted users, and note that a model can still _follow instructions written in words_: treat its output as untrusted.

## Build from source

You need Rust, CMake, a C++ toolchain, libclang (and the CUDA Toolkit for GPU builds). See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/RhinoRobert/n-infer && cd n-infer
npm install
npm run build          # tuned for this machine (CUDA if nvcc is found)
npm run build:portable # what releases use: runs on any AVX2-class CPU
```

## Roadmap

Toward 1.0: JSON-schema / grammar-constrained output, embeddings, function calling, Vulkan, Linux and macOS prebuilt binaries (the code is portable but only Windows is tested today), and a published benchmark against other libraries. See [CHANGELOG.md](CHANGELOG.md).

## License

MIT. Third-party components: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
