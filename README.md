# n-infer

In-process GGUF inference for Node.js. Rust + llama.cpp (`llama-cpp-2`) compiled to a native addon with napi-rs. No Python, no HTTP server.

```js
import { LocalModel } from 'n-infer';

const ai = new LocalModel({ modelPath: './models/Llama-3-8B.gguf', gpuLayers: 30 });
await ai.init();

console.log(await ai.generate('Parse this data:'));

for await (const chunk of ai.stream('Analyze this data:')) process.stdout.write(chunk);
```

Generation runs off the event loop (libuv pool for `generate`, a dedicated thread for `stream`). Breaking out of a `for await` cancels generation. Prompts are passed raw: apply your model's chat template yourself.

## Layout

```
src/            Rust core
  lib.rs          module map
  options.rs      option objects shared with JS
  backend.rs      process-global llama.cpp init
  engine.rs       model loading + generation loop (no N-API types)
  binding.rs      the N-API surface JS sees
  error.rs        error conversion
lib/            JavaScript wrapper (vanilla ESM)
  index.js        public exports
  local-model.js  LocalModel class
  stream.js       native callback -> async iterator
  native.js       finds and loads the compiled .node
  index.d.ts      types
scripts/build.js  picks CUDA or CPU and runs the napi build
examples/basic.js
```

## Prerequisites

Windows: Rust (MSVC toolchain), Visual Studio Build Tools ("Desktop development with C++"), CMake, LLVM (set `LIBCLANG_PATH` to `C:\Program Files\LLVM\bin`). For CUDA add the CUDA Toolkit.
Linux / Raspberry Pi: `build-essential cmake clang` and Rust.

## Build and test

```bash
npm install
npm run build          # CUDA if nvcc is on PATH, else CPU
npm run build:cpu      # force CPU (LattePanda / Raspberry Pi)
npm run build:cuda     # force CUDA
node examples/basic.js ./models/your-model.gguf 30
```

The build writes `lib/n-infer.node`. To use it from another local project: `npm install ../n-infer`.

## API

- `new LocalModel({ modelPath, gpuLayers = 0, contextSize = 4096, threads, verbose = false })`
- `await ai.init()`
- `await ai.generate(prompt, { maxTokens = 256, temperature = 0.8, topP = 0.95, repetitionPenalty = 1.1, seed })` (`temperature: 0` is greedy, `repetitionPenalty: 1` is off)
- `ai.stream(prompt, options)`: async iterator of text chunks
- `ai.close()`; the `backend` export is `'cuda' | 'vulkan' | 'cpu'`

A CPU-only binary ignores `gpuLayers` and emits a process warning. `verbose: true` prints llama.cpp's load logs to stderr.
