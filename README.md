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

## Prerequisites

Windows: Rust (rustup, MSVC toolchain), Visual Studio Build Tools ("Desktop development with C++"), CMake, LLVM (bindgen needs libclang; set `LIBCLANG_PATH` to `C:\Program Files\LLVM\bin`). For CUDA add the CUDA Toolkit.
Linux / Raspberry Pi: `build-essential cmake clang` and Rust.

## Build and test

```bash
npm install
npm run build          # CUDA if nvcc is on PATH, else CPU
npm run build:cpu      # force CPU (LattePanda / Raspberry Pi)
npm run build:cuda     # force CUDA
node examples/basic.js ./models/your-model.gguf 30
```

`npm run build` produces `n-infer.node` in the project root; `index.js` loads it directly. To use it from another local project: `npm install ../n-infer` (or `npm link`).

## API

- `new LocalModel({ modelPath, gpuLayers = 0, contextSize = 4096, threads, verbose = false })`
- `await ai.init()`
- `await ai.generate(prompt, { maxTokens = 256, temperature = 0.8, topP = 0.95, seed })` (`temperature: 0` is greedy)
- `ai.stream(prompt, options)` async iterator of text chunks
- `ai.close()`; `backend` export is `'cuda' | 'vulkan' | 'cpu'`

A CPU-only binary ignores `gpuLayers` and emits a process warning.
