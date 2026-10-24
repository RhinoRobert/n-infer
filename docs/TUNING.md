# Tuning

n-infer picks defaults for the machine it runs on, so most people never touch these. This page explains what is chosen, why, and what was measured.

## What is automatic

| Setting                    | Default                                   | Why                                                                                          |
| -------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| `gpuLayers`                | `'auto'` on GPU builds, `0` on CPU builds | llama.cpp's fitter splits the model between VRAM and RAM for the free memory right now       |
| `threads`/`threadsBatch`   | physical core count                       | Hyper-threads share execution units and do not help memory-bound decoding                    |
| `batchSize` / `ubatchSize` | 2048 / 512                                | Large prompts are processed in big chunks                                                    |
| `topK`                     | 40                                        | Top-p alone sorts the whole vocabulary on every token; capping candidates first removes that |
| Kernel targets             | this CPU and GPU (`npm run build`)        | Native builds use exactly the instruction sets and CUDA architecture present                 |

Releases are built with `npm run build:portable` (AVX2-class CPUs, multi-generation CUDA) so one binary runs widely; building from source on your own machine gives the native-tuned version.

## What was measured

Machine: AMD Ryzen 9 5900 (12 cores), 56 GB RAM at 2133 MT/s, NVIDIA RTX 3080 10 GB, Windows 11. Model: a 4.4 GB 4-bit quantized chat model. Greedy unless noted; see [BENCHMARKS.md](BENCHMARKS.md).

| Finding                                                | Result                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| Top-k before top-p (sampled decoding, GPU)             | 81 -> 99.4 tokens/s (+23%), within 3% of greedy               |
| GPU decode is memory-bandwidth bound                   | about 100 tokens/s (roughly 58% of the card's peak bandwidth) |
| Micro-batch 256 / 512 / 1024 / 2048 (GPU prompt speed) | 3135 / 3390 / 3455 / 3436 tokens/s: within noise above 512    |
| Generation threads 1 / 4 / default (GPU)               | 98.7 / 99.2 / 100.8 tokens/s: threads do not matter on GPU    |
| CPU-only decode, threads 4 / 6 / 8 / 12 / 24           | 4.9 / 5.0 / 4.9 / 5.0 / 4.6 tokens/s: flat                    |
| Partial GPU offload, 8 / 16 / 24 / all layers          | 6.5 / 11.4 / 25.1 / about 100 tokens/s                        |

## Reading the numbers

- **On the GPU you are at the hardware ceiling.** Decoding reads the whole model once per token, so tokens per second is roughly memory bandwidth divided by model size. More threads or bigger batches cannot beat that; a smaller quantization or a faster GPU can.
- **CPU decoding is limited by RAM speed, not cores.** About 5 tokens/s here equals 4.4 GB x 5 = 22 GB/s, which is what dual-channel DDR4-2133 delivers. On this machine the installed memory is rated faster than it runs: enabling XMP/DOCP in the BIOS (or matching the DIMMs) would raise CPU decoding in proportion to the bandwidth gained.
- **Partial offload is nearly linear:** every layer you can fit in VRAM helps. `gpuLayers: 'auto'` finds the largest split that leaves 1 GiB of VRAM free.

## Manual overrides

```js
new LocalModel({
  modelPath,
  gpuLayers: 20, // fixed split instead of 'auto'
  threads: 6, // generation threads
  threadsBatch: 12, // prompt-processing threads
  batchSize: 2048,
  ubatchSize: 1024,
  contextSize: 8192,
  maxConcurrent: 2, // two parallel generations, two contexts in memory
});
```

Bigger `contextSize` and more `maxConcurrent` use more memory: each worker holds a full KV cache.
