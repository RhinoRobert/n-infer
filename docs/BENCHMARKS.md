# Benchmarks

Numbers here are **measurements, not promises**: each row says what machine and model produced it, and `bench/` reproduces them.

## Reproduce

```bash
npm run build                                       # or install the published package
node bench/compare.js --model ./model.gguf --engine n-infer
npm install --prefix bench                          # adds node-llama-cpp for the comparison
node bench/compare.js --model ./model.gguf --engine both --runs 3
```

Each engine runs in its own process, three times, and the median is reported. Decoding is greedy so every engine does identical work; the GPU should be idle (pause miners, games and other models first).

## n-infer on the development machine

AMD Ryzen 9 5900 (12 cores), 56 GB RAM at 2133 MT/s, NVIDIA RTX 3080 10 GB, Windows 11, Node.js 24, a 4.4 GB 4-bit quantized chat model, native build, default settings.

| Metric                                 | Result               |
| -------------------------------------- | -------------------- |
| Generation (greedy)                    | about 100 tokens/s   |
| Generation (default sampler)           | about 99 tokens/s    |
| Prompt processing (about 2,000 tokens) | about 3,400 tokens/s |
| Warm one-token call                    | about 10 ms          |
| Model load (auto GPU fit)              | 3 to 4 s             |
| CPU only: generation                   | about 5 tokens/s     |

The CPU-only figure is limited by memory bandwidth, not cores: see [TUNING.md](TUNING.md).

## Comparison with node-llama-cpp

**Not published yet.** The comparison needs to run both engines on the same model and machine, and the result will be added here with its hardware and versions, whichever way it comes out. Until then n-infer makes no speed claim relative to other libraries.
