# Packaging notes

How the binaries are built and shipped, and the experiments behind those choices. All numbers were measured on Windows 11 with an AMD Ryzen 9 5900 (Zen 3, AVX2, no AVX-512) and an RTX 3080, using CUDA 13.4.

## What ships

| Package                             | Contents                                                     | Packed size |
| ----------------------------------- | ------------------------------------------------------------ | ----------- |
| `n-infer`                           | JavaScript and types, no dependencies, no binary             | 10 kB       |
| `@n-infer/<platform>`               | CPU binary (macOS arm64: Metal)                              | 2.1 MB      |
| `@n-infer/<platform>-cuda` (opt-in) | CUDA binary; falls back to the CPU package if it cannot load | 20 MB       |

`npm install n-infer` pulls the one CPU package that matches the machine (`optionalDependencies` filtered by `os`/`cpu`/`libc`). GPU add-ons are separate so CPU users never download them.

## Experiment 1: can the CUDA add-on avoid needing the CUDA toolkit?

**Windows: no.** The CUDA build imports exactly one CUDA library, `cublas64_<major>.dll` (which loads `cublasLt64`). The CUDA runtime itself is already linked in, and the driver (`nvcuda.dll`) comes with the GPU driver. The toolkit ships **no static cuBLAS** on Windows (only `cudart_static`), and the DLLs are too large to bundle:

| DLL (CUDA 13.4) | Size   |
| --------------- | ------ |
| `cublas64_13`   | 52 MB  |
| `cublasLt64_13` | 470 MB |

So the Windows add-on needs the CUDA runtime on the machine (installing the CUDA Toolkit provides it). To make that painless, `lib/native.js`:

- loads the add-on, and if Windows cannot find the DLLs, appends `%CUDA_PATH%\bin` to `PATH` once and retries;
- if it still cannot load, warns with what is missing and **falls back to the CPU package**.

All four cases were verified from a clean project installed from the packed tarballs: CUDA present (loads CUDA), CUDA runtime missing (warns, runs on CPU), only `CUDA_PATH` set (repairs `PATH`, loads CUDA), CPU package only.

**Linux:** the upstream build links cudart, cuBLAS and cuBLASLt statically, so only the NVIDIA driver is needed. The resulting binary will be much larger; its size is the first thing to check in the CI build (`linux-x64-gnu-cuda`). If it is too big to publish comfortably, switch that target to dynamic linking and document the runtime requirement, as on Windows.

**CUDA version:** DLL names carry the major version (`cublas64_12` vs `cublas64_13`), so a binary built against CUDA 12 needs a CUDA 12 runtime. Release builds use CUDA 12.8 (works with current drivers); builds from source use whatever is installed.

**Not tried:** a Vulkan build, which needs only the GPU driver (no toolkit) and also covers AMD and Intel GPUs, at some cost in speed on NVIDIA cards. It is the natural "GPU with zero install" option and is on the roadmap.

## Experiment 2: one CPU binary that is fast on any CPU

llama.cpp can build its CPU code as separate libraries, one per instruction-set level, and pick the best at runtime. n-infer exposes this as the `dynamic` feature (`npm run build -- --dynamic`) and tells the engine where the libraries are (`setBackendDir`, called by the loader).

Result on this machine:

- The addon compiled for generic x86-64 (0.8 MB) plus nine CPU variants (SSE4.2 through Ice Lake, about 1 MB each) and about 11 MB of shared libraries: **20 MB unpacked**.
- It selected the AVX2/FMA/BMI2 variant on its own (confirmed by `hardware().systemInfo`).
- Speed equals a static AVX2 build (pure CPU, 4.4 GB model):

| Build                                          | Prefill (tok/s) | Decode (tok/s) |
| ---------------------------------------------- | --------------- | -------------- |
| Static, AVX2 compiled in (`x86-64-v3`), 4.9 MB | 58 to 61        | 5.4 to 5.5     |
| Dynamic, variant chosen at runtime             | 58 to 61        | 5.1 to 5.2     |

Decode is memory-bandwidth bound, so it cannot differ; prefill is compute bound and showed no penalty.

**Decision for the beta: ship the static `x86-64-v3` build.** It is 4x smaller, a single file, and works on every CPU from about 2013 (Intel Haswell, AMD Excavator/Zen) onwards. A CPU without AVX2 gets a clear error instead of a crash (`hardware.rs`). The dynamic build adds support for older CPUs and AVX-512 machines but also adds per-platform risk that cannot be checked from Windows alone (finding the shared libraries on Linux and macOS, and shipping them in the package). It stays an opt-in feature and CI should build it on Linux before it is considered for a release.

## A caveat found while measuring

On a **CUDA build, `gpuLayers: 0` does not mean "no GPU"**: llama.cpp still sends large prompt batches to the GPU, so prompt processing ran at about 460 tokens/s versus about 60 on a pure CPU build, while generation (single tokens) stayed on the CPU at about 5 tokens/s. Use a CPU build when you need the GPU untouched.
