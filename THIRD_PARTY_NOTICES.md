# Third-party notices

n-infer is MIT licensed (see `LICENSE`). Its native binaries include or link the following. Each is used under its own license.

| Component                                                                  | License                              | Role                                         |
| -------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml                  | MIT                                  | Inference engine                             |
| [llama-cpp-2 / llama-cpp-sys-2](https://github.com/utilityai/llama-cpp-rs) | MIT OR Apache-2.0                    | Rust bindings to llama.cpp                   |
| [napi-rs](https://github.com/napi-rs/napi-rs)                              | MIT                                  | Node-API bindings for Rust                   |
| [encoding_rs](https://github.com/hsivonen/encoding_rs)                     | (Apache-2.0 OR MIT) AND BSD-3-Clause | Streaming UTF-8 decoding                     |
| [num_cpus](https://github.com/seanmonstar/num_cpus)                        | MIT OR Apache-2.0                    | Physical core detection                      |
| [sha2](https://github.com/RustCrypto/hashes)                               | MIT OR Apache-2.0                    | `modelSha256` verification                   |
| NVIDIA CUDA runtime (CUDA add-on packages only)                            | NVIDIA EULA                          | GPU acceleration; redistribution terms apply |
| OpenMP runtime (where the platform toolchain links it)                     | per toolchain                        | CPU threading                                |

The full list of Rust dependencies and their licenses is checked in CI with `cargo deny` (see `deny.toml`). Model files are not part of this package and carry their own licenses.
