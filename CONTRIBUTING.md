# Contributing

Thanks for helping. n-infer values **small, readable code** over cleverness: a new contributor should be able to follow a request from `lib/` to llama.cpp in one sitting. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the map.

## Set up

You need Node.js 20.19+, Rust (stable), CMake, and a C++ toolchain (Visual Studio Build Tools on Windows). llama.cpp's bindings also need libclang (`LIBCLANG_PATH` on Windows). For GPU builds add the CUDA Toolkit.

```bash
npm install
npm run build:cpu        # or `npm run build` (CUDA if nvcc is found)
npm run test:fetch       # downloads two small, hash-checked test models
npm test
```

`npm run build` is tuned for _your_ machine (native CPU and GPU architecture) and is not portable; releases use `build:portable`.

## Before you open a pull request

```bash
npm run lint             # rustfmt, clippy, prettier, TypeScript types
npm run test:rust
npm test
```

- Add or update tests for behaviour changes. Unit tests need no model; integration tests use the models from `npm run test:fetch`.
- Keep modules focused (the largest source files are about 150 lines of code) and comment _why_, not _what_.
- Validation ranges live in Rust (`src/options.rs`); the JavaScript wrapper only checks types and copies own properties.
- No runtime dependencies in the JavaScript package. Dev dependencies should stay minimal.
- Anything that changes sampling, caching or tokenization should say how it was measured or verified.

## Reporting bugs

Open an issue with the output of:

```bash
node -e "import('n-infer').then(m => console.log(m.backend, JSON.stringify(m.hardware(), null, 2)))"
```

Security problems go through [SECURITY.md](SECURITY.md), not public issues.
