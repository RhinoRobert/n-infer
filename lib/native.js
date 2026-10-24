// Finds and loads the compiled addon.
//
// Order of preference:
//   1. lib/n-infer.node      built from source on this machine (`npm run build`)
//   2. @n-infer/<target>-cuda  optional GPU add-on, if installed and loadable
//   3. @n-infer/<target>       the prebuilt CPU binary for this platform
//
// Prebuilt binaries are published for Windows x64 only; the source builds elsewhere
// (see CONTRIBUTING.md). A GPU add-on that cannot load (for example, no CUDA runtime) falls back to the next choice with
// a warning instead of failing.
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { delimiter, dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const { version: packageVersion } = require('../package.json');

/** `process.platform-process.arch` -> published package suffix. */
const TARGETS = {
  'win32-x64': 'win32-x64-msvc',
};
/** Targets that also publish a CUDA add-on. */
const CUDA_TARGETS = new Set(['win32-x64-msvc']);

const key = `${process.platform}-${process.arch}`;
const target = TARGETS[key];

/**
 * Loads a binary package. Returns `{ addon, file }`, or throws with
 * `error.code === 'MODULE_NOT_FOUND'` when the package is not installed.
 */
function tryLoad(request) {
  const file = require.resolve(request);
  try {
    return { addon: require(file), file };
  } catch (error) {
    if (process.platform === 'win32' && error?.code !== 'MODULE_NOT_FOUND' && addCudaToPath()) {
      return { addon: require(file), file }; // the CUDA runtime was not on PATH: retry once with it
    }
    throw error;
  }
}

/**
 * Appends the CUDA toolkit's bin directories to PATH so Windows can find the CUDA runtime DLLs.
 * Only used after a load has already failed, only appends (system directories keep priority),
 * and only trusts an absolute, plain `CUDA_PATH`. Returns whether PATH changed.
 */
function addCudaToPath() {
  const cudaPath = process.env.CUDA_PATH;
  if (!cudaPath || !isAbsolute(cudaPath) || /[;\0]|\.\./.test(cudaPath)) return false;
  const existing = (process.env.PATH ?? '').split(delimiter).map((p) => p.toLowerCase());
  const missing = [join(cudaPath, 'bin', 'x64'), join(cudaPath, 'bin')].filter(
    (dir) => existsSync(dir) && !existing.includes(dir.toLowerCase()),
  );
  if (missing.length === 0) return false;
  process.env.PATH = [process.env.PATH ?? '', ...missing].join(delimiter);
  return true;
}

function loadFirstAvailable() {
  const localBuild = resolve(here, 'n-infer.node');
  if (existsSync(localBuild)) return { addon: require(localBuild), file: localBuild };

  if (!target) {
    throw new Error(
      `n-infer has no prebuilt binary for ${key} (prebuilt binaries are published for Windows x64 ` +
        'only). Build from source with `npm run build` (needs Rust, CMake and a C++ toolchain; ' +
        'see CONTRIBUTING.md).',
    );
  }

  if (CUDA_TARGETS.has(target)) {
    const name = `@n-infer/${target}-cuda`;
    try {
      return tryLoad(name);
    } catch (error) {
      if (error?.code !== 'MODULE_NOT_FOUND') {
        process.emitWarning(
          `${name} is installed but could not be loaded (${error.message.trim().split('\n')[0]}). ` +
            'It needs the NVIDIA driver and the CUDA runtime (cuBLAS) on this machine. ' +
            'Using the CPU build instead.',
          'N-InferWarning',
        );
      }
    }
  }

  const name = `@n-infer/${target}`;
  try {
    return tryLoad(name);
  } catch (error) {
    if (error?.code === 'MODULE_NOT_FOUND') {
      throw new Error(
        `n-infer could not find its binary package ${name}. Reinstall without \`--omit=optional\`, ` +
          'or build from source with `npm run build`.',
        { cause: error },
      );
    }
    throw new Error(`n-infer failed to load ${name}: ${error.message}`, { cause: error });
  }
}

const { addon, file } = loadFirstAvailable();

// The JavaScript and the binary are released together; a mismatch means a broken install.
if (addon.version() !== packageVersion) {
  throw new Error(
    `n-infer binary ${file} is version ${addon.version()} but the package is ${packageVersion}. ` +
      'Reinstall n-infer so both match.',
  );
}

// Builds that keep compute backends in separate libraries look for them next to the binary.
addon.setBackendDir(dirname(file));

export const native = addon;
