// Finds and loads the compiled addon.
//
// Order of preference:
//   1. lib/n-infer.node      built from source on this machine (`npm run build`)
//   2. @n-infer/<target>-cuda  optional GPU add-on, if installed and loadable
//   3. @n-infer/<target>       the prebuilt binary for this platform (CPU, or Metal on macOS)
//
// A GPU add-on that cannot load (for example, no CUDA runtime) falls back to the next choice with
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
  'linux-x64': 'linux-x64-gnu',
  'linux-arm64': 'linux-arm64-gnu',
  'darwin-arm64': 'darwin-arm64',
};
/** Targets that also publish a CUDA add-on. */
const CUDA_TARGETS = new Set(['win32-x64-msvc', 'linux-x64-gnu']);

const key = `${process.platform}-${process.arch}`;
const target = TARGETS[key];

/** Returns the addon's exports, or throws with `error.code === 'MODULE_NOT_FOUND'` when absent. */
function tryLoad(request) {
  try {
    return require(request);
  } catch (error) {
    if (process.platform === 'win32' && error?.code !== 'MODULE_NOT_FOUND' && addCudaToPath()) {
      return require(request); // the CUDA runtime was not on PATH: retry once with it
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
  if (existsSync(localBuild)) return { addon: require(localBuild), source: localBuild };

  if (!target) {
    throw new Error(
      `n-infer has no prebuilt binary for ${key}. Build from source with \`npm run build\` ` +
        '(needs Rust, CMake and a C++ toolchain; see the README).',
    );
  }

  if (CUDA_TARGETS.has(target)) {
    const name = `@n-infer/${target}-cuda`;
    try {
      return { addon: tryLoad(name), source: name };
    } catch (error) {
      if (error?.code !== 'MODULE_NOT_FOUND') {
        process.emitWarning(
          `${name} is installed but could not be loaded (${error.message.split('\n')[0]}). ` +
            'Using the CPU build instead.',
          'N-InferWarning',
        );
      }
    }
  }

  const name = `@n-infer/${target}`;
  try {
    return { addon: tryLoad(name), source: name };
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

const { addon, source } = loadFirstAvailable();

// The JavaScript and the binary are released together; a mismatch means a broken install.
if (addon.version() !== packageVersion) {
  throw new Error(
    `n-infer binary ${source} is version ${addon.version()} but the package is ${packageVersion}. ` +
      'Reinstall n-infer so both match.',
  );
}

export const native = addon;
