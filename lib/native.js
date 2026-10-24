// Locates and loads the compiled addon that `npm run build` drops next to this file.
import { createRequire } from 'node:module';
import { lstatSync, statSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Explicit allowlist of canonical napi-rs binary names; rejects symlinks and unexpected filenames.
const CANDIDATES = [
  `n-infer.${process.platform}-${process.arch}-msvc.node`,
  `n-infer.${process.platform}-${process.arch}-gnu.node`,
  `n-infer.${process.platform}-${process.arch}.node`,
  'n-infer.node',
];

function isRegularFile(filePath) {
  try {
    const st = lstatSync(filePath);
    return st.isFile() && !st.isSymbolicLink();
  } catch {
    return false;
  }
}

function isDirectory(dirPath) {
  try {
    return statSync(dirPath).isDirectory();
  } catch {
    return false;
  }
}

// On Windows, CUDA builds link cudart/cublas dynamically. Append (never prepend)
// validated %CUDA_PATH%\bin\x64 and %CUDA_PATH%\bin directories to PATH so system
// DLL search order is preserved while CUDA runtime DLLs resolve automatically.
function appendWindowsCudaPaths() {
  if (process.platform !== 'win32') return;
  const cudaPath = process.env.CUDA_PATH;
  if (
    typeof cudaPath !== 'string' ||
    !cudaPath ||
    cudaPath.includes(';') ||
    cudaPath.includes('\0') ||
    cudaPath.includes('..') ||
    !isAbsolute(cudaPath)
  ) {
    return;
  }

  const currentEntries = (process.env.PATH ?? '').split(';').map((p) => p.trim().toLowerCase());
  const toAppend = [];
  for (const sub of [resolve(cudaPath, 'bin', 'x64'), resolve(cudaPath, 'bin')]) {
    if (isDirectory(sub) && !currentEntries.includes(sub.toLowerCase())) {
      toAppend.push(sub);
    }
  }
  if (toAppend.length > 0) {
    process.env.PATH = `${process.env.PATH ?? ''};${toAppend.join(';')}`;
  }
}

function findBinary() {
  for (const name of CANDIDATES) {
    const full = resolve(here, name);
    if (isRegularFile(full)) return full;
  }
  return null;
}

const binaryPath = findBinary();
if (!binaryPath) {
  throw new Error(`n-infer: no native binary found in ${here}. Run \`npm run build\` first.`);
}

function load() {
  appendWindowsCudaPaths();
  try {
    return require(binaryPath);
  } catch (cause) {
    throw new Error(
      `n-infer: failed to load ${binaryPath}. If this is a CUDA build, ensure the CUDA runtime ` +
        'is installed (%CUDA_PATH%\\bin\\x64 and %CUDA_PATH%\\bin), or rebuild CPU-only with `npm run build:cpu`.',
      { cause },
    );
  }
}

export const native = load();
