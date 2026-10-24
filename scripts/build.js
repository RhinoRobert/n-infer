// Builds the native addon into lib/.
//
//   node scripts/build.js [--cuda | --vulkan | --cpu] [--native | --portable | --baseline]
//                         [--dynamic] [--debug]
//
// Accelerator: CUDA is used when nvcc is found, unless --cpu / --vulkan says otherwise.
//
// Tuning (default --native):
//   --native    compile for the CPU (and CUDA GPU) of this machine: fastest, not portable
//   --portable  x86-64-v3 CPU (AVX2/FMA, ~2013+) and a multi-generation CUDA build: for distribution
//   --baseline  generic x86-64: runs anywhere, slowest on the CPU
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const ALLOWED_FLAGS = new Set([
  '--cuda',
  '--vulkan',
  '--cpu',
  '--native',
  '--portable',
  '--baseline',
  '--dynamic',
  '--debug',
]);
const flags = new Set(process.argv.slice(2));

for (const flag of flags) {
  if (!ALLOWED_FLAGS.has(flag)) {
    console.error(`n-infer: unknown build flag "${flag}"`);
    process.exit(1);
  }
}

function hasNvcc() {
  if (spawnSync('nvcc', ['--version'], { stdio: 'ignore', shell: false }).status === 0) {
    return true;
  }
  const cudaPath = process.env.CUDA_PATH;
  if (cudaPath) {
    const exe = process.platform === 'win32' ? 'nvcc.exe' : 'nvcc';
    return existsSync(resolve(cudaPath, 'bin', exe));
  }
  return false;
}

const feature = flags.has('--cuda')
  ? 'cuda'
  : flags.has('--vulkan')
    ? 'vulkan'
    : !flags.has('--cpu') && hasNvcc()
      ? 'cuda'
      : null;

const tuning = flags.has('--portable')
  ? 'portable'
  : flags.has('--baseline')
    ? 'baseline'
    : 'native';

// llama.cpp's build reads `target-cpu` from RUSTFLAGS: `native` also selects this machine's CUDA
// architecture; anything else gets a portable multi-architecture CUDA build.
const cpuFlag = { native: 'native', portable: 'x86-64-v3', baseline: null }[tuning];
const useCpuFlag = cpuFlag && !(tuning === 'portable' && process.arch !== 'x64');
const env = { ...process.env };
if (useCpuFlag) env.RUSTFLAGS = `${env.RUSTFLAGS ?? ''} -C target-cpu=${cpuFlag}`.trim();

// MSBuild (used by CMake on Windows) otherwise leaves idle worker processes running after a build.
env.MSBUILDDISABLENODEREUSE = '1';

// Visual Studio's CUDA integration looks the toolkit up through CUDA_PATH_V<major>_<minor>. The
// CUDA installer sets it for new shells, but an older shell only has CUDA_PATH.
const cudaVersion = /[\\/]v(\d+)\.(\d+)$/i.exec(env.CUDA_PATH ?? '');
if (feature === 'cuda' && cudaVersion) {
  env[`CUDA_PATH_V${cudaVersion[1]}_${cudaVersion[2]}`] ??= env.CUDA_PATH;
}

const require = createRequire(import.meta.url);
const napiCli = require.resolve('@napi-rs/cli/scripts/index.js');

// `lib` is napi's output directory; napi-generated.d.ts is a generated file we do not use.
const args = [napiCli, 'build', '--js', 'false', '--dts', 'napi-generated.d.ts'];
if (!flags.has('--debug')) args.push('--release');
const features = [feature, flags.has('--dynamic') && 'dynamic'].filter(Boolean);
if (features.length > 0) args.push('--features', features.join(','));
args.push('lib');

console.log(
  `n-infer: building (${features.join('+') || 'cpu'}, ${tuning}${flags.has('--debug') ? ', debug' : ''})`,
);
const result = spawnSync(process.execPath, args, { stdio: 'inherit', shell: false, env });
process.exit(result.status ?? 1);
