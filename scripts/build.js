// Builds the native addon into lib/. Uses CUDA when nvcc is on PATH, otherwise CPU.
//   node scripts/build.js [--cuda | --vulkan | --cpu] [--debug]
import { spawnSync } from 'node:child_process';

const flags = new Set(process.argv.slice(2));

const hasNvcc = () => spawnSync('nvcc', ['--version'], { stdio: 'ignore', shell: true }).status === 0;

const feature = flags.has('--cuda')
  ? 'cuda'
  : flags.has('--vulkan')
    ? 'vulkan'
    : !flags.has('--cpu') && hasNvcc()
      ? 'cuda'
      : null;

// `lib` is napi's output directory; napi-generated.d.ts is a generated file we do not use.
const args = ['napi', 'build', '--js', 'false', '--dts', 'napi-generated.d.ts'];
if (!flags.has('--debug')) args.push('--release');
if (feature) args.push('--features', feature);
args.push('lib');

console.log(`n-infer: building (${feature ?? 'cpu'}${flags.has('--debug') ? ', debug' : ''})`);
process.exit(spawnSync('npx', args, { stdio: 'inherit', shell: true }).status ?? 1);
