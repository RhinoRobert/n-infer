// Builds the native addon. Picks CUDA when nvcc is on PATH, otherwise CPU.
//   node scripts/build.js [--cuda | --cpu | --vulkan] [--debug]
import { spawnSync } from 'node:child_process';

const args = new Set(process.argv.slice(2));

function hasNvcc() {
  return spawnSync('nvcc', ['--version'], { stdio: 'ignore', shell: true }).status === 0;
}

let feature = null;
if (args.has('--cuda')) feature = 'cuda';
else if (args.has('--vulkan')) feature = 'vulkan';
else if (!args.has('--cpu') && hasNvcc()) feature = 'cuda';

const cliArgs = ['napi', 'build', '--no-js', '--dts', 'native.d.ts'];
if (!args.has('--debug')) cliArgs.push('--release');
if (feature) cliArgs.push('--features', feature);

console.log(`n-infer: building (${feature ?? 'cpu'}${args.has('--debug') ? ', debug' : ''})`);
const res = spawnSync('npx', cliArgs, { stdio: 'inherit', shell: true });
process.exit(res.status ?? 1);
