// Stages the per-platform npm packages from built binaries, ready for `npm publish`.
//
//   node scripts/stage-npm.js <artifacts-dir>
//
// <artifacts-dir> holds one sub-folder per target, each containing the `n-infer.node` that
// `npm run build:portable` produced for it (CI downloads these from the build matrix):
//
//   artifacts/win32-x64-msvc/n-infer.node        -> @n-infer/win32-x64-msvc
//   artifacts/win32-x64-msvc-cuda/n-infer.node   -> @n-infer/win32-x64-msvc-cuda
//   ...
//
// Output goes to dist/npm/<target>/, each folder a complete package at the root version.
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const { version, license, repository, author } = JSON.parse(
  readFileSync(join(root, 'package.json'), 'utf8'),
);

/** Every package we publish: how npm should match it to a machine. */
const TARGETS = {
  'win32-x64-msvc': { os: 'win32', cpu: 'x64' },
  'win32-x64-msvc-cuda': { os: 'win32', cpu: 'x64', gpu: 'NVIDIA CUDA' },
  'linux-x64-gnu': { os: 'linux', cpu: 'x64', libc: 'glibc' },
  'linux-x64-gnu-cuda': { os: 'linux', cpu: 'x64', libc: 'glibc', gpu: 'NVIDIA CUDA' },
  'linux-arm64-gnu': { os: 'linux', cpu: 'arm64', libc: 'glibc' },
  'darwin-arm64': { os: 'darwin', cpu: 'arm64', gpu: 'Apple Metal' },
};

const artifactsDir = process.argv[2];
if (!artifactsDir || !existsSync(artifactsDir)) {
  console.error('usage: node scripts/stage-npm.js <artifacts-dir>');
  process.exit(1);
}

const outDir = join(root, 'dist', 'npm');
rmSync(outDir, { recursive: true, force: true });

const staged = [];
for (const target of readdirSync(artifactsDir)) {
  const spec = TARGETS[target];
  if (!spec) throw new Error(`unexpected artifact folder "${target}"`);
  const binary = join(artifactsDir, target, 'n-infer.node');
  if (!existsSync(binary)) throw new Error(`missing ${binary}`);

  const packageDir = join(outDir, target);
  mkdirSync(packageDir, { recursive: true });
  copyFileSync(binary, join(packageDir, 'n-infer.node'));
  copyFileSync(join(root, 'LICENSE'), join(packageDir, 'LICENSE'));
  writeFileSync(
    join(packageDir, 'package.json'),
    `${JSON.stringify(packageJson(target, spec), null, 2)}\n`,
  );
  writeFileSync(join(packageDir, 'README.md'), readme(target, spec));
  staged.push(target);
}
console.log(
  `staged ${staged.length} package(s) at version ${version} in ${outDir}:\n  ${staged.join('\n  ')}`,
);

function packageJson(target, spec) {
  return {
    name: `@n-infer/${target}`,
    version,
    description: `${spec.gpu ? `${spec.gpu} ` : ''}native binary for n-infer on ${spec.os}-${spec.cpu}`,
    license,
    author,
    repository,
    main: 'n-infer.node',
    files: ['n-infer.node', 'README.md', 'LICENSE'],
    os: [spec.os],
    cpu: [spec.cpu],
    ...(spec.libc && { libc: [spec.libc] }),
    engines: { node: '>=20.19' },
    publishConfig: { access: 'public', provenance: true },
  };
}

function readme(target, spec) {
  const accel = spec.gpu ? ` with ${spec.gpu} acceleration` : '';
  return `# @n-infer/${target}\n\nPrebuilt native binary${accel} for [n-infer](https://github.com/RhinoRobert/n-infer) on ${spec.os} ${spec.cpu}.\nYou do not install this directly: \`npm install n-infer\` selects the right one.\n`;
}
