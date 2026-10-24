// Stages every npm package for a release from the built binaries, ready for `npm publish`.
//
//   node scripts/stage-npm.js <artifacts-dir>
//
// <artifacts-dir> holds one sub-folder per target, each containing the `n-infer.node` that
// `npm run build:portable` produced for it (CI downloads these from the build matrix):
//
//   artifacts/win32-x64-msvc/n-infer.node        -> @n-infer/win32-x64-msvc
//   artifacts/win32-x64-msvc-cuda/n-infer.node   -> @n-infer/win32-x64-msvc-cuda
//
// Output goes to dist/npm/, one complete package per folder at the root version:
//
//   dist/npm/<target>/   a binary package
//   dist/npm/n-infer/    the main package, whose manifest lists exactly the binary packages that
//                        were staged (so a target that failed to build is simply not referenced)
//
// The repository's own package.json deliberately has no optionalDependencies: they point at
// packages that only exist once a release has been published, which would break `npm ci`.
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
const rootManifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const { version, license, repository, author } = rootManifest;

/** Every binary package we publish, and how npm should match it to a machine. */
const TARGETS = {
  'win32-x64-msvc': { os: 'win32', cpu: 'x64' },
  'win32-x64-msvc-cuda': { os: 'win32', cpu: 'x64', gpu: 'NVIDIA CUDA' },
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
  writeJson(join(packageDir, 'package.json'), binaryManifest(target, spec));
  writeFileSync(join(packageDir, 'README.md'), binaryReadme(target, spec));
  staged.push(target);
}
if (!staged.includes('win32-x64-msvc')) {
  throw new Error(
    'the CPU package (win32-x64-msvc) is required: refusing to stage a release without it',
  );
}
stageMainPackage(staged);

console.log(
  `staged ${staged.length + 1} package(s) at version ${version} in ${outDir}:\n  n-infer\n  ${staged.join('\n  ')}`,
);

function stageMainPackage(targets) {
  const dir = join(outDir, 'n-infer');
  mkdirSync(join(dir, 'lib'), { recursive: true });
  for (const file of readdirSync(join(root, 'lib'))) {
    if (file.endsWith('.js') || file === 'index.d.ts') {
      copyFileSync(join(root, 'lib', file), join(dir, 'lib', file));
    }
  }
  copyFileSync(join(root, 'LICENSE'), join(dir, 'LICENSE'));
  copyFileSync(join(root, 'README.md'), join(dir, 'README.md'));

  const manifest = { ...rootManifest };
  delete manifest.scripts; // build tooling is not part of the published package
  delete manifest.devDependencies;
  manifest.optionalDependencies = Object.fromEntries(
    targets.filter((t) => !TARGETS[t].gpu).map((t) => [`@n-infer/${t}`, version]),
  );
  const gpuTargets = targets.filter((t) => TARGETS[t].gpu);
  if (gpuTargets.length > 0) {
    manifest.peerDependencies = Object.fromEntries(
      gpuTargets.map((t) => [`@n-infer/${t}`, version]),
    );
    manifest.peerDependenciesMeta = Object.fromEntries(
      gpuTargets.map((t) => [`@n-infer/${t}`, { optional: true }]),
    );
  }
  writeJson(join(dir, 'package.json'), manifest);
}

function binaryManifest(target, spec) {
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
    engines: { node: '>=20.19' },
    publishConfig: { access: 'public', provenance: true },
  };
}

function binaryReadme(target, spec) {
  const accel = spec.gpu ? ` with ${spec.gpu} acceleration` : '';
  return `# @n-infer/${target}\n\nPrebuilt native binary${accel} for [n-infer](https://github.com/RhinoRobert/n-infer) on ${spec.os} ${spec.cpu}.\nYou do not install this directly: \`npm install n-infer\` selects the right one.\n`;
}

function writeJson(path, value) {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}
