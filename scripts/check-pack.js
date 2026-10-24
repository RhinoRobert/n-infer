// Fails if `npm pack` would publish anything unexpected (a stray binary, model, key or test file).
//
//   node scripts/check-pack.js [directory]      (default: the repo root)
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const directory = resolve(process.argv[2] ?? root);

/** Paths a published package may contain (the main package; platform packages are staged separately). */
const ALLOWED = [
  /^package\.json$/,
  /^README\.md$/,
  /^LICENSE$/,
  /^lib\/[a-z-]+\.js$/,
  /^lib\/index\.d\.ts$/,
  /^n-infer\.node$/, // platform packages only
];

// Run npm through node when we can (no shell needed); `npm_execpath` is set when launched by `npm run`.
const npmCli = process.env.npm_execpath;
const [command, args] = npmCli
  ? [process.execPath, [npmCli, 'pack', '--dry-run', '--json']]
  : ['npm', ['pack', '--dry-run', '--json']];
const result = spawnSync(command, args, {
  cwd: directory,
  encoding: 'utf8',
  shell: !npmCli && process.platform === 'win32',
});
if (result.status !== 0) {
  console.error(result.stderr || result.stdout);
  process.exit(1);
}

const [{ name, version, files, size, unpackedSize }] = JSON.parse(result.stdout);
const unexpected = files
  .map((f) => f.path)
  .filter((path) => !ALLOWED.some((rule) => rule.test(path)));

console.log(
  `${name}@${version}: ${files.length} files, ${(size / 1024).toFixed(0)} kB packed, ${(unpackedSize / 1024).toFixed(0)} kB unpacked`,
);
if (unexpected.length > 0) {
  console.error(`unexpected files would be published:\n  ${unexpected.join('\n  ')}`);
  process.exit(1);
}
console.log('package contents OK');
