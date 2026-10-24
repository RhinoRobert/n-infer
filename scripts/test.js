// Runs the JavaScript tests with Node's built-in runner, listing files explicitly so it behaves
// the same on every supported Node version and shell.
//
//   node scripts/test.js [unit] [integration]      (default: both)
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'test');
const suites = process.argv.slice(2);
const wanted = suites.length > 0 ? suites : ['unit', 'integration'];

const files = wanted.flatMap((suite) => {
  const dir = join(testDir, suite);
  try {
    return readdirSync(dir)
      .filter((name) => name.endsWith('.test.js'))
      .map((name) => join(dir, name));
  } catch {
    console.error(`unknown test suite "${suite}" (choose from: unit, integration)`);
    process.exit(1);
  }
});

// One file at a time: each integration file loads a model, and parallel loads fight over RAM/VRAM.
const result = spawnSync(process.execPath, ['--test', '--test-concurrency=1', ...files], {
  stdio: 'inherit',
});
process.exit(result.status ?? 1);
