// Locates and loads the compiled addon that `npm run build` drops next to this file.
import { createRequire } from 'node:module';
import { readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

function findBinary() {
  const platformPrefix = `n-infer.${process.platform}-${process.arch}`;
  const binaries = readdirSync(here).filter((f) => f.startsWith('n-infer') && f.endsWith('.node'));
  return binaries.find((f) => f.startsWith(platformPrefix)) ?? binaries.find((f) => f === 'n-infer.node');
}

const file = findBinary();
if (!file) {
  throw new Error(`n-infer: no native binary found in ${here}. Run \`npm run build\` first.`);
}

export const native = require(resolve(here, file));
