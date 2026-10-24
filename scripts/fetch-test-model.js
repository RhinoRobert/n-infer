// Downloads the small GGUF models the integration tests use into test/fixtures/, verifying each
// against a pinned SHA-256. Nothing is run or trusted until the hash matches.
//
//   node scripts/fetch-test-model.js [tiny] [chat]      (default: both)
import { createHash } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, renameSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';

const MODELS = {
  tiny: {
    file: 'stories260K.gguf',
    url: 'https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories260K.gguf',
    sha256: '047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b',
    bytes: 1_185_376,
  },
  chat: {
    file: 'SmolLM2-135M-Instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf',
    sha256: '2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d',
    bytes: 105_454_432,
  },
};

const fixturesDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'test', 'fixtures');
const requested = process.argv.slice(2);
const kinds = requested.length > 0 ? requested : Object.keys(MODELS);

mkdirSync(fixturesDir, { recursive: true });
for (const kind of kinds) {
  if (!(kind in MODELS)) {
    console.error(`unknown model "${kind}" (choose from: ${Object.keys(MODELS).join(', ')})`);
    process.exit(1);
  }
  await fetchModel(MODELS[kind]);
}

async function fetchModel({ file, url, sha256, bytes }) {
  const target = join(fixturesDir, file);
  if (existsSync(target)) {
    console.log(`${file}: already present`);
    return;
  }
  console.log(`${file}: downloading ${(bytes / 1e6).toFixed(1)} MB from ${new URL(url).host}`);

  const response = await fetch(url);
  if (!response.ok) throw new Error(`download failed: HTTP ${response.status}`);

  const partial = `${target}.partial`;
  const hash = createHash('sha256');
  try {
    const body = Readable.fromWeb(response.body);
    body.on('data', (chunk) => hash.update(chunk));
    await pipeline(body, createWriteStream(partial));
    const actual = hash.digest('hex');
    if (actual !== sha256) throw new Error(`checksum mismatch: expected ${sha256}, got ${actual}`);
    renameSync(partial, target);
    console.log(`${file}: verified and saved`);
  } catch (error) {
    rmSync(partial, { force: true });
    throw error;
  }
}
