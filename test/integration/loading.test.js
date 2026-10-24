import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { closeSync, createReadStream, openSync, readSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { fixturePath, loadLibrary, skipUnlessModel } from '../helpers/fixtures.js';

const library = await loadLibrary();
const skip = library ? skipUnlessModel('tiny') : 'native binary not built: run `npm run build`';
const modelPath = fixturePath('tiny');
const load = (config) => new library.LocalModel({ modelPath, contextSize: 256, ...config }).init();

/** The first `length` bytes of a file (models can be far larger than a Node buffer allows). */
function readHead(path, length) {
  const file = openSync(path, 'r');
  try {
    const head = Buffer.alloc(length);
    readSync(file, head, 0, length, 0);
    return head;
  } finally {
    closeSync(file);
  }
}

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

describe('loading', { skip }, () => {
  test('rejects a file that is not a GGUF model', async () => {
    const notGguf = join(tmpdir(), `n-infer-not-gguf-${process.pid}.gguf`);
    writeFileSync(notGguf, 'x'.repeat(256));
    await assert.rejects(new library.LocalModel({ modelPath: notGguf }).init(), /GGUF/);
  });

  test('rejects a truncated GGUF file', async () => {
    const truncated = join(tmpdir(), `n-infer-truncated-${process.pid}.gguf`);
    writeFileSync(truncated, readHead(modelPath, 200));
    await assert.rejects(new library.LocalModel({ modelPath: truncated }).init());
  });

  test('accepts a matching modelSha256 and rejects a different one', async () => {
    const actual = await sha256(modelPath);
    const model = await load({ modelSha256: actual.toUpperCase() });
    model.close();
    await assert.rejects(load({ modelSha256: '0'.repeat(64) }), /does not match modelSha256/);
    await assert.rejects(load({ modelSha256: 'not-a-hash' }), /modelSha256/);
  });

  test('rejects out-of-range configuration with a clear message', async () => {
    await assert.rejects(load({ contextSize: 8 }), /contextSize/);
    await assert.rejects(load({ maxConcurrent: 0 }), /maxConcurrent/);
    await assert.rejects(load({ maxConcurrent: 999 }), /maxConcurrent/);
    await assert.rejects(load({ threads: 0 }), /threads/);
    await assert.rejects(load({ gpuLayers: 100000 }), /gpuLayers/);
  });

  test('gpuLayers: "auto" and explicit 0 both load', async () => {
    for (const gpuLayers of ['auto', 0]) {
      const model = await load({ gpuLayers });
      assert.equal(model.ready, true);
      model.close();
    }
  });

  test('several workers each get their own context', async () => {
    const model = await load({ maxConcurrent: 2 });
    const [a, b] = await Promise.all([
      model.generate('Once upon', { temperature: 0, maxTokens: 6 }),
      model.generate('Once upon', { temperature: 0, maxTokens: 6 }),
    ]);
    assert.equal(a, b);
    model.close();
  });
});
