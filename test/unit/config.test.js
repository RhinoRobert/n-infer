import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { normalizeConfig } from '../../lib/config.js';

const modelFile = join(tmpdir(), `n-infer-config-test-${process.pid}.gguf`);
writeFileSync(modelFile, 'placeholder');

test('accepts a minimal config and freezes the result', () => {
  const config = normalizeConfig({ modelPath: modelFile });
  assert.equal(config.modelPath, modelFile);
  assert.ok(Object.isFrozen(config));
});

test("maps gpuLayers: 'auto' to the native gpuAuto flag", () => {
  const config = normalizeConfig({ modelPath: modelFile, gpuLayers: 'auto' });
  assert.equal(config.gpuAuto, true);
  assert.equal(config.gpuLayers, undefined);
});

test('passes integer tuning options through', () => {
  const config = normalizeConfig({
    modelPath: modelFile,
    gpuLayers: 20,
    threads: 8,
    maxConcurrent: 2,
  });
  assert.deepEqual(
    { ...config },
    { modelPath: modelFile, gpuLayers: 20, threads: 8, maxConcurrent: 2 },
  );
});

test('rejects a missing, empty or non-string modelPath', () => {
  assert.throws(() => normalizeConfig({}), /modelPath/);
  assert.throws(() => normalizeConfig({ modelPath: '  ' }), /modelPath/);
  assert.throws(() => normalizeConfig({ modelPath: 42 }), /modelPath/);
});

test('rejects a model path that does not exist or is a directory', () => {
  assert.throws(() => normalizeConfig({ modelPath: join(tmpdir(), 'nope.gguf') }), /not found/);
  assert.throws(() => normalizeConfig({ modelPath: tmpdir() }), /regular file/);
});

test('rejects unknown keys so typos are not silently ignored', () => {
  assert.throws(
    () => normalizeConfig({ modelPath: modelFile, gpulayers: 3 }),
    /unknown config field `gpulayers`/,
  );
});

test('rejects wrong types', () => {
  assert.throws(() => normalizeConfig({ modelPath: modelFile, threads: 1.5 }), /threads/);
  assert.throws(() => normalizeConfig({ modelPath: modelFile, threads: '4' }), /threads/);
  assert.throws(() => normalizeConfig({ modelPath: modelFile, verbose: 1 }), /verbose/);
  assert.throws(() => normalizeConfig({ modelPath: modelFile, modelSha256: 7 }), /modelSha256/);
  assert.throws(() => normalizeConfig(null), /object/);
  assert.throws(() => normalizeConfig([modelFile]), /object/);
});

test('ignores properties inherited from a polluted Object.prototype', () => {
  Object.prototype.threads = 999;
  try {
    const config = normalizeConfig({ modelPath: modelFile });
    assert.equal(config.threads, undefined);
  } finally {
    delete Object.prototype.threads;
  }
});
