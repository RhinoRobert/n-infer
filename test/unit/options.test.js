import assert from 'node:assert/strict';
import { test } from 'node:test';
import { normalizeInput, normalizeOptions } from '../../lib/options.js';

test('no options means empty native options', () => {
  const { native, signal, parseSpecial } = normalizeOptions(undefined);
  assert.deepEqual({ ...native }, {});
  assert.equal(signal, undefined);
  assert.equal(parseSpecial, false);
});

test('passes recognised options to the native side', () => {
  const { native } = normalizeOptions({
    maxTokens: 32,
    temperature: 0.5,
    topP: 0.9,
    topK: 20,
    repetitionPenalty: 1.2,
    seed: 7,
    cache: false,
    stop: 'END',
  });
  assert.deepEqual(
    { ...native },
    {
      maxTokens: 32,
      temperature: 0.5,
      topP: 0.9,
      topK: 20,
      repetitionPenalty: 1.2,
      seed: 7,
      cache: false,
      stop: ['END'],
    },
  );
});

test('keeps parseSpecial and signal out of the native options', () => {
  const controller = new AbortController();
  const { native, signal, parseSpecial } = normalizeOptions({
    parseSpecial: true,
    signal: controller.signal,
  });
  assert.equal('parseSpecial' in native, false);
  assert.equal('signal' in native, false);
  assert.equal(signal, controller.signal);
  assert.equal(parseSpecial, true);
});

test('rejects unknown options and wrong types', () => {
  assert.throws(() => normalizeOptions({ temprature: 1 }), /unknown options field `temprature`/);
  assert.throws(() => normalizeOptions({ maxTokens: 1.5 }), /maxTokens/);
  assert.throws(() => normalizeOptions({ temperature: '1' }), /temperature/);
  assert.throws(() => normalizeOptions({ cache: 'yes' }), /cache/);
  assert.throws(() => normalizeOptions({ stop: [1] }), /stop/);
  assert.throws(() => normalizeOptions({ signal: {} }), /AbortSignal/);
  assert.throws(() => normalizeOptions('fast'), /object/);
});

test('NaN is passed through for Rust to reject (types only are checked here)', () => {
  assert.doesNotThrow(() => normalizeOptions({ temperature: Number.NaN }));
});

test('ignores inherited properties', () => {
  Object.prototype.maxTokens = 5;
  try {
    assert.equal(normalizeOptions({}).native.maxTokens, undefined);
  } finally {
    delete Object.prototype.maxTokens;
  }
});

test('normalizeInput accepts text and chat messages', () => {
  assert.deepEqual(normalizeInput('hello'), { text: 'hello' });
  assert.deepEqual(normalizeInput([{ role: 'user', content: 'hi' }]), {
    messages: [{ role: 'user', content: 'hi' }],
  });
});

test('normalizeInput rejects empty text and malformed messages', () => {
  assert.throws(() => normalizeInput(''), /empty/);
  assert.throws(() => normalizeInput(42), /string or an array/);
  assert.throws(() => normalizeInput([{ role: 'user' }]), /role.*content/);
  assert.throws(
    () => normalizeInput([{ role: 'user', content: 'x', extra: 1 }]),
    /unknown message field/,
  );
  assert.throws(() => normalizeInput(['not an object']), /object/);
});
