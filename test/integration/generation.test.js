import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import { GREEDY, loadLibrary, openModel, skipUnlessModel } from '../helpers/fixtures.js';

const library = await loadLibrary();
const skip = library ? skipUnlessModel('tiny') : 'native binary not built: run `npm run build`';
const config = { contextSize: 512 };

describe('generation', { skip }, () => {
  test('greedy decoding is deterministic', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const first = await model.generate('Once upon a time', GREEDY);
    const second = await model.generate('Once upon a time', GREEDY);
    assert.ok(first.length > 0);
    assert.equal(second, first);
  });

  test('a reused cache prefix gives the same text as a cold cache', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const shared = 'The little dog ran to the park and saw a big red ball. ';
    const cold = await model.generate(`${shared}Then`, { ...GREEDY, cache: false });
    await model.generate(`${shared}Afterwards`, GREEDY); // warms the shared prefix
    const warm = await model.generate(`${shared}Then`, GREEDY);
    assert.equal(warm, cold);
  });

  test('stops at a stop sequence without emitting it', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const full = await model.generate('Once upon a time', { ...GREEDY, maxTokens: 40 });
    const stop = full.slice(12, 16);
    const expected = full.slice(0, full.indexOf(stop));
    assert.equal(
      await model.generate('Once upon a time', { ...GREEDY, maxTokens: 40, stop }),
      expected,
    );
  });

  test('maxTokens limits the output and 0 produces nothing', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    assert.equal(await model.generate('Once upon', { ...GREEDY, maxTokens: 0 }), '');
    const short = await model.generate('Once upon', { ...GREEDY, maxTokens: 2 });
    const long = await model.generate('Once upon', { ...GREEDY, maxTokens: 20 });
    assert.ok(short.length < long.length);
  });

  test('stream yields the same text generate returns', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    let streamed = '';
    for await (const piece of model.stream('Once upon a time', GREEDY)) streamed += piece;
    assert.equal(streamed, await model.generate('Once upon a time', GREEDY));
  });

  test('queued requests give the same answers as running one at a time', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const prompts = ['Once upon', 'The dog', 'A big red', 'In the park'];
    const alone = [];
    for (const prompt of prompts) alone.push(await model.generate(prompt, GREEDY));
    const together = await Promise.all(prompts.map((prompt) => model.generate(prompt, GREEDY)));
    assert.deepEqual(together, alone);
  });

  test('a full queue rejects new requests instead of piling up', async (t) => {
    const model = await openModel(t, library, 'tiny', { ...config, maxConcurrent: 1, maxQueue: 1 });
    const results = await Promise.allSettled(
      Array.from({ length: 12 }, (_, i) =>
        model.generate(`Story ${i}`, { ...GREEDY, maxTokens: 8 }),
      ),
    );
    const rejected = results.filter((r) => r.status === 'rejected');
    assert.ok(
      results.some((r) => r.status === 'fulfilled'),
      'some requests complete',
    );
    assert.ok(rejected.length > 0, 'some requests are turned away');
    assert.match(rejected[0].reason.message, /too many pending/);
  });

  test('aborting a stream rejects with AbortError and the model stays usable', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const controller = new AbortController();
    let pieces = 0;
    await assert.rejects(
      async () => {
        for await (const _ of model.stream('Once upon a time', {
          maxTokens: 200,
          signal: controller.signal,
        })) {
          if (++pieces === 2) controller.abort();
        }
      },
      { name: 'AbortError' },
    );
    assert.ok((await model.generate('Once upon', GREEDY)).length > 0);
  });

  test('aborting generate rejects with AbortError', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const controller = new AbortController();
    const pending = model.generate('Once upon a time', {
      maxTokens: 400,
      signal: controller.signal,
    });
    controller.abort();
    await assert.rejects(pending, { name: 'AbortError' });
  });

  test('bad sampling options are rejected before generating', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    for (const [options, pattern] of [
      [{ temperature: Number.NaN }, /temperature/],
      [{ temperature: -1 }, /temperature/],
      [{ topP: 0 }, /topP/],
      [{ repetitionPenalty: 0 }, /repetitionPenalty/],
      [{ maxTokens: 1e9 }, /maxTokens/],
      [{ stop: [''] }, /stop/],
    ]) {
      await assert.rejects(model.generate('hi', options), pattern);
    }
  });

  test('tokenize and countTokens agree', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const ids = model.tokenize('Once upon a time');
    assert.ok(ids.length > 0 && ids.every(Number.isInteger));
    assert.equal(model.countTokens('Once upon a time'), ids.length);
  });

  test('a prompt longer than the context window is rejected', async (t) => {
    const model = await openModel(t, library, 'tiny', { contextSize: 64 });
    await assert.rejects(model.generate('word '.repeat(500)), /context window/);
  });

  test('hardware() describes this machine', () => {
    const info = library.hardware();
    assert.ok(info.physicalCores > 0 && info.logicalCores >= info.physicalCores);
    assert.ok(info.devices.some((device) => device.kind === 'Cpu'));
    assert.match(library.backend, /^(cpu|cuda|vulkan|metal)$/);
  });
});
