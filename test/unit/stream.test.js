import assert from 'node:assert/strict';
import { test } from 'node:test';
import { collect, toAsyncIterator } from '../../lib/stream.js';

/** A fake native run: feeds `script` to the callback on later ticks and records cancel(). */
function fakeRun(script) {
  const run = { cancelled: false };
  run.start = (callback) => {
    (async () => {
      for (const step of script) {
        await new Promise((resolve) => setImmediate(resolve));
        if (run.cancelled) return;
        callback(step.error ?? null, step.chunk ?? null);
      }
    })();
    return { cancel: () => (run.cancelled = true) };
  };
  return run;
}

const text = (chunk) => ({ chunk });
const done = { chunk: null };

test('yields chunks in order and finishes', async () => {
  const run = fakeRun([text('a'), text('b'), done]);
  const seen = [];
  for await (const chunk of toAsyncIterator(run.start)) seen.push(chunk);
  assert.deepEqual(seen, ['a', 'b']);
});

test('collect joins every chunk', async () => {
  assert.equal(await collect(fakeRun([text('he'), text('llo'), done]).start), 'hello');
});

test('surfaces native errors', async () => {
  const run = fakeRun([text('a'), { error: new Error('boom') }]);
  await assert.rejects(collect(run.start), /boom/);
});

test('breaking out of the loop cancels the run', async () => {
  const run = fakeRun([text('a'), text('b'), text('c'), done]);
  for await (const chunk of toAsyncIterator(run.start)) {
    assert.equal(chunk, 'a');
    break;
  }
  assert.equal(run.cancelled, true);
});

test('an already-aborted signal rejects without starting', async () => {
  const controller = new AbortController();
  controller.abort();
  let started = false;
  const start = () => ((started = true), { cancel() {} });
  await assert.rejects(collect(start, controller.signal), { name: 'AbortError' });
  assert.equal(started, false);
});

test('aborting mid-run cancels and rejects with the signal reason', async () => {
  const run = fakeRun([text('a'), text('b'), text('c'), done]);
  const controller = new AbortController();
  const seen = [];
  await assert.rejects(
    async () => {
      for await (const chunk of toAsyncIterator(run.start, controller.signal)) {
        seen.push(chunk);
        controller.abort();
      }
    },
    { name: 'AbortError' },
  );
  assert.equal(run.cancelled, true);
  assert.deepEqual(seen, ['a']);
});

test('a custom abort reason is preserved', async () => {
  const run = fakeRun([text('a'), text('b'), done]);
  const reason = new Error('timed out');
  const controller = new AbortController();
  const promise = collect(run.start, controller.signal);
  controller.abort(reason);
  await assert.rejects(promise, /timed out/);
});
