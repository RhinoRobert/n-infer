import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import {
  GREEDY,
  loadLibrary,
  openModel,
  skipUnlessModel,
  fixturePath,
} from '../helpers/fixtures.js';

const library = await loadLibrary();
const skip = library ? skipUnlessModel('tiny') : 'native binary not built: run `npm run build`';
const config = { contextSize: 512 };

describe('lifecycle', { skip }, () => {
  test('using a model before init() fails clearly', async () => {
    const model = new library.LocalModel({ modelPath: fixturePath('tiny') });
    assert.equal(model.ready, false);
    await assert.rejects(model.generate('hi'), /init/);
  });

  test('init() is idempotent and shares one load', async (t) => {
    const model = new library.LocalModel({ modelPath: fixturePath('tiny'), ...config });
    t.after(() => model.close());
    const [first, second] = await Promise.all([model.init(), model.init()]);
    assert.equal(first, model);
    assert.equal(second, model);
    assert.equal(await model.init(), model);
    assert.equal(model.ready, true);
  });

  test('close() frees the model and later calls fail', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    model.close();
    assert.equal(model.ready, false);
    await assert.rejects(model.generate('hi'), /init/);
    model.close(); // closing twice is harmless
  });

  test('a closed model can be loaded again', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    model.close();
    await model.init();
    assert.ok((await model.generate('Once upon', GREEDY)).length > 0);
  });

  test('closing during a generation rejects it instead of hanging', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    const pending = model.generate('Once upon a time', { maxTokens: 400, temperature: 0 });
    setImmediate(() => model.close());
    await assert.rejects(pending, /closed/);
  });

  test('closing during a stream ends it with an error', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    await assert.rejects(async () => {
      let pieces = 0;
      for await (const _ of model.stream('Once upon a time', { maxTokens: 400, temperature: 0 })) {
        if (++pieces === 2) model.close();
      }
    }, /closed/);
  });

  test('Symbol.dispose closes the model', async (t) => {
    const model = await openModel(t, library, 'tiny', config);
    model[Symbol.dispose]();
    assert.equal(model.ready, false);
  });

  test('loading and closing repeatedly does not crash', async () => {
    for (let round = 0; round < 5; round += 1) {
      const model = await new library.LocalModel({
        modelPath: fixturePath('tiny'),
        ...config,
      }).init();
      assert.ok((await model.generate('Once', { ...GREEDY, maxTokens: 4 })).length >= 0);
      model.close();
    }
  });
});
