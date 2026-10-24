// n-infer driver. Prints one JSON line of results.   node bench/drivers/n-infer.js <model.gguf>
import { LocalModel } from '../../lib/index.js';
import { measure } from '../scenario.js';

const modelPath = process.argv[2];
let model;

const results = await measure({
  load: async () => {
    model = await new LocalModel({ modelPath }).init();
  },
  generate: (prompt, maxTokens) =>
    model.generate(prompt, { maxTokens, temperature: 0, repetitionPenalty: 1 }),
  countTokens: (text) => model.countTokens(text),
  close: () => model.close(),
});
console.log(JSON.stringify(results));
process.exit(0);
