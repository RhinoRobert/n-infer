// node-llama-cpp driver (v3 API). Prints one JSON line of results.
//   npm install --prefix bench && node bench/drivers/node-llama-cpp.js <model.gguf>
//
// Settings are matched to n-infer's defaults where they exist: GPU offload as much as fits,
// 4096-token context, greedy decoding, no repetition penalty.
import { measure } from '../scenario.js';

const { getLlama, LlamaCompletion } = await import('node-llama-cpp').catch(() => {
  console.error('node-llama-cpp is not installed: run `npm install --prefix bench`');
  process.exit(2);
});

const modelPath = process.argv[2];
let llama;
let model;
let context;
let completion;

const results = await measure({
  load: async () => {
    llama = await getLlama();
    model = await llama.loadModel({ modelPath, gpuLayers: 'auto' });
    context = await model.createContext({ contextSize: 4096 });
    completion = new LlamaCompletion({ contextSequence: context.getSequence() });
  },
  generate: (prompt, maxTokens) =>
    completion.generateCompletion(prompt, { maxTokens, temperature: 0, repeatPenalty: false }),
  countTokens: (text) => model.tokenize(text).length,
  close: async () => {
    await context.dispose();
    await model.dispose();
  },
});
console.log(JSON.stringify(results));
process.exit(0);
