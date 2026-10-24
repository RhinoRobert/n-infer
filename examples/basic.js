import { LocalModel, backend } from '../index.js';

const modelPath = process.argv[2];
if (!modelPath) {
  console.error('usage: node examples/basic.js <model.gguf> [gpuLayers]');
  process.exit(1);
}

const ai = new LocalModel({ modelPath, gpuLayers: Number(process.argv[3] ?? 0) });

console.log(`backend: ${backend}`);
console.time('load');
await ai.init();
console.timeEnd('load');

console.time('generate');
console.log(await ai.generate('The capital of France is', { maxTokens: 32, temperature: 0 }));
console.timeEnd('generate');

for await (const chunk of ai.stream('Write one sentence about Algorand:', { maxTokens: 64 })) {
  process.stdout.write(chunk);
}
process.stdout.write('\n');
