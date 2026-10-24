// The shortest useful program: load a model, ask it something, stream another answer.
//   node examples/basic.js ./models/model.gguf
import { LocalModel, backend } from '../lib/index.js';

const modelPath = process.argv[2];
if (!modelPath) {
  console.error('usage: node examples/basic.js <model.gguf>');
  process.exit(1);
}

const ai = await new LocalModel({ modelPath }).init();
try {
  console.log(`running on: ${backend}\n`);
  console.log(await ai.generate('The capital of France is', { maxTokens: 24, temperature: 0 }));

  console.log('\n--- streaming ---');
  for await (const piece of ai.stream('Write one sentence about Algorand:', { maxTokens: 64 })) {
    process.stdout.write(piece);
  }
  process.stdout.write('\n');
} finally {
  ai.close();
}
