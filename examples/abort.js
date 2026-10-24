// Stop a generation after a time limit, or when the consumer has seen enough.
//   node examples/abort.js ./models/model.gguf
import { LocalModel } from '../lib/index.js';

const modelPath = process.argv[2];
if (!modelPath) {
  console.error('usage: node examples/abort.js <model.gguf>');
  process.exit(1);
}

const ai = await new LocalModel({ modelPath }).init();
try {
  try {
    const text = await ai.generate('Count from one to one thousand, in words:', {
      maxTokens: 2000,
      signal: AbortSignal.timeout(1_500),
    });
    console.log('finished in time:', text.length, 'characters');
  } catch (error) {
    if (error.name !== 'TimeoutError' && error.name !== 'AbortError') throw error;
    console.log('stopped by the time limit');
  }

  // Breaking out of the loop stops generation too.
  let pieces = 0;
  for await (const piece of ai.stream('Tell me a long story:', { maxTokens: 500 })) {
    process.stdout.write(piece);
    if (++pieces === 20) break;
  }
  console.log('\n(stopped after 20 pieces)');
} finally {
  ai.close();
}
