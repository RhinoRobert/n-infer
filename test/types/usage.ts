// Compile-only check that the published type declarations accept the documented usage and
// reject mistakes. Run with `npm run test:types`.
import {
  LocalModel,
  backend,
  hardware,
  type ChatMessage,
  type GenerateOptions,
} from '../../lib/index.js';

async function main(): Promise<void> {
  const model = new LocalModel({ modelPath: './model.gguf', gpuLayers: 'auto', threads: 8 });
  await model.init();

  const text: string = await model.generate('Hello');
  const messages: ChatMessage[] = [
    { role: 'system', content: 'Be brief.' },
    { role: 'user', content: 'Hi' },
  ];
  const options: GenerateOptions = {
    maxTokens: 32,
    temperature: 0,
    stop: ['\n\n'],
    signal: AbortSignal.timeout(5_000),
  };
  const reply: string = await model.generate(messages, options);

  for await (const piece of model.stream(messages)) {
    const chunk: string = piece;
    void chunk;
  }

  const ids: number[] = model.tokenize('hello', { parseSpecial: true });
  const count: number = model.countTokens(messages);
  const cores: number = hardware().physicalCores;
  const accelerator: 'cuda' | 'vulkan' | 'metal' | 'cpu' = backend;
  using scoped = new LocalModel({ modelPath: './other.gguf' });
  void [text, reply, ids, count, cores, accelerator, scoped];
  model.close();
}
void main();

// @ts-expect-error: a model path is required
new LocalModel({});
// @ts-expect-error: gpuLayers must be a number or 'auto'
new LocalModel({ modelPath: 'x.gguf', gpuLayers: 'all' });
// @ts-expect-error: unknown roles are rejected at compile time
const bad: ChatMessage = { role: 'tool', content: 'x' };
void bad;
// @ts-expect-error: temperature must be a number
const options: GenerateOptions = { temperature: 'hot' };
void options;
