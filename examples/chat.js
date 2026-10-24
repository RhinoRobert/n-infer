// A small interactive chat. Message text is never parsed for control tokens.
//   node examples/chat.js ./models/chat-model.gguf
import { createInterface } from 'node:readline/promises';
import { LocalModel } from '../lib/index.js';

const modelPath = process.argv[2];
if (!modelPath) {
  console.error('usage: node examples/chat.js <chat-model.gguf>');
  process.exit(1);
}

const ai = await new LocalModel({ modelPath, contextSize: 4096 }).init();
const terminal = createInterface({ input: process.stdin, output: process.stdout });
const history = [{ role: 'system', content: 'You are a helpful, concise assistant.' }];

try {
  while (true) {
    const line = (await terminal.question('\nyou> ')).trim();
    if (!line || line === '/quit') break;

    history.push({ role: 'user', content: line });
    let reply = '';
    process.stdout.write('ai>  ');
    for await (const piece of ai.stream(history, { maxTokens: 400 })) {
      process.stdout.write(piece);
      reply += piece;
    }
    history.push({ role: 'assistant', content: reply });
  }
} finally {
  terminal.close();
  ai.close();
}
