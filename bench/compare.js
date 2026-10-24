// Benchmarks local-inference engines on the same model, in separate processes, and prints a table.
//
//   node bench/compare.js --model ./model.gguf [--engine n-infer|node-llama-cpp|both] [--runs 3]
//
// To include node-llama-cpp:  npm install --prefix bench
//
// What is measured (greedy decoding, so every engine does the same work):
//   load      time from construction to ready
//   prefill   tokens/s processing a long prompt (1 token generated)
//   decode    tokens/s generating from a short prompt
//   warm 1tok median latency of repeating the same one-token request
//   memory    peak resident set size of the benchmark process
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    engine: { type: 'string', default: 'both' },
    runs: { type: 'string', default: '3' },
  },
});

if (!values.model) {
  console.error(
    'usage: node bench/compare.js --model <file.gguf> [--engine n-infer|node-llama-cpp|both] [--runs 3]',
  );
  process.exit(1);
}
const engines = values.engine === 'both' ? ['n-infer', 'node-llama-cpp'] : [values.engine];
const runs = Number(values.runs);

const results = [];
for (const engine of engines) {
  const driver = join(here, 'drivers', `${engine}.js`);
  const samples = [];
  for (let run = 0; run < runs; run += 1) {
    const child = spawnSync(process.execPath, [driver, resolve(values.model)], {
      encoding: 'utf8',
    });
    if (child.status !== 0) {
      console.error(
        `${engine}: run ${run + 1} failed\n${child.stderr.trim().split('\n').slice(-5).join('\n')}`,
      );
      break;
    }
    samples.push(JSON.parse(child.stdout.trim().split('\n').at(-1)));
  }
  if (samples.length > 0) results.push({ engine, ...median(samples) });
}

printTable(results);

function median(samples) {
  const keys = Object.keys(samples[0]);
  return Object.fromEntries(
    keys.map((key) => {
      const sorted = samples.map((s) => s[key]).sort((a, b) => a - b);
      return [key, sorted[Math.floor(sorted.length / 2)]];
    }),
  );
}

function printTable(rows) {
  const columns = [
    ['engine', 'engine'],
    ['loadMs', 'load (ms)'],
    ['prefillTps', 'prefill (tok/s)'],
    ['decodeTps', 'decode (tok/s)'],
    ['warmMs', 'warm 1-token (ms)'],
    ['rssMb', 'peak RSS (MB)'],
  ];
  const line = (cells) => `| ${cells.join(' | ')} |`;
  console.log(line(columns.map(([, title]) => title)));
  console.log(line(columns.map(() => '---')));
  for (const row of rows) console.log(line(columns.map(([key]) => row[key])));
}
