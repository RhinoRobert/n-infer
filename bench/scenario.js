// The measurement itself, shared by every driver so each engine is timed identically.
// A driver supplies `engine`: { load(), generate(prompt, maxTokens), countTokens(text), close() }.

const RECORDS = 140;
const LONG_PROMPT = Array.from(
  { length: RECORDS },
  (_, i) => `Record ${i}: sensor reading ${i * 13} units, status nominal.`,
).join(' ');
const SHORT_PROMPT = 'Write a long story about a dragon who learns to code.';
const DECODE_TOKENS = 128;
const WARM_REPEATS = 15;

/** Runs the scenario and returns one flat object of numbers. */
export async function measure(engine) {
  const loadStart = performance.now();
  await engine.load();
  const loadMs = performance.now() - loadStart;

  const time = async (prompt, maxTokens) => {
    const start = performance.now();
    const text = await engine.generate(prompt, maxTokens);
    return { ms: performance.now() - start, text };
  };

  await time(SHORT_PROMPT, 4); // warm-up: kernels, caches
  await time('An unrelated prompt to evict the cache.', 1);

  const promptTokens = engine.countTokens(LONG_PROMPT);
  const prefill = await time(LONG_PROMPT, 1);

  await time('Another unrelated prompt.', 1);
  const oneToken = await time(SHORT_PROMPT, 1);
  const decode = await time(SHORT_PROMPT, DECODE_TOKENS + 1);
  const produced = engine.countTokens(decode.text);

  await time(SHORT_PROMPT, 1);
  const warm = [];
  for (let i = 0; i < WARM_REPEATS; i += 1) warm.push((await time(SHORT_PROMPT, 1)).ms);
  warm.sort((a, b) => a - b);

  await engine.close();
  return {
    loadMs: Math.round(loadMs),
    prefillTps: Math.round(promptTokens / (prefill.ms / 1000)),
    decodeTps: +(Math.max(produced - 1, 1) / ((decode.ms - oneToken.ms) / 1000)).toFixed(1),
    warmMs: +warm[Math.floor(warm.length / 2)].toFixed(1),
    rssMb: Math.round(process.memoryUsage().rss / 1024 / 1024),
  };
}
