// Where the tests find models and the library, and when they should skip instead of fail.
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const fixturesDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'fixtures');

/** Each fixture can be overridden with an environment variable pointing at any GGUF file. */
const FIXTURES = {
  /** ~1 MB story model: fast, no chat template. */
  tiny: { env: 'N_INFER_TEST_MODEL', file: 'stories260K.gguf' },
  /** ~100 MB instruction model with a ChatML template. */
  chat: { env: 'N_INFER_TEST_CHAT_MODEL', file: 'SmolLM2-135M-Instruct-Q4_K_M.gguf' },
};

export function fixturePath(kind) {
  const { env, file } = FIXTURES[kind];
  return process.env[env] ?? join(fixturesDir, file);
}

/** `false` when the model exists, otherwise a reason string for `{ skip }`. */
export function skipUnlessModel(kind) {
  if (existsSync(fixturePath(kind))) return false;
  return `no ${kind} model: run \`npm run test:fetch\` or set ${FIXTURES[kind].env}`;
}

/** Loads the library, or returns null when the native binary has not been built. */
export async function loadLibrary() {
  try {
    return await import('../../lib/index.js');
  } catch (error) {
    if (/no prebuilt binary|could not find its binary|failed to load/.test(error.message))
      return null;
    throw error;
  }
}

/** Opens a model for a test and closes it afterwards. */
export async function openModel(t, library, kind, config = {}) {
  const model = await new library.LocalModel({ modelPath: fixturePath(kind), ...config }).init();
  t.after(() => model.close());
  return model;
}

/** Greedy, penalty-free options: output depends only on the prompt, so it can be compared. */
export const GREEDY = Object.freeze({ temperature: 0, repetitionPenalty: 1, maxTokens: 16 });
