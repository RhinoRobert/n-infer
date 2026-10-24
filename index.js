import { createRequire } from 'node:module';
import { readdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));

function loadNative() {
  const platformPrefix = `n-infer.${process.platform}-${process.arch}`;
  const binaries = readdirSync(here).filter((f) => f.startsWith('n-infer') && f.endsWith('.node'));
  const file =
    binaries.find((f) => f.startsWith(platformPrefix)) ??
    binaries.find((f) => f === 'n-infer.node');
  if (!file) {
    throw new Error(
      `n-infer: no native binary found in ${here}. Run \`npm run build\` first.`,
    );
  }
  return require(resolve(here, file));
}

const native = loadNative();

/** "cuda", "vulkan" or "cpu" - the accelerator the loaded binary was compiled with. */
export const backend = native.buildBackend();

export class LocalModel {
  #config;
  #model = null;

  /**
   * @param {object} config
   * @param {string} config.modelPath      Path to a .gguf file.
   * @param {number} [config.gpuLayers=0]  Layers to offload (ignored on CPU-only builds).
   * @param {number} [config.contextSize=4096]
   * @param {number} [config.threads]      CPU threads (defaults to llama.cpp's choice).
   * @param {boolean} [config.verbose=false] Print llama.cpp logs.
   */
  constructor(config) {
    if (!config || typeof config.modelPath !== 'string') {
      throw new TypeError('LocalModel: `modelPath` is required');
    }
    if (!existsSync(config.modelPath)) {
      throw new Error(`LocalModel: model file not found: ${config.modelPath}`);
    }
    this.#config = { ...config };
  }

  get ready() {
    return this.#model !== null;
  }

  /** Loads the model into RAM (and VRAM, if offloading). Resolves with `this`. */
  async init() {
    if (this.#model) return this;
    const { gpuLayers = 0 } = this.#config;
    if (gpuLayers > 0 && backend === 'cpu') {
      process.emitWarning(
        'n-infer: gpuLayers was set but this binary is CPU-only; running on CPU. ' +
          'Rebuild with `npm run build:cuda` for GPU offload.',
      );
    }
    this.#model = await native.loadModel(this.#config);
    return this;
  }

  /**
   * Resolves with the full completion.
   * @param {string} prompt
   * @param {{maxTokens?: number, temperature?: number, topP?: number, seed?: number}} [options]
   */
  async generate(prompt, options) {
    return this.#loaded().generate(prompt, options ?? null);
  }

  /**
   * Async iterator yielding text chunks as they are produced.
   * Breaking out of the loop cancels generation.
   */
  async *stream(prompt, options) {
    const model = this.#loaded();
    const queue = [];
    let done = false;
    let failure = null;
    let wake = null;

    const handle = model.stream(prompt, options ?? null, (err, chunk) => {
      if (err) failure = err;
      else if (chunk === null) done = true;
      else queue.push(chunk);
      if (wake) {
        const resume = wake;
        wake = null;
        resume();
      }
    });

    try {
      for (;;) {
        if (queue.length) {
          yield queue.shift();
        } else if (failure) {
          throw failure;
        } else if (done) {
          return;
        } else {
          await new Promise((resolve) => (wake = resolve));
        }
      }
    } finally {
      if (!done && !failure) handle.cancel();
    }
  }

  /** Drops the reference so the model memory can be reclaimed by GC. */
  close() {
    this.#model = null;
  }

  #loaded() {
    if (!this.#model) throw new Error('LocalModel: call `await init()` first');
    return this.#model;
  }
}

export default LocalModel;
