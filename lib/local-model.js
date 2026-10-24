import { existsSync } from 'node:fs';
import { native } from './native.js';
import { toAsyncIterator } from './stream.js';

/** "cuda", "vulkan" or "cpu": the accelerator the loaded binary was compiled with. */
export const backend = native.buildBackend();

export class LocalModel {
  #config;
  #model = null;

  /**
   * @param {object} config
   * @param {string} config.modelPath        Path to a .gguf file.
   * @param {number} [config.gpuLayers=0]    Layers to offload (ignored on CPU-only builds).
   * @param {number} [config.contextSize=4096]
   * @param {number} [config.threads]        CPU threads (default: llama.cpp's choice).
   * @param {boolean} [config.verbose=false] Print llama.cpp logs.
   */
  constructor(config) {
    if (typeof config?.modelPath !== 'string') {
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
    if (this.#config.gpuLayers > 0 && backend === 'cpu') {
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
   * @param {{maxTokens?: number, temperature?: number, topP?: number, repetitionPenalty?: number, seed?: number}} [options]
   */
  generate(prompt, options) {
    return this.#loaded().generate(prompt, options ?? null);
  }

  /** Async iterator of text chunks. Breaking out of the loop cancels generation. */
  stream(prompt, options) {
    const model = this.#loaded();
    return toAsyncIterator((callback) => model.stream(prompt, options ?? null, callback));
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
