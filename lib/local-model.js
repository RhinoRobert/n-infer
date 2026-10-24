import { normalizeConfig } from './config.js';
import { native } from './native.js';
import { normalizeInput, normalizeOptions } from './options.js';
import { collect, toAsyncIterator } from './stream.js';

/** The accelerator the loaded binary was compiled for: "cuda", "vulkan", "metal" or "cpu". */
export const backend = native.buildBackend();

/** CPU cores and instruction sets, plus every compute device with its free memory. */
export function hardware() {
  return native.hardwareInfo();
}

export class LocalModel {
  #config;
  #model = null;
  #loading = null;
  #closed = false;

  /**
   * @param {object} config
   * @param {string} config.modelPath            Path to a .gguf file.
   * @param {number | 'auto'} [config.gpuLayers] Layers to offload; 'auto' fits the free VRAM
   *                                             (the default on GPU builds, 0 on CPU builds).
   * @param {number} [config.contextSize=4096]
   * @param {number} [config.threads]            Defaults to the physical core count.
   * @param {number} [config.maxConcurrent=1]    Generations that may run at once.
   * @param {string} [config.modelSha256]        Refuse to load unless the file matches.
   * @param {boolean} [config.verbose=false]     Print llama.cpp's own logs.
   */
  constructor(config) {
    this.#config = normalizeConfig(config);
  }

  get ready() {
    return this.#model !== null;
  }

  /** Loads the model (into RAM and, if offloading, VRAM) and resolves with `this`. */
  async init() {
    if (this.#model) return this;
    if (this.#loading) return this.#loading;

    this.#closed = false;
    if (this.#config.gpuLayers > 0 && backend === 'cpu') {
      process.emitWarning(
        'gpuLayers was set but this build has no GPU support; running on the CPU.',
        'N-InferWarning',
      );
    }

    this.#loading = native
      .loadModel(this.#config)
      .then((loaded) => {
        if (this.#closed) {
          loaded.close();
          throw new Error('LocalModel was closed while loading');
        }
        this.#model = loaded;
        return this;
      })
      .finally(() => {
        this.#loading = null;
      });
    return this.#loading;
  }

  /**
   * Generates a completion and resolves with all of it.
   *
   * @param {string | Array<{ role: 'system' | 'user' | 'assistant', content: string }>} input
   *   Plain text, or chat messages (the model's own chat template is applied).
   * @param {GenerateOptions} [options]
   * @returns {Promise<string>}
   */
  async generate(input, options) {
    const { start, signal } = this.#prepare(input, options);
    return collect(start, signal);
  }

  /**
   * Generates a completion piece by piece. Breaking out of the loop, or aborting
   * `options.signal`, stops generation.
   *
   * @param {string | Array<{ role: string, content: string }>} input
   * @param {GenerateOptions} [options]
   * @returns {AsyncGenerator<string, void, void>}
   */
  async *stream(input, options) {
    const { start, signal } = this.#prepare(input, options);
    yield* toAsyncIterator(start, signal);
  }

  /**
   * Token ids a prompt becomes, exactly as `generate()` would tokenize it.
   *
   * @param {string | Array<{ role: string, content: string }>} input
   * @param {{ parseSpecial?: boolean }} [options]
   * @returns {number[]}
   */
  tokenize(input, { parseSpecial = false } = {}) {
    const model = this.#loaded();
    return model.tokenize(this.#segments(model, normalizeInput(input), parseSpecial));
  }

  /** How many tokens a prompt takes up. */
  countTokens(input, options) {
    return this.tokenize(input, options).length;
  }

  /** Stops all work and frees the model and its memory before returning. */
  close() {
    this.#closed = true;
    this.#model?.close();
    this.#model = null;
  }

  [Symbol.dispose]() {
    this.close();
  }

  /** Validates arguments and returns what `stream`/`collect` need to start a native run. */
  #prepare(input, options) {
    const model = this.#loaded();
    const { native: nativeOptions, signal, parseSpecial } = normalizeOptions(options);
    const segments = this.#segments(model, normalizeInput(input), parseSpecial);
    return { start: (callback) => model.run(segments, nativeOptions, callback), signal };
  }

  /**
   * Splits a prompt into segments. Chat messages go through the model's template, with template
   * text trusted and message text never parsed for control tokens; plain text is untrusted
   * unless `parseSpecial` says otherwise.
   */
  #segments(model, prompt, parseSpecial) {
    return prompt.messages
      ? model.renderChat(prompt.messages)
      : [{ text: prompt.text, special: parseSpecial }];
  }

  #loaded() {
    if (!this.#model) throw new Error('LocalModel: call `await init()` first');
    return this.#model;
  }
}

/**
 * @typedef {object} GenerateOptions
 * @property {number} [maxTokens=256]
 * @property {number} [temperature=0.8]        0 selects greedy decoding.
 * @property {number} [topP=0.95]
 * @property {number} [topK=40]                0 disables the cut.
 * @property {number} [repetitionPenalty=1.1]  1 disables it.
 * @property {number} [seed]
 * @property {string | string[]} [stop]        Generation ends, without emitting it, at any of these.
 * @property {boolean} [cache=true]            Reuse the warm KV cache for shared prompt prefixes.
 * @property {boolean} [parseSpecial=false]    Let a plain prompt contain control tokens such as
 *                                             `<|im_start|>`. Leave off for untrusted text.
 * @property {AbortSignal} [signal]
 */
