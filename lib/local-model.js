import { statSync } from 'node:fs';
import { native } from './native.js';
import { toAsyncIterator } from './stream.js';

const MAX_PROMPT_BYTES = 4 * 1024 * 1024;

/** "cuda", "vulkan" or "cpu": the accelerator the loaded binary was compiled with. */
export const backend = native.buildBackend();

/** CPU cores and instruction sets, plus every compute device with its free memory. */
export function hardware() {
  return native.hardwareInfo();
}

function own(obj, key) {
  return Object.hasOwn(obj, key) ? obj[key] : undefined;
}

function validateInt(name, val, min, max) {
  if (val === undefined) return undefined;
  if (typeof val !== 'number' || !Number.isInteger(val) || val < min || val > max) {
    throw new RangeError(`LocalModel: \`${name}\` must be an integer between ${min} and ${max}`);
  }
  return val;
}

function sanitizeConfig(config) {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('LocalModel: configuration object is required');
  }
  const modelPath = own(config, 'modelPath');
  if (typeof modelPath !== 'string' || modelPath.trim() === '' || modelPath.includes('\0')) {
    throw new TypeError('LocalModel: `modelPath` must be a non-empty string');
  }
  let stat;
  try {
    stat = statSync(modelPath);
  } catch {
    throw new Error(`LocalModel: model file not found: ${modelPath}`);
  }
  if (!stat.isFile()) {
    throw new Error(`LocalModel: modelPath is not a regular file: ${modelPath}`);
  }

  const clean = Object.create(null);
  clean.modelPath = modelPath;

  const rawGpuLayers = own(config, 'gpuLayers');
  if (rawGpuLayers === 'auto') {
    clean.gpuAuto = true;
  } else {
    const gpuLayers = validateInt('gpuLayers', rawGpuLayers, 0, 1000);
    if (gpuLayers !== undefined) clean.gpuLayers = gpuLayers;
  }

  const contextSize = validateInt('contextSize', own(config, 'contextSize'), 64, 131072);
  if (contextSize !== undefined) clean.contextSize = contextSize;

  for (const [key, min, max] of [
    ['threads', 1, 256],
    ['threadsBatch', 1, 256],
    ['batchSize', 1, 16384],
    ['ubatchSize', 1, 16384],
  ]) {
    const value = validateInt(key, own(config, key), min, max);
    if (value !== undefined) clean[key] = value;
  }

  const verbose = own(config, 'verbose');
  if (verbose !== undefined) {
    if (typeof verbose !== 'boolean') {
      throw new TypeError('LocalModel: `verbose` must be a boolean');
    }
    clean.verbose = verbose;
  }

  return Object.freeze(clean);
}

function validatePrompt(prompt) {
  if (typeof prompt !== 'string') {
    throw new TypeError('LocalModel: `prompt` must be a string');
  }
  if (prompt.length === 0) {
    throw new RangeError('LocalModel: `prompt` must not be empty');
  }
  if (Buffer.byteLength(prompt, 'utf8') > MAX_PROMPT_BYTES) {
    throw new RangeError(`LocalModel: \`prompt\` exceeds maximum size (${MAX_PROMPT_BYTES} bytes)`);
  }
}

function sanitizeOptions(options) {
  if (options === undefined || options === null) {
    return { nativeOpts: null, signal: undefined };
  }
  if (typeof options !== 'object') {
    throw new TypeError('LocalModel: `options` must be an object');
  }

  const clean = Object.create(null);

  const maxTokens = validateInt('maxTokens', own(options, 'maxTokens'), 0, 131072);
  if (maxTokens !== undefined) clean.maxTokens = maxTokens;

  const temperature = own(options, 'temperature');
  if (temperature !== undefined) {
    if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature < 0 || temperature > 100) {
      throw new RangeError('LocalModel: `temperature` must be a finite number between 0 and 100');
    }
    clean.temperature = temperature;
  }

  const topP = own(options, 'topP');
  if (topP !== undefined) {
    if (typeof topP !== 'number' || !Number.isFinite(topP) || topP <= 0 || topP > 1) {
      throw new RangeError('LocalModel: `topP` must be a finite number in (0, 1]');
    }
    clean.topP = topP;
  }

  const topK = validateInt('topK', own(options, 'topK'), 0, 100000);
  if (topK !== undefined) clean.topK = topK;

  const repetitionPenalty = own(options, 'repetitionPenalty');
  if (repetitionPenalty !== undefined) {
    if (
      typeof repetitionPenalty !== 'number' ||
      !Number.isFinite(repetitionPenalty) ||
      repetitionPenalty <= 0 ||
      repetitionPenalty > 10
    ) {
      throw new RangeError('LocalModel: `repetitionPenalty` must be a finite number in (0, 10]');
    }
    clean.repetitionPenalty = repetitionPenalty;
  }

  const seed = validateInt('seed', own(options, 'seed'), 0, 0xffffffff);
  if (seed !== undefined) clean.seed = seed;

  const signal = own(options, 'signal');
  if (signal !== undefined && (signal === null || typeof signal.aborted !== 'boolean')) {
    throw new TypeError('LocalModel: `signal` must be an AbortSignal');
  }

  return { nativeOpts: clean, signal };
}

export class LocalModel {
  #config;
  #model = null;
  #initPromise = null;
  #closed = false;

  /**
   * @param {object} config
   * @param {string} config.modelPath        Path to a .gguf file.
   * @param {number} [config.gpuLayers=0]    Layers to offload (ignored on CPU-only builds).
   * @param {number} [config.contextSize=4096]
   * @param {number} [config.threads]        CPU threads (default: llama.cpp's choice).
   * @param {boolean} [config.verbose=false] Print llama.cpp logs.
   */
  constructor(config) {
    this.#config = sanitizeConfig(config);
  }

  get ready() {
    return this.#model !== null;
  }

  /** Loads the model into RAM (and VRAM, if offloading). Resolves with `this`. */
  async init() {
    if (this.#model) return this;
    if (this.#initPromise) return this.#initPromise;

    this.#closed = false;
    if (this.#config.gpuLayers > 0 && backend === 'cpu') {
      process.emitWarning(
        'n-infer: gpuLayers was set but this binary is CPU-only; running on CPU. ' +
          'Rebuild with `npm run build:cuda` for GPU offload.',
      );
    }

    this.#initPromise = native
      .loadModel(this.#config)
      .then((loaded) => {
        if (this.#closed) {
          loaded.close();
          throw new Error('LocalModel: closed during initialization');
        }
        this.#model = loaded;
        return this;
      })
      .finally(() => {
        this.#initPromise = null;
      });

    return this.#initPromise;
  }

  /**
   * Resolves with the full completion.
   * @param {string} prompt
   * @param {{maxTokens?: number, temperature?: number, topP?: number, repetitionPenalty?: number, seed?: number, signal?: AbortSignal}} [options]
   */
  async generate(prompt, options) {
    const model = this.#loaded();
    validatePrompt(prompt);
    const { nativeOpts, signal } = sanitizeOptions(options);

    if (!signal) {
      return model.generate(prompt, nativeOpts, null);
    }
    if (signal.aborted) {
      throw signal.reason ?? new DOMException('The operation was aborted', 'AbortError');
    }

    const handle = new native.CancelHandle();
    const onAbort = () => handle.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      return await model.generate(prompt, nativeOpts, handle);
    } catch (error) {
      if (signal.aborted) throw signal.reason ?? new DOMException('The operation was aborted', 'AbortError');
      throw error;
    } finally {
      signal.removeEventListener('abort', onAbort);
    }
  }

  /** Async iterator of text chunks. Breaking out of the loop or aborting `options.signal` cancels generation. */
  stream(prompt, options) {
    const model = this.#loaded();
    validatePrompt(prompt);
    const { nativeOpts, signal } = sanitizeOptions(options);
    return toAsyncIterator((callback) => model.stream(prompt, nativeOpts, callback), signal);
  }

  /** Deterministically releases native model weights and KV-cache VRAM immediately. */
  close() {
    this.#closed = true;
    if (this.#model) {
      this.#model.close();
      this.#model = null;
    }
  }

  #loaded() {
    if (!this.#model) throw new Error('LocalModel: call `await init()` first');
    return this.#model;
  }
}
