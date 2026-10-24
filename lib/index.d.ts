export interface LocalModelConfig {
  /** Path to a .gguf file. */
  modelPath: string;
  /**
   * Layers to offload to the GPU, or 'auto' to fit as many as the free VRAM allows.
   * Default: 'auto' on GPU builds, 0 on CPU builds.
   */
  gpuLayers?: number | 'auto';
  /** Context window in tokens. Default 4096. */
  contextSize?: number;
  /** Threads for token generation. Default: physical core count. */
  threads?: number;
  /** Threads for prompt processing. Default: physical core count. */
  threadsBatch?: number;
  /** Tokens per decode call while processing a prompt. Default 2048. */
  batchSize?: number;
  /** Tokens per compute pass. Default 512. */
  ubatchSize?: number;
  /** Generations that may run at once; each owns a full context. Default 1. */
  maxConcurrent?: number;
  /** Requests allowed to wait for a free worker before new ones are rejected. Default 64. */
  maxQueue?: number;
  /** Refuse to load the file unless its SHA-256 (hex) matches. */
  modelSha256?: string;
  /** Print llama.cpp's own logs to stderr. */
  verbose?: boolean;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface GenerateOptions {
  /** Default 256. */
  maxTokens?: number;
  /** 0 selects greedy decoding. Default 0.8. */
  temperature?: number;
  /** Default 0.95. */
  topP?: number;
  /** Candidates kept before top-p; 0 disables the cut. Default 40. */
  topK?: number;
  /** Penalty on recently seen tokens; 1 disables it. Default 1.1. */
  repetitionPenalty?: number;
  seed?: number;
  /** Generation ends, without emitting it, when any of these appears. */
  stop?: string | string[];
  /** Reuse the warm KV cache for shared prompt prefixes. Default true. */
  cache?: boolean;
  /**
   * Let a plain-string prompt contain control tokens such as `<|im_start|>`.
   * Leave off for untrusted text. Chat messages are never parsed for control tokens.
   */
  parseSpecial?: boolean;
  /** Stops generation when aborted; the call then rejects with the signal's reason. */
  signal?: AbortSignal;
}

export type Prompt = string | ChatMessage[];

export interface DeviceInfo {
  name: string;
  description: string;
  backend: string;
  kind: string;
  totalMib: number;
  freeMib: number;
}

export interface HardwareInfo {
  physicalCores: number;
  logicalCores: number;
  buildBackend: 'cuda' | 'vulkan' | 'metal' | 'cpu';
  /** Instruction sets this CPU supports (x86). */
  cpuFeatures: string[];
  /** Instruction sets this binary requires (x86). */
  compiledFor: string[];
  /** llama.cpp's own report of the CPU features it was compiled to use. */
  systemInfo: string;
  devices: DeviceInfo[];
}

/** The accelerator the loaded binary was compiled for. */
export const backend: 'cuda' | 'vulkan' | 'metal' | 'cpu';

/** CPU cores and instruction sets, plus every compute device with its free memory. */
export function hardware(): HardwareInfo;

export class LocalModel {
  constructor(config: LocalModelConfig);
  /** True once `init()` has finished and until `close()`. */
  readonly ready: boolean;
  /** Loads the model into memory. Resolves with the same instance. */
  init(): Promise<this>;
  /** Generates a completion and resolves with all of it. */
  generate(prompt: Prompt, options?: GenerateOptions): Promise<string>;
  /** Generates a completion piece by piece. Breaking out of the loop stops generation. */
  stream(prompt: Prompt, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  /** Token ids a prompt becomes, exactly as `generate()` would tokenize it. */
  tokenize(prompt: Prompt, options?: { parseSpecial?: boolean }): number[];
  /** How many tokens a prompt takes up. */
  countTokens(prompt: Prompt, options?: { parseSpecial?: boolean }): number;
  /** Stops all work and frees the model and its memory before returning. */
  close(): void;
  [Symbol.dispose](): void;
}

export default LocalModel;
