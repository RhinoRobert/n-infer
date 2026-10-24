export interface LocalModelConfig {
  modelPath: string;
  /**
   * Layers to offload to the GPU, or 'auto' to fit as many as the free VRAM allows.
   * Default: 'auto' on GPU builds, 0 on CPU builds.
   */
  gpuLayers?: number | 'auto';
  contextSize?: number;
  /** Threads for token generation. Default: physical core count. */
  threads?: number;
  /** Threads for prompt processing. Default: physical core count. */
  threadsBatch?: number;
  /** Tokens per decode call while processing a prompt. Default 2048. */
  batchSize?: number;
  /** Tokens per compute pass. Default 512. */
  ubatchSize?: number;
  verbose?: boolean;
}

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
  buildBackend: 'cuda' | 'vulkan' | 'cpu';
  /** Instruction sets this CPU supports (x86). */
  cpuFeatures: string[];
  /** Instruction sets this binary requires (x86). */
  compiledFor: string[];
  /** llama.cpp's own report of the CPU features it was compiled to use. */
  systemInfo: string;
  devices: DeviceInfo[];
}

export function hardware(): HardwareInfo;

export interface GenerateOptions {
  maxTokens?: number;
  /** 0 selects greedy decoding. */
  temperature?: number;
  topP?: number;
  /** Candidates kept before top-p; 0 disables the cut. Default 40. */
  topK?: number;
  /** Penalty on recently seen tokens; 1 disables it. Default 1.1. */
  repetitionPenalty?: number;
  seed?: number;
  /** Optional AbortSignal to cancel generation early. */
  signal?: AbortSignal;
}

export const backend: 'cuda' | 'vulkan' | 'cpu';

export class LocalModel {
  constructor(config: LocalModelConfig);
  readonly ready: boolean;
  init(): Promise<this>;
  generate(prompt: string, options?: GenerateOptions): Promise<string>;
  stream(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  close(): void;
}

export default LocalModel;
