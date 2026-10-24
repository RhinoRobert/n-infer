export interface LocalModelConfig {
  modelPath: string;
  gpuLayers?: number;
  contextSize?: number;
  threads?: number;
  verbose?: boolean;
}

export interface GenerateOptions {
  maxTokens?: number;
  /** 0 or less selects greedy decoding. */
  temperature?: number;
  topP?: number;
  /** Penalty on recently seen tokens; 1 disables it. Default 1.1. */
  repetitionPenalty?: number;
  seed?: number;
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
