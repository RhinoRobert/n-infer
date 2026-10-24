// Turns the caller's model config into a plain, frozen object the native side can trust.
//
// This file only checks *types* and copies *own* properties (so a polluted Object.prototype in
// the host app cannot change the configuration). Numeric ranges are enforced in Rust.
import { statSync } from 'node:fs';
import { assertBoolean, assertInteger, ownProps } from './validate.js';

const INTEGER_KEYS = [
  'contextSize',
  'threads',
  'threadsBatch',
  'batchSize',
  'ubatchSize',
  'maxConcurrent',
  'maxQueue',
];
const KNOWN_KEYS = new Set(['modelPath', 'gpuLayers', 'modelSha256', 'verbose', ...INTEGER_KEYS]);

/**
 * @param {object} config
 * @returns {Readonly<object>} config for `native.loadModel`
 */
export function normalizeConfig(config) {
  const own = ownProps(config, KNOWN_KEYS, 'config');

  if (typeof own.modelPath !== 'string' || own.modelPath.trim() === '') {
    throw new TypeError('LocalModel: `modelPath` must be a non-empty string');
  }
  assertModelFile(own.modelPath);

  const clean = Object.create(null);
  clean.modelPath = own.modelPath;

  if (own.gpuLayers === 'auto') clean.gpuAuto = true;
  else if (own.gpuLayers !== undefined) clean.gpuLayers = assertInteger('gpuLayers', own.gpuLayers);

  for (const name of INTEGER_KEYS) {
    if (own[name] !== undefined) clean[name] = assertInteger(name, own[name]);
  }
  if (own.modelSha256 !== undefined) {
    if (typeof own.modelSha256 !== 'string')
      throw new TypeError('LocalModel: `modelSha256` must be a string');
    clean.modelSha256 = own.modelSha256;
  }
  if (own.verbose !== undefined) clean.verbose = assertBoolean('verbose', own.verbose);

  return Object.freeze(clean);
}

function assertModelFile(path) {
  let stat;
  try {
    stat = statSync(path);
  } catch {
    throw new Error(`LocalModel: model file not found: ${path}`);
  }
  if (!stat.isFile()) throw new Error(`LocalModel: modelPath is not a regular file: ${path}`);
}
