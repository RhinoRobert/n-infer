// Turns what callers pass to generate()/stream() into native arguments.
//
// Like config.js: types and own properties only; ranges are enforced in Rust.
import { assertBoolean, assertInteger, assertNumber, ownProps } from './validate.js';

const OPTION_KEYS = new Set([
  'maxTokens',
  'temperature',
  'topP',
  'topK',
  'repetitionPenalty',
  'seed',
  'stop',
  'cache',
  'parseSpecial',
  'signal',
]);
const MESSAGE_KEYS = new Set(['role', 'content']);

/**
 * @param {object | undefined} options
 * @returns {{ native: object, signal: AbortSignal | undefined, parseSpecial: boolean }}
 */
export function normalizeOptions(options) {
  const own =
    options === undefined || options === null
      ? Object.create(null)
      : ownProps(options, OPTION_KEYS, 'options');
  const native = Object.create(null);

  for (const name of ['maxTokens', 'topK', 'seed']) {
    if (own[name] !== undefined) native[name] = assertInteger(name, own[name]);
  }
  for (const name of ['temperature', 'topP', 'repetitionPenalty']) {
    if (own[name] !== undefined) native[name] = assertNumber(name, own[name]);
  }
  if (own.cache !== undefined) native.cache = assertBoolean('cache', own.cache);
  if (own.stop !== undefined) native.stop = normalizeStop(own.stop);

  const parseSpecial =
    own.parseSpecial === undefined ? false : assertBoolean('parseSpecial', own.parseSpecial);

  const { signal } = own;
  if (
    signal !== undefined &&
    !(
      signal &&
      typeof signal.aborted === 'boolean' &&
      typeof signal.addEventListener === 'function'
    )
  ) {
    throw new TypeError('LocalModel: `signal` must be an AbortSignal');
  }
  return { native, signal, parseSpecial };
}

function normalizeStop(stop) {
  const list = typeof stop === 'string' ? [stop] : stop;
  if (!Array.isArray(list) || list.some((s) => typeof s !== 'string')) {
    throw new TypeError('LocalModel: `stop` must be a string or an array of strings');
  }
  return [...list];
}

/**
 * Splits the prompt argument into plain text or chat messages.
 *
 * @param {string | Array<{ role: string, content: string }>} input
 * @returns {{ text: string } | { messages: Array<{ role: string, content: string }> }}
 */
export function normalizeInput(input) {
  if (typeof input === 'string') {
    if (input === '') throw new RangeError('LocalModel: `prompt` must not be empty');
    return { text: input };
  }
  if (Array.isArray(input)) {
    return { messages: input.map(normalizeMessage) };
  }
  throw new TypeError('LocalModel: the prompt must be a string or an array of chat messages');
}

function normalizeMessage(message) {
  const own = ownProps(message, MESSAGE_KEYS, 'message');
  if (typeof own.role !== 'string' || typeof own.content !== 'string') {
    throw new TypeError('LocalModel: each message needs a string `role` and `content`');
  }
  return { role: own.role, content: own.content };
}
