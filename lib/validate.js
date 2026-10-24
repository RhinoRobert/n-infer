// Small type-checking helpers shared by config.js and options.js.

/**
 * Copies the *own* enumerable properties of `value` that are in `known`, rejecting anything else.
 * Rejecting unknown keys turns typos (`gpulayers`) into errors instead of silent no-ops.
 *
 * @param {unknown} value
 * @param {Set<string>} known
 * @param {string} label  what is being checked ("config", "options", "message")
 * @returns {Record<string, unknown>}
 */
export function ownProps(value, known, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`LocalModel: ${label} must be an object`);
  }
  const copy = Object.create(null);
  for (const key of Object.keys(value)) {
    if (!known.has(key)) throw new TypeError(`LocalModel: unknown ${label} field \`${key}\``);
    copy[key] = value[key];
  }
  return copy;
}

export function assertInteger(name, value) {
  if (!Number.isInteger(value)) throw new TypeError(`LocalModel: \`${name}\` must be an integer`);
  return value;
}

export function assertNumber(name, value) {
  if (typeof value !== 'number') throw new TypeError(`LocalModel: \`${name}\` must be a number`);
  return value;
}

export function assertBoolean(name, value) {
  if (typeof value !== 'boolean') throw new TypeError(`LocalModel: \`${name}\` must be a boolean`);
  return value;
}
