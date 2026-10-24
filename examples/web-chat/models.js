// Finds GGUF models already on this machine, without running anything.
//
// - Ollama's model folder: its manifests give each blob a readable name (read only: nothing is
//   modified, and Ollama does not need to be running).
// - Any folder you pass with --models-dir, and the small test models in test/fixtures.
import { createHash } from 'node:crypto';
import {
  closeSync,
  existsSync,
  openSync,
  readFileSync,
  readSync,
  readdirSync,
  statSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { basename, join } from 'node:path';

const MIN_BYTES = 20 * 1024 * 1024; // skips tiny files such as vision projectors and tokenizers-only
const VISION = /vision|vl\b|llava|mmproj/i;

/** @returns {Array<{ id: string, name: string, path: string, bytes: number, source: string, textOnly: boolean }>} */
export function discoverModels(extraDirs = []) {
  const found = new Map();
  const add = (model) => {
    if (found.has(model.path) || !isGguf(model.path)) return;
    found.set(model.path, {
      ...model,
      id: createHash('sha256').update(model.path).digest('hex').slice(0, 12),
      textOnly: VISION.test(model.name),
    });
  };

  for (const model of ollamaModels()) add(model);
  for (const dir of extraDirs) for (const model of folderModels(dir, 'folder')) add(model);
  return [...found.values()].sort((a, b) => a.bytes - b.bytes);
}

function ollamaModels() {
  const root = process.env.OLLAMA_MODELS ?? join(homedir(), '.ollama', 'models');
  const manifests = join(root, 'manifests');
  if (!existsSync(manifests)) return [];

  const models = [];
  for (const file of walk(manifests)) {
    try {
      const manifest = JSON.parse(readFileSync(file, 'utf8'));
      const layer = manifest.layers?.find(
        (l) => l.mediaType === 'application/vnd.ollama.image.model',
      );
      if (!layer) continue;
      const path = join(root, 'blobs', layer.digest.replace(':', '-'));
      if (!existsSync(path)) continue;
      models.push({ name: ollamaName(file, manifests), path, bytes: layer.size, source: 'Ollama' });
    } catch {
      // a manifest we cannot read is simply not offered
    }
  }
  return models;
}

/** `registry.ollama.ai/library/qwen2.5vl/7b` -> `qwen2.5vl:7b`. */
function ollamaName(file, manifests) {
  const parts = file.slice(manifests.length + 1).split(/[\\/]/);
  const tag = parts.pop();
  const model = parts.at(-1) ?? 'model';
  const readable = /^[0-9a-f]{40,}$/.test(tag) ? `${model}:${tag.slice(0, 8)}` : `${model}:${tag}`;
  return readable;
}

function folderModels(dir, source) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => name.toLowerCase().endsWith('.gguf') && !/mmproj/i.test(name))
    .map((name) => join(dir, name))
    .map((path) => ({
      name: basename(path).replace(/\.gguf$/i, ''),
      path,
      bytes: statSync(path).size,
      source,
    }));
}

function* walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(path);
    else yield path;
  }
}

/** True for a regular file that starts with the GGUF magic bytes and is a plausible model size. */
function isGguf(path) {
  try {
    if (statSync(path).size < MIN_BYTES) return false;
    const file = openSync(path, 'r');
    try {
      const magic = Buffer.alloc(4);
      readSync(file, magic, 0, 4, 0);
      return magic.toString('latin1') === 'GGUF';
    } finally {
      closeSync(file);
    }
  } catch {
    return false;
  }
}
