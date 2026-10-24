// A local chat app, to see n-infer working. The model runs inside THIS Node.js process:
// no Ollama, no Python, no separate inference server.
//
//   node examples/web-chat/server.js                      pick a model in the page
//   node examples/web-chat/server.js --model ./model.gguf load this one at start
//
// Options: --port 3939   --models-dir <folder> (repeatable)
//
// It listens on 127.0.0.1 only and refuses requests that did not come from its own page.
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { cpus, totalmem } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { LocalModel, backend, hardware } from '../../lib/index.js';
import { discoverModels } from './models.js';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({
  options: {
    port: { type: 'string', default: '3939' },
    model: { type: 'string' },
    'models-dir': { type: 'string', multiple: true, default: [] },
  },
});
const port = Number(values.port);
const CONTEXT = 4096;
const MAX_BODY_BYTES = 256 * 1024;

const models = discoverModels([
  join(here, '..', '..', 'test', 'fixtures'),
  ...values['models-dir'],
]);
const pages = {
  '/': ['index.html', 'text/html; charset=utf-8'],
  '/app.js': ['app.js', 'text/javascript; charset=utf-8'],
};
const files = Object.fromEntries(
  Object.entries(pages).map(([url, [file, type]]) => [
    url,
    { body: readFileSync(join(here, file)), type },
  ]),
);

/** The model currently loaded, if any. Loading and generating never overlap. */
const state = { current: null, loading: false };

async function loadModel(id) {
  // Only ids the server itself discovered are accepted: a client can never name a file path.
  const info = models.find((m) => m.id === id);
  if (!info) throw httpError(400, 'unknown model');
  if (state.loading) throw httpError(409, 'a model is already loading');

  state.loading = true;
  try {
    state.current?.model.close();
    state.current = null;
    const started = performance.now();
    const model = await new LocalModel({ modelPath: info.path, contextSize: CONTEXT }).init();
    state.current = { id, name: info.name, model, loadMs: Math.round(performance.now() - started) };
    return state.current;
  } finally {
    state.loading = false;
  }
}

function snapshot() {
  const gpu = hardware();
  const memory = process.memoryUsage();
  return {
    backend,
    node: process.version,
    pid: process.pid,
    rssMb: Math.round(memory.rss / 1048576),
    cpu: {
      name: cpus()[0]?.model.trim(),
      cores: gpu.physicalCores,
      threads: gpu.logicalCores,
      ramGb: Math.round(totalmem() / 2 ** 30),
    },
    devices: gpu.devices
      .filter((d) => d.kind !== 'Cpu')
      .map(({ description, backend: kind, totalMib, freeMib }) => ({
        name: description.trim(),
        kind,
        totalMib,
        freeMib,
      })),
    models: models.map(({ id, name, bytes, source, textOnly }) => ({
      id,
      name,
      bytes,
      source,
      textOnly,
    })),
    current: state.current && {
      id: state.current.id,
      name: state.current.name,
      loadMs: state.current.loadMs,
    },
    loading: state.loading,
    contextSize: CONTEXT,
  };
}

/** Drops the oldest turns until the conversation fits the context window. */
function fitToContext(model, messages, maxTokens) {
  const trimmed = [...messages];
  let dropped = 0;
  while (trimmed.length > 1 && model.countTokens(trimmed) > CONTEXT - maxTokens - 16) {
    const oldest = trimmed.findIndex((m) => m.role !== 'system');
    if (oldest === -1 || oldest === trimmed.length - 1) break;
    trimmed.splice(oldest, 1);
    dropped += 1;
  }
  return { messages: trimmed, dropped };
}

async function chat(request, response) {
  const { model } = state.current ?? {};
  if (!model || state.loading) return json(response, 409, { error: 'no model is loaded yet' });

  const body = await readJson(request);
  const maxTokens = clamp(body.maxTokens, 16, 1024, 400);
  const temperature = clamp(body.temperature, 0, 2, 0.7);
  if (!Array.isArray(body.messages) || body.messages.length === 0 || body.messages.length > 100) {
    return json(response, 400, { error: 'messages must be a non-empty array' });
  }
  const { messages, dropped } = fitToContext(model, body.messages, maxTokens);

  response.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-store',
    connection: 'keep-alive',
  });
  const send = (event) => response.write(`data: ${JSON.stringify(event)}\n\n`);

  const stop = new AbortController();
  response.on('close', () => stop.abort()); // the browser went away or pressed Stop

  try {
    send({ start: { promptTokens: model.countTokens(messages), dropped } });
    const started = performance.now();
    let firstAt = null;
    let pieces = 0;
    for await (const piece of model.stream(messages, {
      maxTokens,
      temperature,
      signal: stop.signal,
    })) {
      firstAt ??= performance.now();
      pieces += 1;
      send({ t: piece });
    }
    const now = performance.now();
    send({
      done: {
        tokens: pieces,
        ttftMs: firstAt ? Math.round(firstAt - started) : null,
        tokensPerSec:
          firstAt && pieces > 1 ? +(((pieces - 1) / (now - firstAt)) * 1000).toFixed(1) : null,
        totalMs: Math.round(now - started),
      },
    });
  } catch (error) {
    if (error.name !== 'AbortError') send({ error: error.message });
  }
  response.end();
}

const server = createServer(async (request, response) => {
  try {
    if (!isLocalRequest(request)) return json(response, 403, { error: 'forbidden' });
    const url = new URL(request.url, 'http://localhost').pathname;

    if (request.method === 'GET' && files[url]) {
      response.writeHead(200, {
        'content-type': files[url].type,
        'content-security-policy':
          "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
        'x-content-type-options': 'nosniff',
      });
      return response.end(files[url].body);
    }
    if (request.method === 'GET' && url === '/api/info') return json(response, 200, snapshot());
    if (request.method === 'POST' && url === '/api/load') {
      const { id } = await readJson(request);
      const loaded = await loadModel(id);
      return json(response, 200, { name: loaded.name, loadMs: loaded.loadMs });
    }
    if (request.method === 'POST' && url === '/api/chat') return await chat(request, response);
    json(response, 404, { error: 'not found' });
  } catch (error) {
    if (!response.headersSent) json(response, error.status ?? 500, { error: error.message });
    else response.end();
  }
});

/** Only this machine's own page may talk to the server (blocks other sites and DNS rebinding). */
function isLocalRequest(request) {
  const allowed = [`127.0.0.1:${port}`, `localhost:${port}`];
  if (!allowed.includes(request.headers.host ?? '')) return false;
  const origin = request.headers.origin;
  return request.method === 'GET' || !origin || allowed.some((host) => origin === `http://${host}`);
}

function json(response, status, payload) {
  response.writeHead(status, { 'content-type': 'application/json', 'cache-control': 'no-store' });
  response.end(JSON.stringify(payload));
}

async function readJson(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw httpError(413, 'request too large');
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
  } catch {
    throw httpError(400, 'request body is not valid JSON');
  }
}

/** An error that carries the HTTP status to reply with. */
const httpError = (status, message) => Object.assign(new Error(message), { status });

const clamp = (value, min, max, fallback) =>
  Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;

// ---- start -----------------------------------------------------------------------------------

server.listen(port, '127.0.0.1', async () => {
  console.log(
    `n-infer chat  ->  http://127.0.0.1:${port}   (backend: ${backend}, pid ${process.pid})`,
  );
  console.log(`models found: ${models.length}   (Ctrl-C to stop)`);

  const wanted = values.model
    ? { id: models.find((m) => m.path === resolve(values.model))?.id }
    : { id: models.filter((m) => !m.textOnly).at(-1)?.id }; // the largest text model
  if (!wanted.id && values.model) {
    models.push({
      id: 'cli',
      name: values.model,
      path: resolve(values.model),
      bytes: 0,
      source: 'command line',
      textOnly: false,
    });
    wanted.id = 'cli';
  }
  if (wanted.id) {
    try {
      const loaded = await loadModel(wanted.id);
      console.log(`loaded "${loaded.name}" in ${loaded.loadMs} ms`);
    } catch (error) {
      console.error(`could not load the default model: ${error.message}`);
    }
  }
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    state.current?.model.close();
    server.close();
    process.exit(0);
  });
}
