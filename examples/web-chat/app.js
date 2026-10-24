// The chat page. Model text is only ever inserted with textContent, never as HTML.
const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const history = []; // { role, content } for the conversation so far
let info = null;
let generating = null; // AbortController while a reply is streaming

const SUGGESTIONS = [
  'Explain how a hash map works, in two short paragraphs.',
  'Write a JavaScript function that debounces another function.',
  'Give me three names for a coffee shop run by robots.',
  'What are the trade-offs between SQL and NoSQL?',
];

// ---- machine and model panels ---------------------------------------------------------------

async function refresh() {
  info = await (await fetch('/api/info')).json();
  renderRuntime();
  renderMachine();
  renderModels();
  updateControls();
}

function renderRuntime() {
  const ready = info.current && !info.loading;
  const pill = $('runtime');
  pill.className = `pill ${info.loading ? 'busy' : ready ? 'ok' : ''}`;
  pill.replaceChildren(
    el('span', 'dot'),
    document.createTextNode(
      info.loading
        ? 'loading model…'
        : ready
          ? `${info.current.name} · ${info.backend.toUpperCase()}`
          : 'no model loaded',
    ),
  );
  $('process').textContent = `runs inside Node ${info.node} · PID ${info.pid} · ${info.rssMb} MB`;
}

function renderMachine() {
  const rows = [['Engine', `${info.backend.toUpperCase()} build`]];
  const gpu = info.devices[0];
  if (gpu) {
    const used = gpu.totalMib - gpu.freeMib;
    rows.push(
      ['GPU', gpu.name],
      ['VRAM', `${(used / 1024).toFixed(1)} / ${(gpu.totalMib / 1024).toFixed(1)} GB used`],
    );
  }
  rows.push(
    ['CPU', `${info.cpu.cores} cores / ${info.cpu.threads} threads`],
    ['RAM', `${info.cpu.ramGb} GB`],
    ['Context', `${info.contextSize} tokens`],
  );

  const card = $('machine');
  card.replaceChildren(
    ...rows.map(([label, value]) => {
      const row = el('div', 'row');
      row.append(el('span', '', label), el('span', '', value));
      return row;
    }),
  );
  if (gpu) {
    const bar = el('div', 'bar');
    const fill = el('i');
    fill.style.width = `${Math.round(((gpu.totalMib - gpu.freeMib) / gpu.totalMib) * 100)}%`;
    bar.append(fill);
    card.append(bar);
  }
}

function renderModels() {
  const select = $('model');
  const chosen = select.value || info.current?.id;
  select.replaceChildren(
    ...info.models.map((model) => {
      const gb = (model.bytes / 2 ** 30).toFixed(1);
      const tag = model.textOnly ? ' · vision model, text only' : '';
      const option = el('option', '', `${model.name}  (${gb} GB${tag})`);
      option.value = model.id;
      return option;
    }),
  );
  if (chosen) select.value = chosen;
  if (info.models.length === 0) {
    $('modelStatus').textContent =
      'No .gguf models found. Start with --model <file> or --models-dir <folder>.';
  }
}

async function loadSelected() {
  $('modelStatus').textContent = 'Loading… (the first time reads the whole file from disk)';
  $('load').disabled = true;
  history.length = 0;
  renderLog();
  info.loading = true;
  renderRuntime();
  const response = await fetch('/api/load', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ id: $('model').value }),
  });
  const result = await response.json();
  $('modelStatus').textContent = response.ok
    ? `Loaded in ${(result.loadMs / 1000).toFixed(1)} s. It stays in memory, so replies start instantly.`
    : `Could not load: ${result.error}`;
  await refresh();
}

// ---- conversation ----------------------------------------------------------------------------

function renderLog() {
  const log = $('log');
  log.replaceChildren();
  if (history.length > 0) return;

  const empty = el('div', 'empty');
  empty.append(el('h1', '', 'A language model, running in this process'));
  empty.append(
    el(
      'p',
      '',
      'Everything you type is answered by a model loaded directly by n-infer. Nothing leaves this computer, and no other program is involved.',
    ),
  );
  const chips = el('div', 'chips');
  for (const text of SUGGESTIONS) {
    const chip = el('button', '', text);
    chip.type = 'button';
    chip.addEventListener('click', () => send(text));
    chips.append(chip);
  }
  empty.append(chips);
  log.append(empty);
}

/** Renders text with ```code fences``` and `inline code`, using DOM nodes (never innerHTML). */
function renderRich(container, text) {
  container.replaceChildren();
  const parts = text.split(/```/);
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      const code = part.replace(/^[a-zA-Z0-9+#-]*\n/, '');
      container.append(el('pre', '', code.replace(/\n$/, '')));
      return;
    }
    // The newline that separates prose from a code fence is layout, not content: drop the one
    // after a closing fence (index > 0) and the one before an opening fence (not the last part).
    let prose = part;
    if (index > 0) prose = prose.replace(/^\n/, '');
    if (index < parts.length - 1) prose = prose.replace(/\n$/, '');
    prose.split(/(`[^`\n]+`)/).forEach((piece) => {
      if (/^`[^`\n]+`$/.test(piece)) container.append(el('code', '', piece.slice(1, -1)));
      else if (piece) container.append(document.createTextNode(piece));
    });
  });
}

function addMessage(role) {
  const wrapper = el('div', `msg ${role}`);
  const bubble = el('div', 'bubble');
  const stats = el('div', 'stats');
  wrapper.append(bubble, stats);
  const log = $('log');
  log.querySelector('.empty')?.remove();
  log.append(wrapper);
  log.scrollTop = log.scrollHeight;
  return { wrapper, bubble, stats };
}

function showStats(node, items) {
  node.replaceChildren(
    ...items.filter(Boolean).map(([text, fast]) => el('span', fast ? 'fast' : '', text)),
  );
}

async function send(text) {
  if (generating || !info?.current || info.loading) return;
  const content = text.trim();
  if (!content) return;

  $('input').value = '';
  autosize();
  history.push({ role: 'user', content });
  addMessage('user').bubble.textContent = content;

  const reply = addMessage('assistant');
  reply.bubble.classList.add('caret');
  const system = $('system').value.trim();
  const messages = [...(system ? [{ role: 'system', content: system }] : []), ...history];

  generating = new AbortController();
  updateControls();
  let answer = '';
  let pieces = 0;
  let firstAt = 0;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      signal: generating.signal,
      body: JSON.stringify({
        messages,
        temperature: Number($('temp').value),
        maxTokens: Number($('maxTokens').value),
      }),
    });
    if (!response.ok) throw new Error((await response.json()).error);

    for await (const event of events(response)) {
      if (event.error) throw new Error(event.error);
      if (event.start) {
        showStats(reply.stats, [[`${event.start.promptTokens} prompt tokens`]]);
        if (event.start.dropped)
          reply.wrapper.append(
            el(
              'div',
              'note',
              `Older messages were dropped to fit the ${info.contextSize}-token context window.`,
            ),
          );
      }
      if (event.t) {
        firstAt ||= performance.now();
        pieces += 1;
        answer += event.t;
        renderRich(reply.bubble, answer);
        reply.bubble.classList.add('caret');
        const seconds = (performance.now() - firstAt) / 1000;
        showStats(reply.stats, [
          [`${pieces} tokens`],
          pieces > 3 && seconds > 0.2 ? [`${(pieces / seconds).toFixed(0)} tokens/s`, true] : null,
        ]);
        $('log').scrollTop = $('log').scrollHeight;
      }
      if (event.done) {
        const d = event.done;
        showStats(reply.stats, [
          [`${d.tokens} tokens`],
          d.tokensPerSec ? [`${d.tokensPerSec} tokens/s`, true] : null,
          d.ttftMs !== null ? [`first token in ${d.ttftMs} ms`] : null,
          [`${(d.totalMs / 1000).toFixed(1)} s total`],
        ]);
      }
    }
  } catch (error) {
    if (error.name === 'AbortError') showStats(reply.stats, [[`stopped after ${pieces} tokens`]]);
    else reply.wrapper.append(el('div', 'note', `Error: ${error.message}`));
  } finally {
    reply.bubble.classList.remove('caret');
    if (answer) history.push({ role: 'assistant', content: answer });
    else if (history.at(-1)?.role === 'user') history.pop();
    generating = null;
    updateControls();
    refresh().catch(() => {});
    $('input').focus();
  }
}

/** Parses the server's event stream (`data: {...}` lines). */
async function* events(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf('\n\n')) !== -1) {
      const line = buffer.slice(0, end).replace(/^data: /, '');
      buffer = buffer.slice(end + 2);
      if (line) yield JSON.parse(line);
    }
  }
}

// ---- controls --------------------------------------------------------------------------------

function updateControls() {
  const ready = Boolean(info?.current) && !info?.loading;
  $('send').hidden = Boolean(generating);
  $('stop').hidden = !generating;
  $('send').disabled = !ready;
  $('input').disabled = !ready;
  $('input').placeholder = ready
    ? 'Ask something…  (Enter to send, Shift+Enter for a new line)'
    : 'Load a model to start chatting';
  $('load').disabled = Boolean(generating) || Boolean(info?.loading) || !info?.models.length;
  for (const chip of document.querySelectorAll('.chips button'))
    chip.disabled = !ready || Boolean(generating);
}

function autosize() {
  const box = $('input');
  box.style.height = 'auto';
  box.style.height = `${Math.min(box.scrollHeight, 180)}px`;
}

$('composer').addEventListener('submit', (event) => {
  event.preventDefault();
  send($('input').value);
});
$('input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    send($('input').value);
  }
});
$('input').addEventListener('input', autosize);
$('stop').addEventListener('click', () => generating?.abort());
$('load').addEventListener('click', loadSelected);
$('clear').addEventListener('click', () => {
  generating?.abort();
  history.length = 0;
  renderLog();
  updateControls();
});
$('temp').addEventListener('input', () => ($('tempValue').textContent = $('temp').value));
$('maxTokens').addEventListener('input', () => ($('maxValue').textContent = $('maxTokens').value));

renderLog();
refresh().catch(() => ($('runtime').textContent = 'cannot reach the server'));
setInterval(() => refresh().catch(() => {}), 3000);
