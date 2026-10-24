// Adapts the native `callback(err, chunk)` stream into an async iterator with AbortSignal support.

function abortError(signal) {
  return signal?.reason ?? new DOMException('The operation was aborted', 'AbortError');
}

/**
 * @param {(callback: (err: Error | null, chunk: string | null) => void) => { cancel(): void }} start
 *   Starts generation; `chunk === null` means finished.
 * @param {AbortSignal} [signal]
 */
export async function* toAsyncIterator(start, signal) {
  if (signal?.aborted) {
    throw abortError(signal);
  }

  const queue = [];
  let finished = false;
  let failure = null;
  let wake = null;

  const notify = () => {
    wake?.();
    wake = null;
  };

  const handle = start((err, chunk) => {
    if (err) failure = err;
    else if (chunk === null) finished = true;
    else queue.push(chunk);
    notify();
  });

  const onAbort = () => {
    failure = abortError(signal);
    handle.cancel();
    notify();
  };
  signal?.addEventListener('abort', onAbort, { once: true });

  try {
    for (;;) {
      if (signal?.aborted && failure) throw failure;
      if (queue.length) yield queue.shift();
      else if (failure) throw failure;
      else if (finished) return;
      else await new Promise((resolve) => (wake = resolve));
    }
  } finally {
    signal?.removeEventListener('abort', onAbort);
    // Reached early when the consumer breaks out of `for await` or aborts.
    if (!finished) handle.cancel();
  }
}
