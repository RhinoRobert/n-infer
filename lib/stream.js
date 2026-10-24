// Adapts the native `callback(error, chunk)` protocol into async iteration with AbortSignal support.

/** The error thrown when a caller's AbortSignal fires. */
function abortError(signal) {
  return signal.reason ?? new DOMException('The operation was aborted', 'AbortError');
}

/**
 * Yields text chunks as the model produces them.
 *
 * Breaking out of the loop, or aborting `signal`, stops generation.
 *
 * @param {(callback: (error: Error | null, chunk: string | null) => void) => { cancel(): void }} start
 *   Starts a native run. `chunk === null` means it finished.
 * @param {AbortSignal} [signal]
 */
export async function* toAsyncIterator(start, signal) {
  if (signal?.aborted) throw abortError(signal);

  const queue = [];
  let finished = false;
  let failure = null;
  let wake = null;

  const notify = () => {
    wake?.();
    wake = null;
  };

  const handle = start((error, chunk) => {
    if (error) failure = error;
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
      if (queue.length > 0) yield queue.shift();
      else if (failure) throw failure;
      else if (finished) return;
      else await new Promise((resolve) => (wake = resolve));
    }
  } finally {
    signal?.removeEventListener('abort', onAbort);
    // Reached early when the consumer breaks out of `for await`, or on failure.
    if (!finished) handle.cancel();
  }
}

/** Runs a native generation to completion and returns all of its text. */
export async function collect(start, signal) {
  let text = '';
  for await (const chunk of toAsyncIterator(start, signal)) text += chunk;
  return text;
}
