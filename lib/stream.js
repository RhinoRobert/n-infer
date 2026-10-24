// Adapts the native `callback(err, chunk)` stream into an async iterator.

/**
 * @param {(callback: (err: Error | null, chunk: string | null) => void) => { cancel(): void }} start
 *   Starts generation; `chunk === null` means finished.
 */
export async function* toAsyncIterator(start) {
  const queue = [];
  let finished = false;
  let failure = null;
  let wake = null;

  const handle = start((err, chunk) => {
    if (err) failure = err;
    else if (chunk === null) finished = true;
    else queue.push(chunk);
    wake?.();
    wake = null;
  });

  try {
    for (;;) {
      if (queue.length) yield queue.shift();
      else if (failure) throw failure;
      else if (finished) return;
      else await new Promise((resolve) => (wake = resolve));
    }
  } finally {
    // Reached early when the consumer breaks out of `for await`.
    if (!finished && !failure) handle.cancel();
  }
}
