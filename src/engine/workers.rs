//! A small, bounded pool of worker threads. Each worker owns one inference [`Session`] for its
//! whole life, so warm KV caches never move between threads and memory use is fixed up front.
//!
//! Every submitted job produces exactly one terminal event ([`Event::Finished`] or
//! [`Event::Failed`]) even if generation panics, so callers can never be left waiting.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{Builder, JoinHandle};

use super::session::Session;
use super::{generate, Event, Request, Shared};
use crate::error::{err, panic_error, Result};

type Sink = Box<dyn FnMut(Event) + Send>;

struct Job {
    request: Request,
    sink: Sink,
}

pub struct Workers {
    queue: Mutex<Option<SyncSender<Job>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Workers {
    /// Starts one worker per session. Up to `queue_len` jobs may wait for a free worker.
    pub fn start(shared: &Arc<Shared>, sessions: Vec<Session>, queue_len: usize) -> Result<Self> {
        let (sender, receiver) = sync_channel::<Job>(queue_len);
        let receiver = Arc::new(Mutex::new(receiver));

        let mut threads = Vec::with_capacity(sessions.len());
        for (index, session) in sessions.into_iter().enumerate() {
            let shared = Arc::clone(shared);
            let receiver = Arc::clone(&receiver);
            let thread = Builder::new()
                .name(format!("n-infer-worker-{index}"))
                .spawn(move || worker_main(&shared, session, &receiver))
                .map_err(|e| err(format!("could not start a worker thread: {e}")))?;
            threads.push(thread);
        }
        Ok(Self { queue: Mutex::new(Some(sender)), threads: Mutex::new(threads) })
    }

    /// Queues a job. Fails at once, without blocking, when the queue is full or the pool is closed.
    pub fn submit(&self, request: Request, sink: impl FnMut(Event) + Send + 'static) -> Result<()> {
        let queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(sender) = queue.as_ref() else {
            return Err(err("LocalModel has been closed"));
        };
        sender.try_send(Job { request, sink: Box::new(sink) }).map_err(|failure| match failure {
            TrySendError::Full(_) => err("too many pending requests; try again later"),
            TrySendError::Disconnected(_) => err("LocalModel has been closed"),
        })
    }

    /// Stops accepting jobs and waits for the workers to finish (in-flight jobs see `closed`).
    pub fn shutdown(&self) {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner).take();
        let threads =
            std::mem::take(&mut *self.threads.lock().unwrap_or_else(PoisonError::into_inner));
        for thread in threads {
            let _ = thread.join();
        }
    }
}

fn worker_main(shared: &Shared, session: Session, jobs: &Mutex<Receiver<Job>>) {
    let mut session = Some(session);
    loop {
        let next = jobs.lock().unwrap_or_else(PoisonError::into_inner).recv();
        match next {
            Ok(job) => handle(shared, &mut session, job),
            Err(_) => break, // the sender was dropped: the pool is shutting down
        }
    }
}

fn handle(shared: &Shared, session: &mut Option<Session>, job: Job) {
    let Job { request, mut sink } = job;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        generate::run(shared, session, &request, &mut |text| sink(Event::Text(text)))
    }));
    match outcome {
        Ok(Ok(())) => sink(Event::Finished),
        Ok(Err(error)) => sink(Event::Failed(error)),
        Err(panic) => {
            // A panic may have left the session half-updated: discard it, a fresh one is made on demand.
            *session = None;
            sink(Event::Failed(panic_error(&panic)));
        }
    }
}
