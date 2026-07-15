use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self as std_mpsc, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    },
    thread,
    time::{Duration, Instant},
};

use futures_util::future::join_all;
use tokio::sync::{Notify, oneshot};

use crate::{
    BackendError, BackendInfo, TranslateError, TranslationBackend, TranslationInput,
    TranslationOutput,
};

const MAX_BATCH_SIZE: usize = 256;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub queue_capacity: usize,
    pub max_batch_size: usize,
    /// Maximum `batch_size * source length bucket` admitted to one batch.
    pub max_padded_source_chars: usize,
    pub batch_window: Duration,
    pub enqueue_timeout: Duration,
    pub request_timeout: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 256,
            max_batch_size: 16,
            max_padded_source_chars: 4_096,
            batch_window: Duration::from_micros(1_500),
            enqueue_timeout: Duration::from_millis(10),
            request_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Default)]
pub struct SchedulerStats {
    accepted: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    timed_out: AtomicU64,
    rejected: AtomicU64,
    batches: AtomicU64,
    batch_items: AtomicU64,
    queue_depth: AtomicUsize,
    in_flight: AtomicUsize,
    largest_batch: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatsSnapshot {
    pub accepted: u64,
    pub completed: u64,
    pub failed: u64,
    pub timed_out: u64,
    pub rejected: u64,
    pub batches: u64,
    pub batch_items: u64,
    pub queue_depth: usize,
    pub in_flight: usize,
    pub largest_batch: usize,
}

impl SchedulerStats {
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            accepted: self.accepted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            timed_out: self.timed_out.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            batch_items: self.batch_items.load(Ordering::Relaxed),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            largest_batch: self.largest_batch.load(Ordering::Relaxed),
        }
    }

    pub fn prometheus(&self) -> String {
        let s = self.snapshot();
        format!(
            concat!(
                "# TYPE marian_edge_requests_total counter\n",
                "marian_edge_requests_total{{status=\"accepted\"}} {}\n",
                "marian_edge_requests_total{{status=\"completed\"}} {}\n",
                "marian_edge_requests_total{{status=\"failed\"}} {}\n",
                "marian_edge_requests_total{{status=\"timed_out\"}} {}\n",
                "marian_edge_requests_total{{status=\"rejected\"}} {}\n",
                "# TYPE marian_edge_batches_total counter\n",
                "marian_edge_batches_total {}\n",
                "# TYPE marian_edge_batch_items_total counter\n",
                "marian_edge_batch_items_total {}\n",
                "# TYPE marian_edge_queue_depth gauge\n",
                "marian_edge_queue_depth {}\n",
                "# TYPE marian_edge_in_flight gauge\n",
                "marian_edge_in_flight {}\n",
                "# TYPE marian_edge_largest_batch gauge\n",
                "marian_edge_largest_batch {}\n",
            ),
            s.accepted,
            s.completed,
            s.failed,
            s.timed_out,
            s.rejected,
            s.batches,
            s.batch_items,
            s.queue_depth,
            s.in_flight,
            s.largest_batch,
        )
    }
}

struct Job {
    input: TranslationInput,
    reply: oneshot::Sender<Result<TranslationOutput, BackendError>>,
}

enum Command {
    Translate(Job),
    Shutdown(oneshot::Sender<()>),
}

struct Inner {
    tx: SyncSender<Command>,
    queue_available: Arc<Notify>,
    stats: Arc<SchedulerStats>,
    closed: AtomicBool,
    worker_stopped: Arc<AtomicBool>,
    request_timeout: Duration,
    enqueue_timeout: Duration,
    info: BackendInfo,
}

#[derive(Clone)]
pub struct Translator {
    inner: Arc<Inner>,
}

impl Translator {
    /// Starts the worker and constructs the backend on that worker's thread.
    pub fn start<B, F>(config: SchedulerConfig, factory: F) -> Result<Self, BackendError>
    where
        B: TranslationBackend,
        F: FnOnce() -> Result<B, BackendError> + Send + 'static,
    {
        if config.queue_capacity == 0
            || config.max_batch_size == 0
            || config.max_padded_source_chars == 0
        {
            return Err(BackendError::InvalidInput(
                "queue_capacity, max_batch_size, and max_padded_source_chars must be greater than zero"
                    .into(),
            ));
        }
        if config.max_batch_size > MAX_BATCH_SIZE {
            return Err(BackendError::InvalidInput(format!(
                "max_batch_size may not exceed {MAX_BATCH_SIZE}"
            )));
        }

        let (tx, rx) = std_mpsc::sync_channel(config.queue_capacity);
        let queue_available = Arc::new(Notify::new());
        let worker_queue_available = Arc::clone(&queue_available);
        let stats = Arc::new(SchedulerStats::default());
        let worker_stats = Arc::clone(&stats);
        let worker_stopped = Arc::new(AtomicBool::new(false));
        let worker_stopped_on_thread = Arc::clone(&worker_stopped);
        let (started_tx, started_rx) = std_mpsc::sync_channel(1);
        let worker_config = config.clone();

        thread::Builder::new()
            .name("marian-edge-worker".into())
            .spawn(move || {
                let _guard = WorkerStoppedGuard {
                    stopped: Arc::clone(&worker_stopped_on_thread),
                    queue_available: Arc::clone(&worker_queue_available),
                };
                match factory() {
                    Ok(mut backend) => {
                        let info = backend.info();
                        if started_tx.send(Ok(info)).is_ok() {
                            worker_loop(
                                &mut backend,
                                rx,
                                &worker_config,
                                &worker_stats,
                                &worker_stopped_on_thread,
                                &worker_queue_available,
                            );
                        }
                    }
                    Err(error) => {
                        let _ = started_tx.send(Err(error));
                    }
                }
            })
            .map_err(|error| BackendError::Model(format!("failed to start worker: {error}")))?;

        let info = started_rx
            .recv()
            .map_err(|_| BackendError::Model("worker stopped during initialization".into()))??;

        Ok(Self {
            inner: Arc::new(Inner {
                tx,
                queue_available,
                stats,
                closed: AtomicBool::new(false),
                worker_stopped,
                request_timeout: config.request_timeout,
                enqueue_timeout: config.enqueue_timeout,
                info,
            }),
        })
    }

    pub fn backend_info(&self) -> &BackendInfo {
        &self.inner.info
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.inner.stats
    }

    pub fn is_ready(&self) -> bool {
        !self.inner.closed.load(Ordering::Acquire)
            && !self.inner.worker_stopped.load(Ordering::Acquire)
    }

    pub async fn translate(
        &self,
        input: TranslationInput,
    ) -> Result<TranslationOutput, TranslateError> {
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(TranslateError::ShuttingDown);
        }
        if self.inner.worker_stopped.load(Ordering::Acquire) {
            return Err(TranslateError::WorkerStopped);
        }

        let (reply, response) = oneshot::channel();
        // Publish the gauge first so a very fast worker cannot subtract the
        // job before the submitting task has accounted for it. A sender that
        // is waiting for queue capacity is useful admission-pressure signal.
        self.inner.stats.queue_depth.fetch_add(1, Ordering::Relaxed);
        match send_command(
            &self.inner.tx,
            &self.inner.queue_available,
            &self.inner.worker_stopped,
            Command::Translate(Job { input, reply }),
            self.inner.enqueue_timeout,
        )
        .await
        {
            Ok(()) => {
                self.inner.stats.accepted.fetch_add(1, Ordering::Relaxed);
            }
            Err(EnqueueError::Disconnected) => {
                self.inner.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                return Err(TranslateError::WorkerStopped);
            }
            Err(EnqueueError::Full) => {
                self.inner.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                self.inner.stats.rejected.fetch_add(1, Ordering::Relaxed);
                return Err(TranslateError::QueueFull);
            }
        }

        match tokio::time::timeout(self.inner.request_timeout, response).await {
            Ok(Ok(Ok(output))) => Ok(output),
            Ok(Ok(Err(error))) => Err(TranslateError::Backend(error)),
            Ok(Err(_)) => Err(TranslateError::WorkerStopped),
            Err(_) => {
                self.inner.stats.timed_out.fetch_add(1, Ordering::Relaxed);
                Err(TranslateError::Timeout(self.inner.request_timeout))
            }
        }
    }

    /// Submits a bounded logical group using the scheduler's canonical batch
    /// key, then restores caller order. Protocol layers should use this rather
    /// than duplicating source-length bucketing policy.
    pub async fn translate_many(
        &self,
        inputs: Vec<TranslationInput>,
    ) -> Result<Vec<TranslationOutput>, TranslateError> {
        let mut indexed = inputs.into_iter().enumerate().collect::<Vec<_>>();
        indexed.sort_by(|(_, left), (_, right)| left.batch_key().cmp(&right.batch_key()));
        let tasks = indexed.into_iter().map(|(index, input)| {
            let translator = self.clone();
            async move {
                translator
                    .translate(input)
                    .await
                    .map(|output| (index, output))
            }
        });
        let results = join_all(tasks).await;
        let mut ordered = std::iter::repeat_with(|| None)
            .take(results.len())
            .collect::<Vec<_>>();
        for result in results {
            let (index, output) = result?;
            ordered[index] = Some(output);
        }
        Ok(ordered
            .into_iter()
            .map(|output| output.expect("every translate_many task returned its input index"))
            .collect())
    }

    pub async fn shutdown(&self) {
        if self.inner.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let (done, wait) = oneshot::channel();
        if send_command(
            &self.inner.tx,
            &self.inner.queue_available,
            &self.inner.worker_stopped,
            Command::Shutdown(done),
            Duration::from_secs(5),
        )
        .await
        .is_ok()
        {
            let _ = tokio::time::timeout(Duration::from_secs(5), wait).await;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnqueueError {
    Full,
    Disconnected,
}

async fn send_command(
    tx: &SyncSender<Command>,
    queue_available: &Notify,
    worker_stopped: &AtomicBool,
    mut command: Command,
    timeout: Duration,
) -> Result<(), EnqueueError> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        // Register before probing the bounded channel so a burst of worker
        // dequeues cannot collapse into a single stored Notify permit.
        let notified = queue_available.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        match tx.try_send(command) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Disconnected(_)) => return Err(EnqueueError::Disconnected),
            Err(TrySendError::Full(returned)) => command = returned,
        }

        if worker_stopped.load(Ordering::Acquire) {
            return Err(EnqueueError::Disconnected);
        }
        if tokio::time::timeout_at(deadline, notified).await.is_err() {
            return Err(EnqueueError::Full);
        }
    }
}

struct WorkerStoppedGuard {
    stopped: Arc<AtomicBool>,
    queue_available: Arc<Notify>,
}

impl Drop for WorkerStoppedGuard {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        self.queue_available.notify_waiters();
    }
}

fn worker_loop<B: TranslationBackend>(
    backend: &mut B,
    rx: Receiver<Command>,
    config: &SchedulerConfig,
    stats: &SchedulerStats,
    worker_stopped: &AtomicBool,
    queue_available: &Notify,
) {
    let mut deferred = VecDeque::new();
    loop {
        let command = deferred.pop_front().or_else(|| {
            loop {
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(command) => {
                        queue_available.notify_one();
                        break Some(command);
                    }
                    Err(RecvTimeoutError::Timeout) if !backend.is_ready() => break None,
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break None,
                }
            }
        });
        let Some(command) = command else {
            if !backend.is_ready() {
                worker_stopped.store(true, Ordering::Release);
                fail_pending(
                    &rx,
                    &mut deferred,
                    stats,
                    "translation worker stopped unexpectedly",
                );
            }
            break;
        };
        let first = match command {
            Command::Translate(job) => job,
            Command::Shutdown(done) => {
                fail_pending(&rx, &mut deferred, stats, "service is shutting down");
                let _ = done.send(());
                break;
            }
        };
        stats.queue_depth.fetch_sub(1, Ordering::Relaxed);

        if first.reply.is_closed() {
            continue;
        }

        let key = first.input.batch_key();
        let key = (key.0.to_owned(), key.1.to_owned(), key.2, key.3);
        let batch_limit = config
            .max_batch_size
            .min((config.max_padded_source_chars / key.3).max(1));
        let mut jobs = Vec::with_capacity(batch_limit);
        jobs.push(first);
        let deadline = Instant::now() + config.batch_window;

        while jobs.len() < batch_limit {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match rx.recv_timeout(deadline.saturating_duration_since(now)) {
                Ok(Command::Translate(job)) => {
                    queue_available.notify_one();
                    if job.reply.is_closed() {
                        stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                        continue;
                    }
                    let job_key = job.input.batch_key();
                    if job_key == (key.0.as_str(), key.1.as_str(), key.2, key.3) {
                        stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                        jobs.push(job);
                    } else {
                        deferred.push_back(Command::Translate(job));
                        // Stop at the first incompatible request. Continuing
                        // to drain the bounded channel into `deferred` would
                        // turn it into an unbounded shadow queue and release
                        // admission slots for new senders.
                        break;
                    }
                }
                Ok(Command::Shutdown(done)) => {
                    queue_available.notify_one();
                    deferred.push_front(Command::Shutdown(done));
                    break;
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        let batch_size = jobs.len();
        stats.batches.fetch_add(1, Ordering::Relaxed);
        stats
            .batch_items
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        stats.in_flight.fetch_add(batch_size, Ordering::Relaxed);
        stats.largest_batch.fetch_max(batch_size, Ordering::Relaxed);

        // Coalesce byte-for-byte identical logical requests inside this batch.
        // Repetition counts let a backend restore physical rows for occupancy
        // without leaking device geometry into the scheduler.
        let mut unique_inputs = Vec::with_capacity(jobs.len());
        let mut unique_by_input = HashMap::with_capacity(jobs.len());
        let mut repetitions = Vec::with_capacity(jobs.len());
        let mut output_indices = Vec::with_capacity(jobs.len());
        for job in &jobs {
            let index = if let Some(&index) = unique_by_input.get(&job.input) {
                repetitions[index] += 1;
                index
            } else {
                let index = unique_inputs.len();
                unique_inputs.push(job.input.clone());
                repetitions.push(1_usize);
                unique_by_input.insert(job.input.clone(), index);
                index
            };
            output_indices.push(index);
        }
        let result = backend.translate_batch_with_repetitions(&unique_inputs, &repetitions);
        let backend_ready = backend.is_ready();
        if !backend_ready {
            // Publish the failed state before replying or draining so new
            // callers cannot race another job into a terminal backend.
            worker_stopped.store(true, Ordering::Release);
        }
        match result {
            Ok(outputs) if outputs.len() == unique_inputs.len() => {
                for (job, output_index) in jobs.into_iter().zip(output_indices) {
                    let output = outputs[output_index].clone();
                    if job.reply.send(Ok(output)).is_ok() {
                        stats.completed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Ok(outputs) => {
                let error = BackendError::Inference(format!(
                    "backend returned {} outputs for {} inputs",
                    outputs.len(),
                    unique_inputs.len()
                ));
                fail_jobs(jobs, error, stats);
            }
            Err(error) => fail_jobs(jobs, error, stats),
        }
        stats.in_flight.fetch_sub(batch_size, Ordering::Relaxed);
        if !backend_ready {
            fail_pending(
                &rx,
                &mut deferred,
                stats,
                "translation worker stopped unexpectedly",
            );
            break;
        }
    }
}

fn fail_jobs(jobs: Vec<Job>, error: BackendError, stats: &SchedulerStats) {
    for job in jobs {
        if job.reply.send(Err(error.clone())).is_ok() {
            stats.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn fail_pending(
    rx: &Receiver<Command>,
    deferred: &mut VecDeque<Command>,
    stats: &SchedulerStats,
    message: &str,
) {
    deferred.extend(rx.try_iter());
    while let Some(command) = deferred.pop_front() {
        if let Command::Translate(job) = command {
            stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
            if job
                .reply
                .send(Err(BackendError::Inference(message.into())))
                .is_ok()
            {
                stats.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        thread,
        time::Duration,
    };

    use super::*;
    use crate::{BackendInfo, EchoBackend, TranslationOutput};

    struct SlowBackend {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
    }

    struct TerminalBackend {
        ready: bool,
    }

    struct SharedReadinessBackend {
        ready: Arc<AtomicBool>,
    }

    struct CountingBackend {
        inferred_inputs: Arc<AtomicUsize>,
    }

    struct PanickingBackend;

    type RepetitionRecord = Vec<(Vec<String>, Vec<usize>)>;

    struct RepetitionBackend {
        records: Arc<Mutex<RepetitionRecord>>,
    }

    impl TranslationBackend for PanickingBackend {
        fn info(&self) -> BackendInfo {
            BackendInfo {
                name: "panic-test".into(),
                device: "none".into(),
                model: "test".into(),
                precision: "n/a".into(),
                attention: None,
                supports_batching: true,
            }
        }

        fn translate_batch(
            &mut self,
            _inputs: &[TranslationInput],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            panic!("intentional backend panic")
        }
    }

    impl TranslationBackend for RepetitionBackend {
        fn info(&self) -> BackendInfo {
            BackendInfo {
                name: "repetition-test".into(),
                device: "none".into(),
                model: "test".into(),
                precision: "n/a".into(),
                attention: None,
                supports_batching: true,
            }
        }

        fn translate_batch(
            &mut self,
            inputs: &[TranslationInput],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            self.translate_batch_with_repetitions(inputs, &vec![1; inputs.len()])
        }

        fn translate_batch_with_repetitions(
            &mut self,
            inputs: &[TranslationInput],
            repetitions: &[usize],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            self.records.lock().unwrap().push((
                inputs.iter().map(|input| input.text.clone()).collect(),
                repetitions.to_vec(),
            ));
            Ok(inputs
                .iter()
                .map(|input| TranslationOutput {
                    text: input.text.clone(),
                    score: None,
                    input_tokens: 1,
                    output_tokens: 1,
                })
                .collect())
        }
    }

    impl TranslationBackend for CountingBackend {
        fn info(&self) -> BackendInfo {
            BackendInfo {
                name: "counting-test".into(),
                device: "none".into(),
                model: "test".into(),
                precision: "n/a".into(),
                attention: None,
                supports_batching: true,
            }
        }

        fn translate_batch(
            &mut self,
            inputs: &[TranslationInput],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            self.inferred_inputs
                .fetch_add(inputs.len(), Ordering::Relaxed);
            Ok(inputs
                .iter()
                .map(|input| TranslationOutput {
                    text: input.text.clone(),
                    score: None,
                    input_tokens: 1,
                    output_tokens: 1,
                })
                .collect())
        }
    }

    impl TranslationBackend for TerminalBackend {
        fn info(&self) -> BackendInfo {
            BackendInfo {
                name: "terminal-test".into(),
                device: "none".into(),
                model: "test".into(),
                precision: "n/a".into(),
                attention: None,
                supports_batching: true,
            }
        }

        fn is_ready(&self) -> bool {
            self.ready
        }

        fn translate_batch(
            &mut self,
            _inputs: &[TranslationInput],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            self.ready = false;
            Err(BackendError::Inference("transport closed".into()))
        }
    }

    impl TranslationBackend for SharedReadinessBackend {
        fn info(&self) -> BackendInfo {
            BackendInfo {
                name: "shared-readiness-test".into(),
                device: "none".into(),
                model: "test".into(),
                precision: "n/a".into(),
                attention: None,
                supports_batching: true,
            }
        }

        fn is_ready(&self) -> bool {
            self.ready.load(Ordering::Acquire)
        }

        fn translate_batch(
            &mut self,
            inputs: &[TranslationInput],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            Ok(inputs
                .iter()
                .map(|input| TranslationOutput {
                    text: input.text.clone(),
                    score: None,
                    input_tokens: 1,
                    output_tokens: 1,
                })
                .collect())
        }
    }

    impl TranslationBackend for SlowBackend {
        fn info(&self) -> BackendInfo {
            BackendInfo {
                name: "slow-test".into(),
                device: "none".into(),
                model: "test".into(),
                precision: "n/a".into(),
                attention: None,
                supports_batching: true,
            }
        }

        fn translate_batch(
            &mut self,
            inputs: &[TranslationInput],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            self.started.store(true, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(1));
            }
            Ok(inputs
                .iter()
                .map(|input| TranslationOutput {
                    text: input.text.clone(),
                    score: None,
                    input_tokens: 1,
                    output_tokens: 1,
                })
                .collect())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_are_dynamically_batched() {
        let translator = Translator::start(
            SchedulerConfig {
                max_batch_size: 8,
                batch_window: Duration::from_millis(10),
                ..SchedulerConfig::default()
            },
            || Ok(EchoBackend),
        )
        .unwrap();
        let translator = Arc::new(translator);

        let mut tasks = Vec::new();
        for n in 0..24 {
            let translator = Arc::clone(&translator);
            tasks.push(tokio::spawn(async move {
                let text = format!("request-{n}");
                let output = translator
                    .translate(TranslationInput::new(text.clone(), "en", "zh"))
                    .await
                    .unwrap();
                assert_eq!(output.text, text);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }

        let stats = translator.stats().snapshot();
        assert_eq!(stats.completed, 24);
        assert!(stats.batches < 24);
        assert!(stats.largest_batch > 1);
        translator.shutdown().await;
        assert!(!translator.is_ready());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn identical_requests_are_inferred_once_per_dynamic_batch() {
        let inferred_inputs = Arc::new(AtomicUsize::new(0));
        let backend_counter = Arc::clone(&inferred_inputs);
        let translator = Arc::new(
            Translator::start(
                SchedulerConfig {
                    max_batch_size: 8,
                    batch_window: Duration::from_millis(50),
                    ..SchedulerConfig::default()
                },
                move || {
                    Ok(CountingBackend {
                        inferred_inputs: backend_counter,
                    })
                },
            )
            .unwrap(),
        );

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let translator = Arc::clone(&translator);
            tasks.push(tokio::spawn(async move {
                translator
                    .translate(TranslationInput::new("same", "en", "zh"))
                    .await
                    .unwrap()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap().text, "same");
        }
        assert_eq!(translator.stats().snapshot().largest_batch, 8);
        assert_eq!(inferred_inputs.load(Ordering::Relaxed), 1);
        translator.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_reports_interleaved_repetitions_without_device_policy() {
        let records = Arc::new(Mutex::new(Vec::new()));
        let backend_records = Arc::clone(&records);
        let translator = Arc::new(
            Translator::start(
                SchedulerConfig {
                    max_batch_size: 8,
                    batch_window: Duration::from_millis(50),
                    ..SchedulerConfig::default()
                },
                move || {
                    Ok(RepetitionBackend {
                        records: backend_records,
                    })
                },
            )
            .unwrap(),
        );
        let texts = ["a", "b", "a", "c", "b", "a", "c", "c"];
        let tasks = texts.map(|text| {
            let translator = Arc::clone(&translator);
            tokio::spawn(async move {
                translator
                    .translate(TranslationInput::new(text, "en", "zh"))
                    .await
                    .unwrap()
            })
        });
        for (task, expected) in tasks.into_iter().zip(texts) {
            assert_eq!(task.await.unwrap().text, expected);
        }

        {
            let records = records.lock().unwrap();
            assert_eq!(records.len(), 1);
            let mut repetitions = records[0]
                .0
                .iter()
                .cloned()
                .zip(records[0].1.iter().copied())
                .collect::<Vec<_>>();
            repetitions.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            assert_eq!(
                repetitions,
                vec![("a".into(), 3), ("b".into(), 2), ("c".into(), 3)]
            );
        }
        translator.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn duplicate_coalescing_never_crosses_dynamic_batch_boundaries() {
        let records = Arc::new(Mutex::new(Vec::new()));
        let backend_records = Arc::clone(&records);
        let translator = Arc::new(
            Translator::start(
                SchedulerConfig {
                    max_batch_size: 4,
                    batch_window: Duration::from_millis(50),
                    ..SchedulerConfig::default()
                },
                move || {
                    Ok(RepetitionBackend {
                        records: backend_records,
                    })
                },
            )
            .unwrap(),
        );
        for _ in 0..2 {
            let tasks = (0..4)
                .map(|_| {
                    let translator = Arc::clone(&translator);
                    tokio::spawn(async move {
                        translator
                            .translate(TranslationInput::new("same", "en", "zh"))
                            .await
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            for task in tasks {
                assert_eq!(task.await.unwrap().text, "same");
            }
        }

        assert_eq!(
            records.lock().unwrap().as_slice(),
            &[
                (vec!["same".into()], vec![4]),
                (vec!["same".into()], vec![4])
            ]
        );
        translator.shutdown().await;
    }

    #[test]
    fn rejects_invalid_capacity() {
        let result = Translator::start(
            SchedulerConfig {
                queue_capacity: 0,
                ..SchedulerConfig::default()
            },
            || Ok(EchoBackend),
        );
        assert!(result.is_err());
    }

    #[test]
    fn rejects_batch_size_above_protocol_limit() {
        let result = Translator::start(
            SchedulerConfig {
                max_batch_size: MAX_BATCH_SIZE + 1,
                ..SchedulerConfig::default()
            },
            || Ok(EchoBackend),
        );
        assert!(matches!(result, Err(BackendError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn terminal_backend_failure_disables_readiness_and_admission() {
        let translator = Translator::start(SchedulerConfig::default(), || {
            Ok(TerminalBackend { ready: true })
        })
        .unwrap();

        let first = translator
            .translate(TranslationInput::new("first", "en", "zh"))
            .await;
        assert!(matches!(
            first,
            Err(TranslateError::Backend(BackendError::Inference(_)))
        ));
        assert!(!translator.is_ready());

        let accepted = translator.stats().snapshot().accepted;
        let second = translator
            .translate(TranslationInput::new("second", "en", "zh"))
            .await;
        assert!(matches!(second, Err(TranslateError::WorkerStopped)));
        assert_eq!(translator.stats().snapshot().accepted, accepted);
    }

    #[tokio::test]
    async fn backend_panic_stops_readiness_and_future_admission() {
        let translator =
            Translator::start(SchedulerConfig::default(), || Ok(PanickingBackend)).unwrap();

        let first = translator
            .translate(TranslationInput::new("panic", "en", "zh"))
            .await;
        assert!(matches!(first, Err(TranslateError::WorkerStopped)));

        tokio::time::timeout(Duration::from_secs(1), async {
            while translator.is_ready() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        let accepted = translator.stats().snapshot().accepted;
        let second = translator
            .translate(TranslationInput::new("after-panic", "en", "zh"))
            .await;
        assert!(matches!(second, Err(TranslateError::WorkerStopped)));
        assert_eq!(translator.stats().snapshot().accepted, accepted);
    }

    #[tokio::test]
    async fn idle_backend_failure_disables_readiness_without_a_request() {
        let ready = Arc::new(AtomicBool::new(true));
        let backend_ready = Arc::clone(&ready);
        let translator = Translator::start(SchedulerConfig::default(), move || {
            Ok(SharedReadinessBackend {
                ready: backend_ready,
            })
        })
        .unwrap();

        ready.store(false, Ordering::Release);
        tokio::time::timeout(Duration::from_secs(1), async {
            while translator.is_ready() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        let result = translator
            .translate(TranslationInput::new("after-stop", "en", "zh"))
            .await;
        assert!(matches!(result, Err(TranslateError::WorkerStopped)));
        assert_eq!(translator.stats().snapshot().accepted, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bounded_queue_rejects_overload_without_unbounded_waiting() {
        let started = Arc::new(AtomicBool::new(false));
        let backend_started = Arc::clone(&started);
        let release = Arc::new(AtomicBool::new(false));
        let backend_release = Arc::clone(&release);
        let translator = Translator::start(
            SchedulerConfig {
                queue_capacity: 1,
                max_batch_size: 1,
                max_padded_source_chars: 64,
                batch_window: Duration::ZERO,
                enqueue_timeout: Duration::from_millis(2),
                request_timeout: Duration::from_secs(1),
            },
            move || {
                Ok(SlowBackend {
                    started: backend_started,
                    release: backend_release,
                })
            },
        )
        .unwrap();

        let first_translator = translator.clone();
        let first = tokio::spawn(async move {
            first_translator
                .translate(TranslationInput::new("first", "en", "zh"))
                .await
        });
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }

        let second_translator = translator.clone();
        let second = tokio::spawn(async move {
            second_translator
                .translate(TranslationInput::new("second", "en", "zh"))
                .await
        });
        // `queue_depth` is published before admission, so it can also count a
        // sender that has not won the channel slot yet. Wait for the accepted
        // counter instead: while the first request is blocked in the backend,
        // this proves that the second request occupies the sole queue slot.
        while translator.stats().snapshot().accepted < 2 {
            tokio::task::yield_now().await;
        }
        let rejected = translator
            .translate(TranslationInput::new("third", "en", "zh"))
            .await;
        release.store(true, Ordering::Release);
        assert!(matches!(rejected, Err(TranslateError::QueueFull)));

        assert_eq!(first.await.unwrap().unwrap().text, "first");
        assert_eq!(second.await.unwrap().unwrap().text, "second");
        let stats = translator.stats().snapshot();
        assert_eq!(stats.completed, 2);
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.queue_depth, 0);
        translator.shutdown().await;
    }
}
