use std::{
    fmt, fs,
    io::{self, BufReader, Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use marian_core::{
    BackendError, BackendInfo, TranslationBackend, TranslationInput, TranslationOutput,
};

const MAX_BATCH_ITEMS: usize = 256;
const MAX_FRAME_BYTES: usize = 1024 * 1024;
const WORKER_IO_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug)]
struct WorkerOutput {
    text: String,
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Debug)]
enum WorkerResponseError {
    Reported(String),
    Transport(io::Error),
}

impl fmt::Display for WorkerResponseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reported(message) => formatter.write_str(message),
            Self::Transport(error) => error.fmt(formatter),
        }
    }
}

impl From<io::Error> for WorkerResponseError {
    fn from(error: io::Error) -> Self {
        Self::Transport(error)
    }
}

struct WorkerRequest {
    texts: Vec<String>,
    reply: SyncSender<io::Result<()>>,
}

struct WorkerTransport {
    requests: Option<SyncSender<WorkerRequest>>,
    responses: Receiver<Result<Vec<WorkerOutput>, WorkerResponseError>>,
    healthy: Arc<AtomicBool>,
    response_timeout: Duration,
}

impl WorkerTransport {
    fn exchange(&mut self, texts: &[String]) -> Result<Vec<WorkerOutput>, BackendError> {
        if !self.healthy.load(Ordering::Acquire) {
            return Err(BackendError::Inference(
                "Bergamot worker transport is unavailable".into(),
            ));
        }

        let (reply, response) = mpsc::sync_channel(1);
        let request = WorkerRequest {
            texts: texts.to_owned(),
            reply,
        };
        match self
            .requests
            .as_ref()
            .map(|requests| requests.try_send(request))
        {
            Some(Ok(())) => {}
            Some(Err(TrySendError::Full(_))) => {
                return Err(self.fail("Bergamot worker writer did not accept a request"));
            }
            Some(Err(TrySendError::Disconnected(_))) | None => {
                return Err(self.fail("Bergamot worker writer stopped unexpectedly"));
            }
        }

        let deadline = Instant::now() + self.response_timeout;
        match response.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(self.fail(format!("worker request failed: {error}")));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(self.fail(format!(
                    "worker I/O timed out after {:?}",
                    self.response_timeout
                )));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(self.fail("Bergamot worker writer stopped unexpectedly"));
            }
        }

        match self
            .responses
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        {
            Ok(Ok(outputs)) => Ok(outputs),
            Ok(Err(WorkerResponseError::Reported(message))) => Err(BackendError::Inference(
                format!("worker rejected request: {message}"),
            )),
            Ok(Err(WorkerResponseError::Transport(error))) => {
                Err(self.fail(format!("worker response failed: {error}")))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Err(self.fail(format!(
                "worker I/O timed out after {:?}",
                self.response_timeout
            ))),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(self.fail("Bergamot worker reader stopped unexpectedly"))
            }
        }
    }

    fn fail(&self, message: impl Into<String>) -> BackendError {
        self.healthy.store(false, Ordering::Release);
        BackendError::Inference(message.into())
    }
}

pub struct BergamotBackend {
    child: Child,
    transport: WorkerTransport,
    writer_thread: Option<JoinHandle<()>>,
    reader_thread: Option<JoinHandle<()>>,
    source_lang: String,
    target_lang: String,
    model_id: String,
    device: String,
}

impl BergamotBackend {
    pub fn load(
        model_dir: impl AsRef<Path>,
        worker: impl AsRef<Path>,
        cpu_threads: usize,
    ) -> Result<Self, BackendError> {
        if !(1..=64).contains(&cpu_threads) {
            return Err(BackendError::InvalidInput(
                "cpu_threads must be between 1 and 64".into(),
            ));
        }
        let model_dir = fs::canonicalize(model_dir.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve model directory {}: {error}",
                model_dir.as_ref().display()
            ))
        })?;
        for name in [
            "bergamot.yml",
            "model.enzh.intgemm.alphas.bin",
            "source.spm",
            "target.spm",
            "shortlist.bin",
        ] {
            let path = model_dir.join(name);
            if !path.is_file() {
                return Err(BackendError::Model(format!(
                    "required Bergamot model file is missing: {}",
                    path.display()
                )));
            }
        }
        let worker = fs::canonicalize(worker.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve Bergamot worker {}: {error}",
                worker.as_ref().display()
            ))
        })?;
        let config = model_dir.join("bergamot.yml");
        let mut child = Command::new(&worker)
            .arg(&config)
            .arg(cpu_threads.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                BackendError::Model(format!(
                    "failed to start Bergamot worker {}: {error}",
                    worker.display()
                ))
            })?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| BackendError::Model("Bergamot worker has no stdin".into()))?;
        let output = child
            .stdout
            .take()
            .map(BufReader::new)
            .ok_or_else(|| BackendError::Model("Bergamot worker has no stdout".into()))?;

        let healthy = Arc::new(AtomicBool::new(true));
        let (responses, response_rx) = mpsc::channel();
        let reader_health = Arc::clone(&healthy);
        let reader_thread = match thread::Builder::new()
            .name("marian-bergamot-reader".into())
            .spawn(move || worker_reader_loop(output, responses, reader_health))
        {
            Ok(thread) => thread,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(BackendError::Model(format!(
                    "failed to start Bergamot reader thread: {error}"
                )));
            }
        };
        let (requests, request_rx) = mpsc::sync_channel(1);
        let writer_health = Arc::clone(&healthy);
        let writer_thread = match thread::Builder::new()
            .name("marian-bergamot-writer".into())
            .spawn(move || worker_writer_loop(input, request_rx, writer_health))
        {
            Ok(thread) => thread,
            Err(error) => {
                let _ = child.kill();
                drop(requests);
                let _ = child.wait();
                let _ = reader_thread.join();
                return Err(BackendError::Model(format!(
                    "failed to start Bergamot writer thread: {error}"
                )));
            }
        };

        let mut backend = Self {
            child,
            transport: WorkerTransport {
                requests: Some(requests),
                responses: response_rx,
                healthy,
                response_timeout: WORKER_IO_TIMEOUT,
            },
            writer_thread: Some(writer_thread),
            reader_thread: Some(reader_thread),
            source_lang: "en".into(),
            target_lang: "zh".into(),
            model_id: "mozilla-firefox-translations-en-zh-base-memory".into(),
            device: format!("Linux {} CPU", std::env::consts::ARCH),
        };
        backend.exchange(&["Hello.".to_owned()]).map_err(|error| {
            BackendError::Model(format!("Bergamot worker warmup failed: {error}"))
        })?;
        Ok(backend)
    }

    fn exchange(&mut self, texts: &[String]) -> Result<Vec<WorkerOutput>, BackendError> {
        self.transport.exchange(texts)
    }
}

impl TranslationBackend for BergamotBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "bergamot".into(),
            device: self.device.clone(),
            model: self.model_id.clone(),
            precision: "int8".into(),
            supports_batching: true,
        }
    }

    fn is_ready(&self) -> bool {
        self.transport.healthy.load(Ordering::Acquire)
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        if inputs.len() > MAX_BATCH_ITEMS {
            return Err(BackendError::InvalidInput(format!(
                "batch contains {} items; maximum is {MAX_BATCH_ITEMS}",
                inputs.len()
            )));
        }
        for input in inputs {
            if input.source_lang != self.source_lang || input.target_lang != self.target_lang {
                return Err(BackendError::UnsupportedDirection(format!(
                    "{} -> {}; loaded model is {} -> {}",
                    input.source_lang, input.target_lang, self.source_lang, self.target_lang
                )));
            }
            if input.text.len() > MAX_FRAME_BYTES {
                return Err(BackendError::InvalidInput(
                    "input exceeds the Bergamot frame limit".into(),
                ));
            }
            if input.max_output_tokens != 512 {
                return Err(BackendError::InvalidInput(
                    "max_output_tokens is configurable only on the MLX backend; Bergamot uses the model's fixed max-length-factor"
                        .into(),
                ));
            }
        }
        let texts: Vec<_> = inputs.iter().map(|input| input.text.clone()).collect();
        let translated = self.exchange(&texts)?;
        if translated.len() != inputs.len() {
            return Err(BackendError::Inference(format!(
                "worker returned {} outputs for {} inputs",
                translated.len(),
                inputs.len()
            )));
        }
        Ok(inputs
            .iter()
            .zip(translated)
            .map(|(_input, output)| TranslationOutput {
                input_tokens: output.input_tokens,
                output_tokens: output.output_tokens,
                text: output.text,
                score: None,
            })
            .collect())
    }
}

impl Drop for BergamotBackend {
    fn drop(&mut self) {
        self.transport.healthy.store(false, Ordering::Release);
        let _ = self.child.kill();
        self.transport.requests.take();
        let _ = self.child.wait();
        if let Some(thread) = self.writer_thread.take() {
            let _ = thread.join();
        }
        if let Some(thread) = self.reader_thread.take() {
            let _ = thread.join();
        }
    }
}

fn worker_writer_loop(
    mut input: impl Write,
    requests: Receiver<WorkerRequest>,
    healthy: Arc<AtomicBool>,
) {
    while let Ok(request) = requests.recv() {
        let result = write_request(&mut input, &request.texts).and_then(|()| input.flush());
        let terminal = result.is_err();
        if terminal {
            healthy.store(false, Ordering::Release);
        }
        let _ = request.reply.send(result);
        if terminal {
            break;
        }
    }
}

fn worker_reader_loop(
    mut output: impl Read,
    responses: mpsc::Sender<Result<Vec<WorkerOutput>, WorkerResponseError>>,
    healthy: Arc<AtomicBool>,
) {
    loop {
        let response = read_response(&mut output);
        let terminal = matches!(&response, Err(WorkerResponseError::Transport(_)));
        if terminal {
            healthy.store(false, Ordering::Release);
        }
        if responses.send(response).is_err() || terminal {
            break;
        }
    }
}

fn write_u32(writer: &mut impl Write, value: usize) -> io::Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "frame value exceeds u32"))?;
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<usize> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes) as usize)
}

fn write_request(writer: &mut impl Write, texts: &[String]) -> io::Result<()> {
    write_u32(writer, texts.len())?;
    for text in texts {
        write_u32(writer, text.len())?;
        writer.write_all(text.as_bytes())?;
    }
    Ok(())
}

fn read_string(reader: &mut impl Read) -> io::Result<String> {
    let length = read_u32(reader)?;
    if length > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "worker frame exceeds 1 MiB",
        ));
    }
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
}

fn read_response(reader: &mut impl Read) -> Result<Vec<WorkerOutput>, WorkerResponseError> {
    let status = read_u32(reader)?;
    if status != 0 {
        return Err(WorkerResponseError::Reported(read_string(reader)?));
    }
    let count = read_u32(reader)?;
    if count > MAX_BATCH_ITEMS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "worker response has too many items",
        )
        .into());
    }
    (0..count)
        .map(|_| {
            Ok(WorkerOutput {
                input_tokens: read_u32(reader)?,
                output_tokens: read_u32(reader)?,
                text: read_string(reader)?,
            })
        })
        .collect::<Result<Vec<_>, io::Error>>()
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn protocol_preserves_utf8_and_order() {
        let mut bytes = Vec::new();
        write_u32(&mut bytes, 0).unwrap();
        write_u32(&mut bytes, 2).unwrap();
        for (input_tokens, output_tokens, text) in [(3, 2, "你好"), (4, 5, "second")] {
            write_u32(&mut bytes, input_tokens).unwrap();
            write_u32(&mut bytes, output_tokens).unwrap();
            write_u32(&mut bytes, text.len()).unwrap();
            bytes.extend_from_slice(text.as_bytes());
        }
        let output = read_response(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(output[0].text, "你好");
        assert_eq!(output[0].input_tokens, 3);
        assert_eq!(output[0].output_tokens, 2);
        assert_eq!(output[1].text, "second");
    }

    #[test]
    fn protocol_surfaces_worker_errors() {
        let mut bytes = Vec::new();
        write_u32(&mut bytes, 1).unwrap();
        write_u32(&mut bytes, 4).unwrap();
        bytes.extend_from_slice(b"boom");
        assert_eq!(
            read_response(&mut Cursor::new(bytes))
                .unwrap_err()
                .to_string(),
            "boom"
        );
    }

    #[test]
    fn reader_marks_eof_as_terminal() {
        let healthy = Arc::new(AtomicBool::new(true));
        let (responses, response_rx) = mpsc::channel();

        worker_reader_loop(
            Cursor::new(Vec::<u8>::new()),
            responses,
            Arc::clone(&healthy),
        );

        assert!(!healthy.load(Ordering::Acquire));
        assert!(matches!(
            response_rx.recv().unwrap(),
            Err(WorkerResponseError::Transport(error))
                if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }

    #[test]
    fn transport_timeout_is_terminal() {
        let healthy = Arc::new(AtomicBool::new(true));
        let (requests, _request_rx) = mpsc::sync_channel(1);
        let (_response_tx, responses) = mpsc::channel();
        let mut transport = WorkerTransport {
            requests: Some(requests),
            responses,
            healthy: Arc::clone(&healthy),
            response_timeout: Duration::from_millis(20),
        };

        let error = transport.exchange(&["hello".into()]).unwrap_err();

        assert!(error.to_string().contains("timed out"));
        assert!(!healthy.load(Ordering::Acquire));
    }
}
