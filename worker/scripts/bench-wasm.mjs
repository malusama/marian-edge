import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

const root = new URL("../../", import.meta.url);
const artifact = new URL(
  "target/wasm32-unknown-unknown/release/marian_worker_wasm.wasm",
  root,
);
const modelDir = new URL("file:///tmp/marian-worker-model/");
const module = await WebAssembly.compile(await readFile(artifact));
const instance = await WebAssembly.instantiate(module, {});
const { exports } = instance;

function transfer(bytes) {
  const pointer = exports.alloc(bytes.byteLength);
  new Uint8Array(exports.memory.buffer, pointer, bytes.byteLength).set(bytes);
  return [pointer, bytes.byteLength];
}

function transferU32(bytes) {
  const words = bytes.byteLength / 4;
  const pointer = exports.alloc_u32(words);
  new Uint8Array(exports.memory.buffer, pointer, bytes.byteLength).set(bytes);
  return [pointer, words];
}

function splitBundle(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const lengths = [
    view.getUint32(12, true),
    view.getUint32(16, true) * 4,
    view.getUint32(20, true),
    view.getUint32(24, true),
  ];
  let offset = 28;
  return lengths.map((length) => {
    const section = bytes.subarray(offset, offset + length);
    offset += length;
    return section;
  });
}

function result() {
  const bytes = new Uint8Array(
    exports.memory.buffer,
    exports.result_pointer(),
    exports.result_length(),
  );
  return JSON.parse(new TextDecoder().decode(bytes));
}

const manifest = await readFile(new URL("manifest.worker.json", modelDir));
const source = await readFile(new URL("source.spm", modelDir));
const target = await readFile(new URL("target.spm", modelDir));
const shortlist = await readFile(new URL("shortlist.bin", modelDir));
const bundle = await readFile(new URL("model.worker-packed-v2.bin", modelDir));
const [metadata, dense, encoderEmbedding, decoderEmbedding] = splitBundle(bundle);
const initStarted = performance.now();
const initStatus = exports.init_packed_parts(
  ...transfer(manifest),
  ...transfer(metadata),
  ...transferU32(dense),
  ...transfer(encoderEmbedding),
  ...transfer(decoderEmbedding),
  ...transfer(source),
  ...transfer(target),
  ...transfer(shortlist),
);
if (initStatus !== 0) throw new Error(JSON.stringify(result()));
const initMilliseconds = performance.now() - initStarted;

const corpus = [
  "Hello, world!",
  "The weather is beautiful today.",
  "Please open the window.",
  "Thank you for your help!",
  "Where is the nearest train station?",
  "This compact translation engine runs entirely on the edge without sending private text to a centralized inference service.",
  "When several page fragments are ready at once, batching them reduces repeated matrix multiplication overhead.",
  "Cloud platforms trade predictable hardware access for fast deployment and elastic request routing.",
  "A careful benchmark separates model computation, initialization, network latency, and queueing delay.",
  "The optimized artifact stores weights in the exact layout expected by the WebAssembly SIMD kernel.",
  "Please preserve punctuation, numbers such as 2026, and product names when translating this sentence.",
  "Performance improvements are useful only when translation quality and numerical correctness remain stable.",
  "The browser extension can collect neighboring text nodes and submit them as one bounded request.",
  "Cold starts become more expensive when a model is split across many remote object reads.",
  "Single-threaded execution still benefits from vector instructions and larger matrix batches.",
  "Measure the real workload before choosing between a local application, an edge Worker, and a native server.",
];

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

function translateBatch(texts) {
  const bytes = new TextEncoder().encode(JSON.stringify(texts));
  const [pointer, length] = transfer(bytes);
  const started = performance.now();
  const status = exports.translate_batch_json(pointer, length, 128);
  const elapsed = performance.now() - started;
  exports.dealloc(pointer, length);
  if (status !== 0) throw new Error(JSON.stringify(result()));
  return elapsed;
}

translateBatch([corpus[0]]);
const measurements = [];
for (const batchSize of [1, 4, 8, 16]) {
  const samples = [];
  for (let iteration = 0; iteration < 12; iteration += 1) {
    const offset = (iteration * batchSize) % corpus.length;
    const texts = Array.from(
      { length: batchSize },
      (_, index) => corpus[(offset + index) % corpus.length],
    );
    samples.push(translateBatch(texts));
  }
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  measurements.push({
    batchSize,
    meanMilliseconds: mean,
    p50Milliseconds: percentile(samples, 0.5),
    p95Milliseconds: percentile(samples, 0.95),
    textsPerSecond: (batchSize * 1000) / mean,
  });
}

console.log(JSON.stringify({ initMilliseconds, measurements }, null, 2));
