import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

const root = new URL("../../", import.meta.url);
const packed = process.argv.includes("--packed");
const artifactArgument = process.argv.slice(2).find((argument) => argument !== "--packed");
const artifact = artifactArgument
  ? new URL(artifactArgument, root)
  : new URL(
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
  if (bytes.byteLength % 4 !== 0) throw new Error("dense section is not u32 aligned");
  const words = bytes.byteLength / 4;
  const pointer = exports.alloc_u32(words);
  new Uint8Array(exports.memory.buffer, pointer, bytes.byteLength).set(bytes);
  return [pointer, words];
}

function splitPackedBundle(bytes) {
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== "MARIBND2") {
    throw new Error("invalid packed bundle magic");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(8, true) !== 2) throw new Error("unsupported packed bundle version");
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

const manifestName = packed ? "manifest.worker.json" : "manifest.json";
const commonNames = [manifestName, "source.spm", "target.spm", "shortlist.bin"];
const common = new Map();
for (const name of commonNames) {
  const bytes = await readFile(new URL(name, modelDir));
  console.log({ phase: "transfer", name, bytes: bytes.byteLength });
  common.set(name, transfer(bytes));
}
let initFunction;
let initArguments;
if (packed) {
  const bundle = await readFile(new URL("model.worker-packed-v2.bin", modelDir));
  const [metadata, dense, encoderEmbedding, decoderEmbedding] = splitPackedBundle(bundle);
  console.log({
    phase: "packed-sections",
    metadata: metadata.byteLength,
    dense: dense.byteLength,
    encoderEmbedding: encoderEmbedding.byteLength,
    decoderEmbedding: decoderEmbedding.byteLength,
  });
  initFunction = exports.init_packed_parts;
  initArguments = [
    ...common.get(manifestName),
    ...transfer(metadata),
    ...transferU32(dense),
    ...transfer(encoderEmbedding),
    ...transfer(decoderEmbedding),
    ...common.get("source.spm"),
    ...common.get("target.spm"),
    ...common.get("shortlist.bin"),
  ];
} else {
  const weights = await readFile(new URL("model.q8.bin", modelDir));
  initFunction = exports.init;
  initArguments = [
    ...common.get(manifestName),
    ...transfer(weights),
    ...common.get("source.spm"),
    ...common.get("target.spm"),
    ...common.get("shortlist.bin"),
  ];
}
console.log({ phase: "before-init", wasmMemoryMiB: exports.memory_bytes() / 1024 / 1024 });
const started = performance.now();
const status = initFunction(...initArguments);
console.log({
  phase: "init",
  status,
  milliseconds: performance.now() - started,
  wasmMemoryMiB: exports.memory_bytes() / 1024 / 1024,
  result: result(),
  packed,
});
if (status !== 0) process.exit(1);

for (const text of [
  "Hello, world!",
  "The weather is beautiful today.",
  "Please open the window.",
  "Thank you for your help!",
  "Where is the nearest train station?",
]) {
  const encoded = new TextEncoder().encode(text);
  const [pointer, length] = transfer(encoded);
  const translateStarted = performance.now();
  const translateStatus = exports.translate(pointer, length, 128);
  const milliseconds = performance.now() - translateStarted;
  exports.dealloc(pointer, length);
  console.log({
    phase: "translate",
    status: translateStatus,
    milliseconds,
    wasmMemoryMiB: exports.memory_bytes() / 1024 / 1024,
    input: text,
    result: result(),
  });
}
