import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

const root = new URL("../../", import.meta.url);
const artifact = process.argv[2]
  ? new URL(process.argv[2], root)
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

function result() {
  const bytes = new Uint8Array(
    exports.memory.buffer,
    exports.result_pointer(),
    exports.result_length(),
  );
  return JSON.parse(new TextDecoder().decode(bytes));
}

const names = [
  "manifest.json",
  "model.q8.bin",
  "source.spm",
  "target.spm",
  "shortlist.bin",
];
const payloads = [];
for (const name of names) {
  const bytes = await readFile(new URL(name, modelDir));
  console.log({ phase: "transfer", name, bytes: bytes.byteLength });
  payloads.push(transfer(bytes));
}
console.log({ phase: "before-init", wasmMemoryMiB: exports.memory_bytes() / 1024 / 1024 });
const started = performance.now();
const status = exports.init(...payloads.flat());
console.log({
  phase: "init",
  status,
  milliseconds: performance.now() - started,
  wasmMemoryMiB: exports.memory_bytes() / 1024 / 1024,
  result: result(),
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
