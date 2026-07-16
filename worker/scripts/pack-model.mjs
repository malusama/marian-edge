import { readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";

const root = new URL("../../", import.meta.url);
const artifact = new URL(
  "target/wasm32-unknown-unknown/release/marian_worker_wasm.wasm",
  root,
);
const modelDir = new URL(
  process.argv[2] ?? "file:///tmp/marian-worker-model/",
  root,
);
const output = new URL(
  process.argv[3] ?? "model.worker-packed-v2.bin",
  modelDir,
);

const module = await WebAssembly.compile(await readFile(artifact));
const instance = await WebAssembly.instantiate(module, {});
const { exports } = instance;

function transfer(bytes) {
  const pointer = exports.alloc(bytes.byteLength);
  new Uint8Array(exports.memory.buffer, pointer, bytes.byteLength).set(bytes);
  return [pointer, bytes.byteLength];
}

const manifest = await readFile(new URL("manifest.json", modelDir));
const weights = await readFile(new URL("model.q8.bin", modelDir));
const started = performance.now();
const status = exports.pack_model(...transfer(manifest), ...transfer(weights));
const result = new Uint8Array(
  exports.memory.buffer,
  exports.result_pointer(),
  exports.result_length(),
);
if (status !== 0) {
  throw new Error(new TextDecoder().decode(result));
}
await writeFile(output, result);
const workerManifest = JSON.parse(manifest.toString("utf8"));
workerManifest.weights = "model.worker-packed-v2.bin";
workerManifest.checksums.weights_sha256 = createHash("sha256").update(result).digest("hex");
const manifestOutput = new URL("manifest.worker.json", modelDir);
await writeFile(manifestOutput, `${JSON.stringify(workerManifest, null, 2)}\n`);
console.log({
  output: output.pathname,
  bytes: result.byteLength,
  milliseconds: performance.now() - started,
  wasmMemoryMiB: exports.memory_bytes() / 1024 / 1024,
  sha256: workerManifest.checksums.weights_sha256,
  manifest: manifestOutput.pathname,
});
