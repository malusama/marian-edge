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
const bundleOutput = new URL("model.worker-bundle-v3.bin", modelDir);

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
const packed = Buffer.from(result);
if (packed.subarray(0, 8).toString("utf8") !== "MARIBND2") {
  throw new Error("packed v2 result has an invalid magic header");
}
const packedView = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
const packedLengths = [
  packedView.getUint32(12, true),
  packedView.getUint32(16, true) * 4,
  packedView.getUint32(20, true),
  packedView.getUint32(24, true),
];
let packedOffset = 28;
const packedSections = packedLengths.map((length) => {
  const section = packed.subarray(packedOffset, packedOffset + length);
  packedOffset += length;
  return section;
});
if (packedOffset !== packed.byteLength) {
  throw new Error("packed v2 section lengths do not cover the artifact");
}

const common = [
  await readFile(manifestOutput),
  await readFile(new URL("source.spm", modelDir)),
  await readFile(new URL("target.spm", modelDir)),
  await readFile(new URL("shortlist.bin", modelDir)),
];
const bootstrapParts = [common[0], packedSections[0], common[1], common[2], common[3]];
const bootstrapHeader = Buffer.alloc(bootstrapParts.length * 4);
bootstrapParts.forEach((part, index) => bootstrapHeader.writeUInt32LE(part.byteLength, index * 4));
const bootstrap = Buffer.concat([bootstrapHeader, ...bootstrapParts]);
const bundleHeader = Buffer.alloc(28);
bundleHeader.write("MARIBND3", 0, "utf8");
bundleHeader.writeUInt32LE(3, 8);
for (const [index, length] of [
  bootstrap.byteLength,
  packedSections[1].byteLength / 4,
  packedSections[2].byteLength,
  packedSections[3].byteLength,
].entries()) {
  bundleHeader.writeUInt32LE(length, 12 + index * 4);
}
const bundle = Buffer.concat([
  bundleHeader,
  bootstrap,
  packedSections[1],
  packedSections[2],
  packedSections[3],
]);
await writeFile(bundleOutput, bundle);
console.log({
  output: output.pathname,
  bytes: result.byteLength,
  milliseconds: performance.now() - started,
  wasmMemoryMiB: exports.memory_bytes() / 1024 / 1024,
  sha256: workerManifest.checksums.weights_sha256,
  manifest: manifestOutput.pathname,
  workerBundle: bundleOutput.pathname,
  workerBundleBytes: bundle.byteLength,
  workerBundleBootstrapBytes: bootstrap.byteLength,
  workerBundleSha256: createHash("sha256").update(bundle).digest("hex"),
  workerBundleMd5: createHash("md5").update(bundle).digest("hex"),
});
