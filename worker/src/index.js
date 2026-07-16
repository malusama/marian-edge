import wasmModule from "../dist/marian_worker_wasm.wasm";

const decoder = new TextDecoder();
const encoder = new TextEncoder();
const instance = new WebAssembly.Instance(wasmModule, {});
const wasm = instance.exports;
let initialization;

const MODEL_SHA256 = {
  "manifest.worker.json": "d2d95fefa172bb88b09fd53c008e56bbaad4f09dd41c5c43e02c205dfb983918",
  "source.spm": "bd9b65504acc6d9726dd281f7defc2adb7c2c22d0688fe2f84697de25197c8c5",
  "target.spm": "aded6993c36e440284d11cec3f6b8aef9c0e43188a772d80be342a713adf223d",
  "shortlist.bin": "8575d8daa10e2dbff316dcdf8e1ce475357bcc2c92bdc63b736a2d5add22f681",
};
const PACKED_KEY = "enzh-q8-packed-v2/model.worker-packed-v2.bin";
const PACKED_ETAG = "63e3ab12906ad2f6ed805181180bcaca";
const PACKED_SIZE = 44_132_384;

function wasmResult() {
  const bytes = new Uint8Array(
    wasm.memory.buffer,
    wasm.result_pointer(),
    wasm.result_length(),
  );
  return JSON.parse(decoder.decode(bytes));
}

async function transferR2Object(bucket, key) {
  const object = await bucket.get(key);
  if (!object) throw new Error(`missing R2 model object ${key}`);
  const buffer = await object.arrayBuffer();
  const name = key.slice(key.lastIndexOf("/") + 1);
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== MODEL_SHA256[name]) {
    throw new Error(`SHA-256 mismatch for R2 model object ${key}`);
  }
  const bytes = new Uint8Array(buffer);
  const pointer = wasm.alloc(bytes.byteLength);
  new Uint8Array(wasm.memory.buffer, pointer, bytes.byteLength).set(bytes);
  return [pointer, bytes.byteLength];
}

function transferBytes(bytes) {
  const pointer = wasm.alloc(bytes.byteLength);
  new Uint8Array(wasm.memory.buffer, pointer, bytes.byteLength).set(bytes);
  return [pointer, bytes.byteLength];
}

function transferDense(bytes) {
  if (bytes.byteLength % 4 !== 0) throw new Error("packed dense section is not u32 aligned");
  const words = bytes.byteLength / 4;
  const pointer = wasm.alloc_u32(words);
  new Uint8Array(wasm.memory.buffer, pointer, bytes.byteLength).set(bytes);
  return [pointer, words];
}

async function readPackedRange(bucket, offset, length) {
  const object = await bucket.get(PACKED_KEY, { range: { offset, length } });
  if (!object) throw new Error(`missing R2 model object ${PACKED_KEY}`);
  const bytes = new Uint8Array(await object.arrayBuffer());
  if (bytes.byteLength !== length) {
    throw new Error(`short R2 range for ${PACKED_KEY}: ${bytes.byteLength} != ${length}`);
  }
  return bytes;
}

async function transferPackedSections(bucket) {
  const object = await bucket.head(PACKED_KEY);
  if (!object || object.size !== PACKED_SIZE || object.etag !== PACKED_ETAG) {
    throw new Error(`R2 packed model identity mismatch for ${PACKED_KEY}`);
  }
  const header = await readPackedRange(bucket, 0, 28);
  if (decoder.decode(header.subarray(0, 8)) !== "MARIBND2") {
    throw new Error("invalid packed bundle magic");
  }
  const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  if (view.getUint32(8, true) !== 2) throw new Error("unsupported packed bundle version");
  const lengths = [
    view.getUint32(12, true),
    view.getUint32(16, true) * 4,
    view.getUint32(20, true),
    view.getUint32(24, true),
  ];
  if (28 + lengths.reduce((sum, value) => sum + value, 0) !== PACKED_SIZE) {
    throw new Error("packed bundle section lengths do not match pinned size");
  }
  const payloads = [];
  let offset = 28;
  for (let index = 0; index < lengths.length; index += 1) {
    const bytes = await readPackedRange(bucket, offset, lengths[index]);
    payloads.push(index === 1 ? transferDense(bytes) : transferBytes(bytes));
    offset += lengths[index];
  }
  return payloads;
}

async function initialize(env) {
  const keys = [
    "manifest.worker.json",
    "source.spm",
    "target.spm",
    "shortlist.bin",
  ];
  const common = new Map();
  for (const key of keys) {
    common.set(key, await transferR2Object(env.MODELS, `enzh-q8-packed-v2/${key}`));
  }
  const packed = await transferPackedSections(env.MODELS);
  const status = wasm.init_packed_parts(
    ...common.get("manifest.worker.json"),
    ...packed[0],
    ...packed[1],
    ...packed[2],
    ...packed[3],
    ...common.get("source.spm"),
    ...common.get("target.spm"),
    ...common.get("shortlist.bin"),
  );
  if (status !== 0) throw new Error(wasmResult().error ?? "Wasm init failed");
}

function ensureInitialized(env) {
  initialization ??= initialize(env).catch((error) => {
    initialization = undefined;
    throw error;
  });
  return initialization;
}

function json(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extraHeaders,
    },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/healthz") {
      return json({
        ok: true,
        initialized: initialization !== undefined,
        wasm_memory_mib: wasm.memory_bytes() / 1024 / 1024,
      });
    }
    if (request.method !== "POST" || url.pathname !== "/v1/translate") {
      return json({ error: "use POST /v1/translate" }, 404);
    }
    if (
      !env.API_TOKEN ||
      request.headers.get("authorization") !== `Bearer ${env.API_TOKEN}`
    ) {
      return json({ error: "unauthorized" }, 401, {
        "www-authenticate": "Bearer",
      });
    }

    const wallStarted = performance.now();
    try {
      const body = await request.json();
      if (typeof body.text !== "string" || body.text.length === 0) {
        return json({ error: "text must be a non-empty string" }, 400);
      }
      if (encoder.encode(body.text).byteLength > 16_384) {
        return json({ error: "text exceeds 16384 UTF-8 bytes" }, 413);
      }
      if ((body.source ?? "en") !== "en" || (body.target ?? "zh") !== "zh") {
        return json({ error: "only en -> zh is supported" }, 400);
      }

      const initStarted = performance.now();
      await ensureInitialized(env);
      const initMilliseconds = performance.now() - initStarted;
      const bytes = encoder.encode(body.text);
      const pointer = wasm.alloc(bytes.byteLength);
      new Uint8Array(wasm.memory.buffer, pointer, bytes.byteLength).set(bytes);
      const inferenceStarted = performance.now();
      const status = wasm.translate(
        pointer,
        bytes.byteLength,
        Number(body.max_output_tokens ?? 128),
      );
      const inferenceMilliseconds = performance.now() - inferenceStarted;
      wasm.dealloc(pointer, bytes.byteLength);
      const output = wasmResult();
      return json(output, status === 0 ? 200 : 500, {
        "server-timing": [
          `model;dur=${initMilliseconds.toFixed(1)}`,
          `inference;dur=${inferenceMilliseconds.toFixed(1)}`,
          `total;dur=${(performance.now() - wallStarted).toFixed(1)}`,
        ].join(", "),
        "x-wasm-memory-mib": (wasm.memory_bytes() / 1024 / 1024).toFixed(2),
      });
    } catch (error) {
      return json({ error: String(error?.message ?? error) }, 500, {
        "server-timing": `total;dur=${(performance.now() - wallStarted).toFixed(1)}`,
      });
    }
  },
};
