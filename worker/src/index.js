import wasmModule from "../dist/marian_worker_wasm.wasm";

const decoder = new TextDecoder();
const encoder = new TextEncoder();
const instance = new WebAssembly.Instance(wasmModule, {});
const wasm = instance.exports;
let initialization;

const BUNDLE_KEY = "enzh-q8-packed-v3/model.worker-bundle-v3.bin";
const BUNDLE_ETAG = "652da5d66f3fcfcc1823daceb625266d";
const BUNDLE_SIZE = 50_197_812;
const BOOTSTRAP_SIZE = 6_415_640;
const MAXIMUM_BATCH = 16;

function wasmResult() {
  const bytes = new Uint8Array(
    wasm.memory.buffer,
    wasm.result_pointer(),
    wasm.result_length(),
  );
  return JSON.parse(decoder.decode(bytes));
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

async function readBundleRange(bucket, offset, length) {
  const object = await bucket.get(BUNDLE_KEY, { range: { offset, length } });
  if (!object) throw new Error(`missing R2 model object ${BUNDLE_KEY}`);
  if (object.size !== BUNDLE_SIZE || object.etag !== BUNDLE_ETAG) {
    throw new Error(`R2 bundle identity mismatch for ${BUNDLE_KEY}`);
  }
  const bytes = new Uint8Array(await object.arrayBuffer());
  if (bytes.byteLength !== length) {
    throw new Error(`short R2 range for ${BUNDLE_KEY}: ${bytes.byteLength} != ${length}`);
  }
  return bytes;
}

function splitBootstrap(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const lengths = Array.from({ length: 5 }, (_, index) => view.getUint32(index * 4, true));
  let offset = 20;
  const sections = lengths.map((length) => {
    const section = bytes.subarray(offset, offset + length);
    offset += length;
    return section;
  });
  if (offset !== bytes.byteLength) throw new Error("bootstrap sections do not cover payload");
  return sections;
}

async function transferBundle(bucket) {
  const leading = await readBundleRange(bucket, 0, 28 + BOOTSTRAP_SIZE);
  const header = leading.subarray(0, 28);
  if (decoder.decode(header.subarray(0, 8)) !== "MARIBND3") {
    throw new Error("invalid Worker bundle magic");
  }
  const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  if (view.getUint32(8, true) !== 3) throw new Error("unsupported Worker bundle version");
  const lengths = [
    view.getUint32(12, true),
    view.getUint32(16, true) * 4,
    view.getUint32(20, true),
    view.getUint32(24, true),
  ];
  if (lengths[0] !== BOOTSTRAP_SIZE) throw new Error("Worker bootstrap size mismatch");
  if (28 + lengths.reduce((sum, value) => sum + value, 0) !== BUNDLE_SIZE) {
    throw new Error("Worker bundle section lengths do not match pinned size");
  }
  const [manifest, metadata, source, target, shortlist] = splitBootstrap(
    leading.subarray(28),
  );
  let offset = 28 + BOOTSTRAP_SIZE;
  const dense = transferDense(await readBundleRange(bucket, offset, lengths[1]));
  offset += lengths[1];
  const encoderEmbedding = transferBytes(await readBundleRange(bucket, offset, lengths[2]));
  offset += lengths[2];
  const decoderEmbedding = transferBytes(await readBundleRange(bucket, offset, lengths[3]));
  return {
    manifest: transferBytes(manifest),
    metadata: transferBytes(metadata),
    dense,
    encoderEmbedding,
    decoderEmbedding,
    source: transferBytes(source),
    target: transferBytes(target),
    shortlist: transferBytes(shortlist),
  };
}

async function initialize(env) {
  const bundle = await transferBundle(env.MODELS);
  const status = wasm.init_packed_parts(
    ...bundle.manifest,
    ...bundle.metadata,
    ...bundle.dense,
    ...bundle.encoderEmbedding,
    ...bundle.decoderEmbedding,
    ...bundle.source,
    ...bundle.target,
    ...bundle.shortlist,
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
        simd: "simd128+relaxed-simd",
        model_loads: 4,
        maximum_batch: MAXIMUM_BATCH,
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
      const isSingle = typeof body.text === "string";
      const isBatch = Array.isArray(body.texts);
      if (isSingle === isBatch) {
        return json({ error: "provide exactly one of text or texts" }, 400);
      }
      const texts = isSingle ? [body.text] : body.texts;
      if (
        texts.length === 0 ||
        texts.length > MAXIMUM_BATCH ||
        texts.some((text) => typeof text !== "string" || text.length === 0)
      ) {
        return json({ error: `texts must contain 1..${MAXIMUM_BATCH} non-empty strings` }, 400);
      }
      const encodedTexts = texts.map((text) => encoder.encode(text));
      if (encodedTexts.some((bytes) => bytes.byteLength > 16_384)) {
        return json({ error: "each text must not exceed 16384 UTF-8 bytes" }, 413);
      }
      if (encodedTexts.reduce((sum, bytes) => sum + bytes.byteLength, 0) > 65_536) {
        return json({ error: "batch exceeds 65536 UTF-8 bytes" }, 413);
      }
      if ((body.source ?? "en") !== "en" || (body.target ?? "zh") !== "zh") {
        return json({ error: "only en -> zh is supported" }, 400);
      }

      const initStarted = performance.now();
      await ensureInitialized(env);
      const initMilliseconds = performance.now() - initStarted;
      const bytes = isSingle ? encodedTexts[0] : encoder.encode(JSON.stringify(texts));
      const pointer = wasm.alloc(bytes.byteLength);
      new Uint8Array(wasm.memory.buffer, pointer, bytes.byteLength).set(bytes);
      const inferenceStarted = performance.now();
      const status = isSingle
        ? wasm.translate(pointer, bytes.byteLength, Number(body.max_output_tokens ?? 128))
        : wasm.translate_batch_json(
            pointer,
            bytes.byteLength,
            Number(body.max_output_tokens ?? 128),
          );
      const inferenceMilliseconds = performance.now() - inferenceStarted;
      wasm.dealloc(pointer, bytes.byteLength);
      const result = wasmResult();
      const output = isBatch && status === 0 ? { translations: result } : result;
      return json(output, status === 0 ? 200 : 500, {
        "server-timing": [
          `model;dur=${initMilliseconds.toFixed(1)}`,
          `inference;dur=${inferenceMilliseconds.toFixed(1)}`,
          `total;dur=${(performance.now() - wallStarted).toFixed(1)}`,
        ].join(", "),
        "x-wasm-memory-mib": (wasm.memory_bytes() / 1024 / 1024).toFixed(2),
        "x-model-batch-size": String(texts.length),
      });
    } catch (error) {
      return json({ error: String(error?.message ?? error) }, 500, {
        "server-timing": `total;dur=${(performance.now() - wallStarted).toFixed(1)}`,
      });
    }
  },
};
