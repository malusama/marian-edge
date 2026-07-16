import wasmModule from "../dist/marian_worker_wasm.wasm";

const decoder = new TextDecoder();
const encoder = new TextEncoder();
const instance = new WebAssembly.Instance(wasmModule, {});
const wasm = instance.exports;
let initialization;

const MODEL_SHA256 = {
  "manifest.json": "bbd3e09c7a5eb70d972fe9808c86d5dc1dae69b4bce1cf4eda21d6f9aba4cec9",
  "model.q8.bin": "4e5accc141373565ddc8fa1565bceaa8d0c3482a82cab8131c719ebcc6c2157c",
  "source.spm": "bd9b65504acc6d9726dd281f7defc2adb7c2c22d0688fe2f84697de25197c8c5",
  "target.spm": "aded6993c36e440284d11cec3f6b8aef9c0e43188a772d80be342a713adf223d",
  "shortlist.bin": "8575d8daa10e2dbff316dcdf8e1ce475357bcc2c92bdc63b736a2d5add22f681",
};

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

async function initialize(env) {
  const keys = [
    "manifest.json",
    "model.q8.bin",
    "source.spm",
    "target.spm",
    "shortlist.bin",
  ];
  const payloads = [];
  for (const key of keys) {
    payloads.push(await transferR2Object(env.MODELS, `enzh-q8/${key}`));
  }
  const status = wasm.init(...payloads.flat());
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
