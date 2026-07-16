import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

const args = Object.fromEntries(
  process.argv.slice(2).map((argument) => {
    const [key, value = "true"] = argument.replace(/^--/, "").split("=", 2);
    return [key, value];
  }),
);
const target = args.target ?? "worker";
const batchSize = Number(args["batch-size"] ?? 1);
const requestCount = Number(args.requests ?? 100);
const concurrency = Number(args.concurrency ?? 1);
const warmup = Number(args.warmup ?? 5);
if (![batchSize, requestCount, concurrency].every(Number.isInteger)) {
  throw new Error("batch-size, requests, and concurrency must be integers");
}

const root = new URL("../../", import.meta.url);
const corpus = (await readFile(new URL("benchmarks/corpus-v1.jsonl", root), "utf8"))
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line).text);
const token =
  target === "worker"
    ? (await readFile("/tmp/marian-worker-api-token", "utf8")).trim()
    : undefined;

function requestFor(index) {
  const texts = Array.from(
    { length: batchSize },
    (_, item) => corpus[(index * batchSize + item) % corpus.length],
  );
  if (target === "worker") {
    return {
      url: "https://marian-worker.malu.moe/v1/translate",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${token}`,
      },
      body: batchSize === 1
        ? { text: texts[0], source: "en", target: "zh" }
        : { texts, source: "en", target: "zh" },
    };
  }
  return batchSize === 1
    ? {
        url: "http://127.0.0.1:3000/translate",
        headers: { "content-type": "application/json" },
        body: { text: texts[0], from: "en", to: "zh" },
      }
    : {
        url: "http://127.0.0.1:3000/imme",
        headers: { "content-type": "application/json" },
        body: { text_list: texts, source_lang: "en", target_lang: "zh" },
      };
}

async function runOne(index) {
  const request = requestFor(index);
  const started = performance.now();
  const response = await fetch(request.url, {
    method: "POST",
    headers: request.headers,
    body: JSON.stringify(request.body),
  });
  const payload = await response.json();
  const milliseconds = performance.now() - started;
  if (!response.ok) throw new Error(`${response.status}: ${JSON.stringify(payload)}`);
  const timing = response.headers.get("server-timing") ?? "";
  const model = /model;dur=([0-9.]+)/.exec(timing);
  return {
    milliseconds,
    modelMilliseconds: model ? Number(model[1]) : 0,
  };
}

for (let index = 0; index < warmup; index += 1) await runOne(index);
const results = new Array(requestCount);
let next = 0;
async function worker() {
  while (true) {
    const index = next++;
    if (index >= requestCount) return;
    results[index] = await runOne(index);
  }
}
const started = performance.now();
await Promise.all(Array.from({ length: concurrency }, worker));
const wallSeconds = (performance.now() - started) / 1000;

function percentile(values, fraction) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.min(ordered.length - 1, Math.ceil(ordered.length * fraction) - 1)];
}

const latencies = results.map((result) => result.milliseconds);
const coldLoads = results.filter((result) => result.modelMilliseconds > 10);
console.log(
  JSON.stringify(
    {
      target,
      batchSize,
      requestCount,
      concurrency,
      wallSeconds,
      requestsPerSecond: requestCount / wallSeconds,
      textsPerSecond: (requestCount * batchSize) / wallSeconds,
      latencyMilliseconds: {
        mean: latencies.reduce((sum, value) => sum + value, 0) / latencies.length,
        p50: percentile(latencies, 0.5),
        p95: percentile(latencies, 0.95),
        p99: percentile(latencies, 0.99),
      },
      coldLoads: coldLoads.length,
      maximumModelLoadMilliseconds: Math.max(0, ...coldLoads.map((item) => item.modelMilliseconds)),
    },
    null,
    2,
  ),
);
