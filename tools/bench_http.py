#!/usr/bin/env python3
"""Small dependency-free concurrent benchmark for the translation endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3100/translate")
    parser.add_argument("--text", default="The weather is beautiful today.")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.5) - 1))
    return ordered[index]


def request(url: str, body: bytes) -> tuple[float, str]:
    started = time.perf_counter()
    call = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(call, timeout=60) as response:
        parsed = json.load(response)
    return (time.perf_counter() - started) * 1000, parsed["text"]


def main() -> None:
    args = parse_args()
    body = json.dumps({"text": args.text, "from": "en", "to": "zh"}).encode()
    for _ in range(args.warmup):
        request(args.url, body)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request, args.url, body) for _ in range(args.requests)]
        results = [future.result() for future in as_completed(futures)]
    wall = time.perf_counter() - started
    latencies = [latency for latency, _ in results]
    translations = {translation for _, translation in results}
    print(
        json.dumps(
            {
                "requests": len(results),
                "concurrency": args.concurrency,
                "throughput_rps": round(len(results) / wall, 2),
                "latency_ms": {
                    "mean": round(statistics.mean(latencies), 2),
                    "p50": round(percentile(latencies, 0.50), 2),
                    "p95": round(percentile(latencies, 0.95), 2),
                    "p99": round(percentile(latencies, 0.99), 2),
                },
                "distinct_outputs": len(translations),
                "sample": next(iter(translations)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
