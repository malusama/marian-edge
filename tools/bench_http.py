#!/usr/bin/env python3
"""Dependency-free, metadata-complete benchmark for Marian HTTP endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000/translate")
    parser.add_argument("--text", default="The weather is beautiful today.")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--from-lang", default="en")
    parser.add_argument("--to-lang", default="zh")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--pid", type=int, help="server PID to sample for peak RSS")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup non-negative")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.5) - 1))
    return ordered[index]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_corpus(path: Path | None, fallback: str) -> tuple[list[str], str]:
    if path is None:
        payload = (fallback + "\n").encode()
        return [fallback], sha256_bytes(payload)
    raw = path.read_bytes()
    texts: list[str] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        text = item["text"] if isinstance(item, dict) else item
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
        texts.append(text)
    if not texts:
        raise ValueError(f"{path}: corpus is empty")
    return texts, sha256_bytes(raw)


def request(url: str, body: bytes) -> tuple[float, Any]:
    started = time.perf_counter()
    call = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(call, timeout=120) as response:
        parsed = json.load(response)
    return (time.perf_counter() - started) * 1000, parsed


def info_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/info", "", ""))


def fetch_info(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(info_url(url), timeout=5) as response:
            value = json.load(response)
            return value if isinstance(value, dict) else {"value": value}
    except Exception as error:  # Metadata collection must not hide benchmark output.
        return {"error": str(error)}


def git_commit(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_metadata(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {}
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, str] = {}
    for field in ("weights", "source_vocab", "target_vocab", "shortlist"):
        name = manifest.get(field)
        if not name:
            continue
        path = model_dir / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[name] = digest.hexdigest()
    return {
        "directory": str(model_dir.resolve()),
        "model_id": manifest.get("model_id"),
        "precision": manifest.get("precision"),
        "files_sha256": files,
    }


class RssSampler:
    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.peak_kib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.pid is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.peak_kib

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                value = subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(self.pid)],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if value:
                    rss = int(value.splitlines()[-1])
                    self.peak_kib = max(self.peak_kib or 0, rss)
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            self._stop.wait(0.01)


def request_bodies(
    url: str, texts: list[str], count: int, source: str, target: str
) -> tuple[list[bytes], int]:
    if urllib.parse.urlsplit(url).path.rstrip("/") == "/imme":
        body = json.dumps(
            {"source_lang": source, "target_lang": target, "text_list": texts},
            ensure_ascii=False,
        ).encode()
        return [body] * count, len(texts)
    bodies = [
        json.dumps({"text": texts[index % len(texts)], "from": source, "to": target}, ensure_ascii=False).encode()
        for index in range(count)
    ]
    return bodies, 1


def stable_output_hash(result: Any) -> str:
    return sha256_bytes(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def main() -> None:
    args = parse_args()
    texts, corpus_sha256 = load_corpus(args.corpus, args.text)
    warmup_bodies, _ = request_bodies(
        args.url, texts, max(args.warmup, 1), args.from_lang, args.to_lang
    )
    for index in range(args.warmup):
        request(args.url, warmup_bodies[index % len(warmup_bodies)])

    bodies, items_per_request = request_bodies(
        args.url, texts, args.requests, args.from_lang, args.to_lang
    )
    sampler = RssSampler(args.pid)
    sampler.start()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request, args.url, body) for body in bodies]
        results = [future.result() for future in as_completed(futures)]
    wall = time.perf_counter() - started
    peak_rss_kib = sampler.stop()
    latencies = [latency for latency, _ in results]
    output_hashes = {stable_output_hash(result) for _, result in results}
    measured_items = len(results) * items_per_request
    report = {
        "schema": "marian-mlx.benchmark.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_commit(args.commit),
        "server": fetch_info(args.url),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "model": model_metadata(args.model_dir),
        "workload": {
            "url": args.url,
            "corpus": str(args.corpus.resolve()) if args.corpus else "inline",
            "corpus_sha256": corpus_sha256,
            "corpus_items": len(texts),
            "requests": len(results),
            "items_per_request": items_per_request,
            "measured_items": measured_items,
            "concurrency": args.concurrency,
            "threads": args.threads,
            "warmup": args.warmup,
        },
        "results": {
            "wall_seconds": round(wall, 6),
            "throughput_requests_per_second": round(len(results) / wall, 3),
            "throughput_items_per_second": round(measured_items / wall, 3),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 3),
                "p50": round(percentile(latencies, 0.50), 3),
                "p95": round(percentile(latencies, 0.95), 3),
                "p99": round(percentile(latencies, 0.99), 3),
            },
            "peak_rss_kib": peak_rss_kib,
            "distinct_output_hashes": len(output_hashes),
            "output_sha256": sorted(output_hashes),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
