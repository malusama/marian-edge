#!/usr/bin/env python3
"""Summarize exported xctrace Metal tables without external dependencies."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAT = "marian-mlx.metal-trace-summary.v1"
EVIDENCE_FILES = {
    "submissions": "submissions.xml",
    "completed": "completed.xml",
    "errors": "errors.xml",
    "gpu_intervals": "gpu-intervals.xml",
    "device": "device.xml",
}


class SummaryError(Exception):
    """Raised when present evidence is corrupt rather than merely incomplete."""


@dataclass(frozen=True)
class Cell:
    element: ET.Element
    resolver: "ReferenceResolver"

    @property
    def resolved(self) -> ET.Element:
        return self.resolver.resolve(self.element)

    @property
    def display(self) -> str:
        element = self.resolved
        return (element.get("fmt") or element.text or "").strip()

    @property
    def raw(self) -> str:
        return (self.resolved.text or "").strip()

    def descendant(self, tag: str) -> str | None:
        element = self.resolved
        if element.tag == tag:
            return self.display
        for child in element.iter():
            if child is element:
                continue
            resolved = self.resolver.resolve(child)
            if resolved.tag == tag:
                return (resolved.get("fmt") or resolved.text or "").strip()
        return None


class ReferenceResolver:
    def __init__(self, node: ET.Element) -> None:
        self._by_id = {
            element_id: element
            for element in node.iter()
            if (element_id := element.get("id")) is not None
        }

    def resolve(self, element: ET.Element) -> ET.Element:
        seen: set[str] = set()
        current = element
        while reference := current.get("ref"):
            if reference in seen or reference not in self._by_id:
                break
            seen.add(reference)
            current = self._by_id[reference]
        return current


@dataclass
class Table:
    path: Path
    schema: str | None
    rows: list[dict[str, Cell]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact summary from exported Metal System Trace tables."
    )
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_table(path: Path, warnings: list[str]) -> Table | None:
    if not path.exists():
        warnings.append(f"evidence table is unavailable: {path.name}")
        return None
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise SummaryError(f"failed to parse {path}: {error}") from error
    node = root.find(".//node")
    if node is None:
        warnings.append(f"evidence table has no query result: {path.name}")
        return Table(path, None, [])
    schema = node.find("schema")
    if schema is None:
        warnings.append(f"evidence table has no schema: {path.name}")
        return Table(path, None, [])
    columns = [column.findtext("mnemonic") or "" for column in schema.findall("col")]
    resolver = ReferenceResolver(node)
    rows: list[dict[str, Cell]] = []
    for row in node.findall("row"):
        cells = list(row)
        rows.append(
            {
                mnemonic: Cell(cell, resolver)
                for mnemonic, cell in zip(columns, cells)
                if mnemonic
            }
        )
    return Table(path, schema.get("name"), rows)


def integer(cell: Cell | None) -> int | None:
    if cell is None:
        return None
    raw = cell.raw.replace(",", "")
    try:
        return int(raw, 0)
    except ValueError:
        return None


def duration_ns(cell: Cell | None) -> int | None:
    raw_value = integer(cell)
    if raw_value is not None:
        return raw_value
    if cell is None:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(ns|µs|us|ms|s)", cell.display)
    if not match:
        return None
    scale = {"ns": 1, "µs": 1_000, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
    return round(float(match.group(1)) * scale[match.group(2)])


def process_pid(cell: Cell | None) -> int | None:
    if cell is None:
        return None
    nested = cell.descendant("pid")
    if nested:
        try:
            return int(nested.replace(",", ""), 0)
        except ValueError:
            pass
    match = re.search(r"\(\s*(\d+)\s*\)\s*$", cell.display)
    return int(match.group(1)) if match else None


def command_buffer_id(row: dict[str, Cell]) -> str | None:
    cell = row.get("cmdbuffer-id")
    if cell is None:
        return None
    return cell.raw or cell.display or None


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.5) - 1))
    return ordered[index]


def milliseconds(value: int | None) -> float | None:
    return round(value / 1_000_000, 6) if value is not None else None


def submission_label(row: dict[str, Cell]) -> str:
    track = row.get("track-label")
    if track and track.display and track.resolved.tag != "sentinel":
        return track.display
    event = row.get("event-label")
    if event:
        nested = event.descendant("metal-object-label")
        if nested:
            return nested
        match = re.search(r'Committed\s+"\s*(.*?)\s*"', event.display)
        if match:
            return match.group(1)
    return "<unlabeled>"


def filter_process_rows(
    table: Table | None,
    pid: int,
    submitted_ids: set[str],
    warnings: list[str],
    label: str,
) -> list[dict[str, Cell]]:
    if table is None:
        return []
    if not table.rows:
        return []
    if any("process" in row for row in table.rows):
        return [row for row in table.rows if process_pid(row.get("process")) == pid]
    if submitted_ids and any("cmdbuffer-id" in row for row in table.rows):
        warnings.append(f"{label} process field is unavailable; filtered by command-buffer ID")
        return [row for row in table.rows if command_buffer_id(row) in submitted_ids]
    warnings.append(f"{label} rows could not be attributed to PID {pid}")
    return []


def load_benchmark(path: Path | None, warnings: list[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        warnings.append(f"benchmark report is unavailable: {path}")
        return None
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"failed to parse benchmark report {path}: {error}") from error
    workload = report.get("workload", {})
    results = report.get("results", {})
    latency = results.get("latency_ms", {})
    return {
        "path": str(path),
        "commit": report.get("commit"),
        "measured_items": workload.get("measured_items"),
        "concurrency": workload.get("concurrency"),
        "wall_seconds": results.get("wall_seconds"),
        "throughput_items_per_second": results.get("throughput_items_per_second"),
        "latency_ms": {"p50": latency.get("p50"), "p95": latency.get("p95")},
    }


def build_summary(
    evidence_dir: Path,
    pid: int,
    benchmark_path: Path | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    tables = {
        name: load_table(evidence_dir / filename, warnings)
        for name, filename in EVIDENCE_FILES.items()
    }

    submission_table = tables["submissions"]
    submission_rows = filter_process_rows(
        submission_table, pid, set(), warnings, "command-buffer submission"
    )
    submitted_ids = {
        identifier
        for row in submission_rows
        if (identifier := command_buffer_id(row)) is not None
    }
    labels = Counter(submission_label(row) for row in submission_rows)
    encoder_count_values = [integer(row.get("num-encoders")) for row in submission_rows]
    submission_duration = [duration_ns(row.get("duration")) for row in submission_rows]
    encoder_duration = [duration_ns(row.get("encoder-time")) for row in submission_rows]
    valid_encoder_counts = [value for value in encoder_count_values if value is not None]
    valid_submission_duration = [value for value in submission_duration if value is not None]
    valid_encoder_duration = [value for value in encoder_duration if value is not None]

    completed_rows = tables["completed"].rows if tables["completed"] else []
    error_rows = tables["errors"].rows if tables["errors"] else []
    completed_ids = {
        identifier
        for row in completed_rows
        if (identifier := command_buffer_id(row)) in submitted_ids
    }
    error_ids = {
        identifier
        for row in error_rows
        if (identifier := command_buffer_id(row)) in submitted_ids
    }

    gpu_rows = filter_process_rows(
        tables["gpu_intervals"], pid, submitted_ids, warnings, "GPU interval"
    )
    if gpu_rows and any("event-depth" in row for row in gpu_rows):
        gpu_rows = [row for row in gpu_rows if integer(row.get("event-depth")) == 0]
    elif gpu_rows:
        warnings.append("GPU interval depth is unavailable; total may include nested intervals")
    gpu_duration_by_command: dict[str, int] = defaultdict(int)
    gpu_interval_count = 0
    for row in gpu_rows:
        duration = duration_ns(row.get("duration"))
        identifier = command_buffer_id(row)
        if duration is None:
            continue
        gpu_interval_count += 1
        if identifier is not None:
            gpu_duration_by_command[identifier] += duration
    gpu_total = sum(gpu_duration_by_command.values())
    per_command = list(gpu_duration_by_command.values())

    process_name = next(
        (
            row["process"].display
            for row in submission_rows + gpu_rows
            if "process" in row and row["process"].display
        ),
        None,
    )
    device_rows = tables["device"].rows if tables["device"] else []
    device_row = device_rows[0] if device_rows else {}
    device_name = (
        device_row.get("device-name").display
        if device_row.get("device-name")
        else next(
            (
                row["gpu"].display
                for row in submission_rows
                if "gpu" in row and row["gpu"].display
            ),
            None,
        )
    )

    if submission_table is not None and not submission_rows:
        warnings.append(f"no command-buffer submissions were attributed to PID {pid}")
    if submission_rows and not gpu_duration_by_command:
        warnings.append("no top-level GPU intervals matched the submitted command buffers")
    missing_gpu_intervals = submitted_ids - gpu_duration_by_command.keys()
    if gpu_duration_by_command and missing_gpu_intervals:
        warnings.append(
            f"{len(missing_gpu_intervals)} submitted command buffer(s) had no top-level "
            "GPU interval, usually at a trace boundary"
        )

    return {
        "format": FORMAT,
        "trace": str(trace_path) if trace_path else None,
        "process": {"pid": pid, "name": process_name},
        "device": {
            "name": device_name,
            "vendor": device_row.get("vendor-name").display
            if device_row.get("vendor-name")
            else None,
            "recommended_max_working_set": device_row.get(
                "recommended-max-working-set-size"
            ).display
            if device_row.get("recommended-max-working-set-size")
            else None,
        },
        "command_buffers": {
            "submitted": len(submission_rows) if submission_table is not None else None,
            "completed": len(completed_ids)
            if tables["completed"] is not None and submission_table is not None
            else None,
            "errored": len(error_ids)
            if tables["errors"] is not None and submission_table is not None
            else None,
            "with_gpu_intervals": len(gpu_duration_by_command),
            "without_gpu_intervals": len(missing_gpu_intervals),
            "encoder_count": sum(valid_encoder_counts) if valid_encoder_counts else None,
            "labels": [
                {"label": label, "count": count}
                for label, count in sorted(labels.items(), key=lambda item: (-item[1], item[0]))
            ],
            "submission_duration_total_ms": milliseconds(
                sum(valid_submission_duration)
            )
            if valid_submission_duration
            else None,
            "encoder_duration_total_ms": milliseconds(sum(valid_encoder_duration))
            if valid_encoder_duration
            else None,
        },
        "gpu": {
            "top_level_intervals": gpu_interval_count,
            "active_total_ms": milliseconds(gpu_total) if gpu_duration_by_command else None,
            "per_command_buffer_ms": {
                "p50": milliseconds(percentile(per_command, 0.50)),
                "p95": milliseconds(percentile(per_command, 0.95)),
                "max": milliseconds(max(per_command)) if per_command else None,
            },
        },
        "benchmark": load_benchmark(benchmark_path, warnings),
        "evidence": {
            name: str(evidence_dir / filename)
            if (evidence_dir / filename).exists()
            else None
            for name, filename in EVIDENCE_FILES.items()
        },
        "warnings": warnings,
    }


def print_summary(summary: dict[str, Any]) -> None:
    command_buffers = summary["command_buffers"]
    gpu = summary["gpu"]
    process = summary["process"]
    device = summary["device"]
    labels = ", ".join(
        f"{entry['label']}={entry['count']}" for entry in command_buffers["labels"][:6]
    ) or "unavailable"
    print("Metal trace summary")
    print(f"  process: {process['name'] or 'unknown'} (pid {process['pid']})")
    print(f"  device: {device['name'] or 'unknown'}")
    print(
        "  command buffers: "
        f"submitted={command_buffers['submitted']} "
        f"completed={command_buffers['completed']} "
        f"errored={command_buffers['errored']}"
    )
    print(f"  labels: {labels}")
    print(
        "  GPU: "
        f"intervals={gpu['top_level_intervals']} "
        f"active_total_ms={gpu['active_total_ms']} "
        f"per_cb_p50_ms={gpu['per_command_buffer_ms']['p50']} "
        f"per_cb_p95_ms={gpu['per_command_buffer_ms']['p95']}"
    )
    benchmark = summary.get("benchmark")
    if benchmark:
        print(
            "  benchmark: "
            f"items={benchmark['measured_items']} "
            f"throughput={benchmark['throughput_items_per_second']} item/s "
            f"p50={benchmark['latency_ms']['p50']} ms "
            f"p95={benchmark['latency_ms']['p95']} ms"
        )
    for warning in summary["warnings"]:
        print(f"  warning: {warning}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    try:
        summary = build_summary(
            args.evidence_dir, args.pid, args.benchmark, args.trace
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (OSError, SummaryError) as error:
        print(f"summarize_metal_trace: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print_summary(summary)


if __name__ == "__main__":
    main()
