#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_metal_trace as summary


def table(schema: str, columns: list[tuple[str, str]], rows: str) -> str:
    encoded_columns = "".join(
        f"<col><mnemonic>{mnemonic}</mnemonic><name>{mnemonic}</name>"
        f"<engineering-type>{kind}</engineering-type></col>"
        for mnemonic, kind in columns
    )
    return (
        "<?xml version='1.0'?><trace-query-result><node>"
        f"<schema name='{schema}'>{encoded_columns}</schema>{rows}"
        "</node></trace-query-result>"
    )


class SummaryTests(unittest.TestCase):
    def write(self, directory: Path, name: str, value: str) -> None:
        (directory / summary.EVIDENCE_FILES[name]).write_text(value)

    def test_summarizes_target_process_and_excludes_nested_gpu_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write(
                directory,
                "submissions",
                table(
                    "metal-application-command-buffer-submissions",
                    [
                        ("duration", "duration"),
                        ("gpu", "metal-device-name"),
                        ("track-label", "metal-object-label"),
                        ("num-encoders", "uint32"),
                        ("encoder-time", "duration"),
                        ("process", "process"),
                        ("event-label", "narrative"),
                        ("cmdbuffer-id", "metal-command-buffer-id"),
                    ],
                    "<row><duration id='1' fmt='1 ms'>1000000</duration>"
                    "<metal-device-name id='2' fmt='M1'>M1</metal-device-name><sentinel/>"
                    "<uint32 id='3'>2</uint32><duration id='4'>900000</duration>"
                    "<process id='5' fmt='server (42)'><pid>42</pid></process>"
                    "<narrative><metal-object-label id='6'>encode</metal-object-label></narrative>"
                    "<metal-command-buffer-id id='7'>0x1</metal-command-buffer-id></row>"
                    "<row><duration ref='1'/><metal-device-name ref='2'/><sentinel/>"
                    "<uint32 ref='3'/><duration ref='4'/><process ref='5'/>"
                    "<narrative><metal-object-label id='8'>decode</metal-object-label></narrative>"
                    "<metal-command-buffer-id id='9'>0x2</metal-command-buffer-id></row>",
                ),
            )
            self.write(
                directory,
                "gpu_intervals",
                table(
                    "metal-gpu-intervals",
                    [
                        ("duration", "duration"),
                        ("event-depth", "metal-nesting-level"),
                        ("process", "process"),
                        ("cmdbuffer-id", "metal-command-buffer-id"),
                    ],
                    "<row><duration>1000000</duration><metal-nesting-level>0</metal-nesting-level>"
                    "<process id='10' fmt='server (42)'><pid>42</pid></process>"
                    "<metal-command-buffer-id>0x1</metal-command-buffer-id></row>"
                    "<row><duration>500000</duration><metal-nesting-level>1</metal-nesting-level>"
                    "<process ref='10'/><metal-command-buffer-id>0x1</metal-command-buffer-id></row>"
                    "<row><duration>2000000</duration><metal-nesting-level>0</metal-nesting-level>"
                    "<process ref='10'/><metal-command-buffer-id>0x2</metal-command-buffer-id></row>"
                    "<row><duration>9000000</duration><metal-nesting-level>0</metal-nesting-level>"
                    "<process fmt='other (99)'><pid>99</pid></process>"
                    "<metal-command-buffer-id>0x9</metal-command-buffer-id></row>",
                ),
            )
            self.write(
                directory,
                "completed",
                table(
                    "metal-command-buffer-completed",
                    [("cmdbuffer-id", "metal-command-buffer-id")],
                    "<row><metal-command-buffer-id>0x1</metal-command-buffer-id></row>"
                    "<row><metal-command-buffer-id>0x2</metal-command-buffer-id></row>"
                    "<row><metal-command-buffer-id>0x9</metal-command-buffer-id></row>",
                ),
            )
            self.write(
                directory,
                "errors",
                table(
                    "metal-command-buffer-error",
                    [("cmdbuffer-id", "metal-command-buffer-id")],
                    "<row><metal-command-buffer-id>0x2</metal-command-buffer-id></row>"
                    "<row><metal-command-buffer-id>0x9</metal-command-buffer-id></row>",
                ),
            )
            self.write(
                directory,
                "device",
                table(
                    "device-gpu-info",
                    [
                        ("device-name", "metal-object-label"),
                        ("vendor-name", "metal-object-label"),
                        ("recommended-max-working-set-size", "size-in-bytes"),
                    ],
                    "<row><metal-object-label>M1</metal-object-label>"
                    "<metal-object-label>Apple</metal-object-label>"
                    "<size-in-bytes fmt='12 GiB'>12884901888</size-in-bytes></row>",
                ),
            )

            report = summary.build_summary(directory, 42)

            self.assertEqual(report["command_buffers"]["submitted"], 2)
            self.assertEqual(report["command_buffers"]["completed"], 2)
            self.assertEqual(report["command_buffers"]["errored"], 1)
            self.assertEqual(report["command_buffers"]["encoder_count"], 4)
            self.assertEqual(
                report["command_buffers"]["labels"],
                [{"label": "decode", "count": 1}, {"label": "encode", "count": 1}],
            )
            self.assertEqual(report["gpu"]["top_level_intervals"], 2)
            self.assertEqual(report["gpu"]["active_total_ms"], 3.0)
            self.assertEqual(report["gpu"]["per_command_buffer_ms"]["p50"], 1.0)
            self.assertEqual(report["gpu"]["per_command_buffer_ms"]["p95"], 2.0)
            self.assertEqual(report["device"]["name"], "M1")
            self.assertEqual(report["warnings"], [])

    def test_missing_optional_fields_degrade_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write(
                directory,
                "submissions",
                table(
                    "metal-application-command-buffer-submissions",
                    [("process", "process"), ("cmdbuffer-id", "metal-command-buffer-id")],
                    "<row><process fmt='server (7)'><pid>7</pid></process>"
                    "<metal-command-buffer-id>11</metal-command-buffer-id></row>",
                ),
            )
            self.write(
                directory,
                "gpu_intervals",
                table(
                    "metal-gpu-intervals",
                    [("duration", "duration"), ("cmdbuffer-id", "metal-command-buffer-id")],
                    "<row><duration fmt='2 ms'/><metal-command-buffer-id>11</metal-command-buffer-id></row>",
                ),
            )

            report = summary.build_summary(directory, 7)

            self.assertEqual(report["gpu"]["active_total_ms"], 2.0)
            self.assertTrue(
                any("filtered by command-buffer ID" in item for item in report["warnings"])
            )
            self.assertTrue(any("depth is unavailable" in item for item in report["warnings"]))
            self.assertIsNone(report["command_buffers"]["completed"])
            self.assertIsNone(report["command_buffers"]["encoder_count"])
            self.assertIsNone(report["command_buffers"]["submission_duration_total_ms"])

    def test_corrupt_present_evidence_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write(directory, "submissions", "<not-closed>")
            with self.assertRaises(summary.SummaryError):
                summary.build_summary(directory, 1)


if __name__ == "__main__":
    unittest.main()
