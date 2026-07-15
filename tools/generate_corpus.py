#!/usr/bin/env python3
"""Generate the deterministic 200-item optimization and differential corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ADJECTIVES = ("small", "large", "quiet", "bright", "ancient", "modern", "careful", "unexpected")
NOUNS = ("train", "window", "compiler", "teacher", "river", "market", "satellite", "library")
VERBS = ("crosses", "opens", "tests", "observes", "builds", "explains", "finds", "protects")
ENDINGS = ("today.", "at noon.", "without warning!", "near the station?")
EDGE_CASES = (
    "Hello, world!",
    "Numbers: 0, 1, 2, 3.14159, and 2026-07-15.",
    "Quotes “like this”, apostrophes, em—dashes, and ellipses… should work.",
    "Café naïve résumé coöperate — Unicode normalization matters.",
    "Emoji test: 🚄🌧️🧪; please keep translating the surrounding English.",
    "Line one\nLine two\tTabbed text.",
    "UPPERCASE and lowercase MixedCase words.",
    "A",
    "I do not know.",
    "Where is platform number twelve?",
    "Rust prevents many memory errors, but careful design still matters.",
    "If the weather changes tomorrow, please close every open window.",
    "The quick brown fox jumps over the lazy dog.",
    "Zero-width? abc\u200bdef and non-breaking\u00a0space.",
    "Repeated punctuation!!!???...",
    "The CPU batch must match each sentence translated alone.",
    "A longer sentence with several clauses, commas, numbers 42 and 9000, plus a final question: does it still agree?",
    "Thank you for your help!",
    "Please open the window.",
    "The weather is beautiful today.",
)


def corpus() -> list[str]:
    result: list[str] = []
    for adjective in ADJECTIVES:
        for noun in NOUNS:
            for verb in VERBS:
                for ending in ENDINGS:
                    index = len(result)
                    result.append(f"{index}: The {adjective} {noun} {verb} the old bridge {ending}")
                    if len(result) == 180:
                        result.extend(EDGE_CASES)
                        assert len(result) == 200
                        return result
    raise AssertionError("template space did not produce 180 items")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks/corpus-v1.jsonl",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps({"id": index, "text": text}, ensure_ascii=False, separators=(",", ":")) + "\n"
        for index, text in enumerate(corpus())
    )
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(f"wrote {len(corpus())} items to {args.output}")


if __name__ == "__main__":
    main()
