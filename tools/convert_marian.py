#!/usr/bin/env python3
"""Convert a Firefox Translations Marian NPZ release to runtime safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


MANIFEST_FORMAT = "marian-edge.transformer-ssru.v1"
# Keep the byte-level FP32 artifact stable. This metadata labels the existing
# safetensors layout; changing the manifest namespace must not rewrite 166 MB
# of otherwise identical weights or invalidate its published checksum.
WEIGHTS_FORMAT = "marian-mlx.transformer-ssru.v1"
EXPECTED_MODEL_SHA256 = (
    "9604368d0fb19aa431a82824cedd92205a68512b89086cbe8c4d8bd1585a8950"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Original Marian NPZ")
    parser.add_argument("--source-vocab", type=Path, required=True)
    parser.add_argument("--target-vocab", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="mozilla-firefox-translations-en-zh-base-memory")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        help="Allow an NPZ hash other than the pinned Mozilla release",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_tensor_names() -> set[str]:
    names = {"encoder_Wemb", "decoder_Wemb", "decoder_ff_logit_out_b"}
    for layer in range(1, 7):
        self_prefix = f"encoder_l{layer}_self"
        names.update(f"{self_prefix}_{suffix}" for suffix in (
            "Wq", "Wk", "Wv", "Wo", "bq", "bk", "bv", "bo",
            "Wo_ln_scale", "Wo_ln_bias",
        ))
        ffn_prefix = f"encoder_l{layer}_ffn"
        names.update(f"{ffn_prefix}_{suffix}" for suffix in (
            "W1", "W2", "b1", "b2", "ffn_ln_scale", "ffn_ln_bias",
        ))
    for layer in range(1, 5):
        rnn_prefix = f"decoder_l{layer}_rnn"
        names.update(f"{rnn_prefix}_{suffix}" for suffix in (
            "W", "Wf", "bf", "ffn_ln_scale", "ffn_ln_bias",
        ))
        context_prefix = f"decoder_l{layer}_context"
        names.update(f"{context_prefix}_{suffix}" for suffix in (
            "Wq", "Wk", "Wv", "Wo", "bq", "bk", "bv", "bo",
            "Wo_ln_scale", "Wo_ln_bias",
        ))
        ffn_prefix = f"decoder_l{layer}_ffn"
        names.update(f"{ffn_prefix}_{suffix}" for suffix in (
            "W1", "W2", "b1", "b2", "ffn_ln_scale", "ffn_ln_bias",
        ))
    return names


def validate_graph_config(raw: bytes) -> str:
    config = raw.rstrip(b"\0").decode("utf-8")
    required_lines = (
        "type: transformer",
        "dim-emb: 384",
        "enc-depth: 6",
        "dec-depth: 4",
        "dec-cell: ssru",
        "transformer-decoder-autoreg: rnn",
        "transformer-dim-ffn: 1536",
        "transformer-heads: 8",
        'transformer-preprocess: ""',
        "transformer-postprocess: dan",
        "transformer-postprocess-emb: d",
        "tied-embeddings: true",
    )
    missing = [line for line in required_lines if line not in config]
    if missing:
        raise ValueError(f"NPZ graph is unsupported; missing config: {missing}")
    return config


def validate_shapes(tensors: dict[str, np.ndarray]) -> None:
    required = required_tensor_names()
    actual = set(tensors)
    if missing := sorted(required - actual):
        raise ValueError(f"NPZ is missing {len(missing)} tensors: {missing[:8]}")
    if extras := sorted(actual - required):
        raise ValueError(f"NPZ contains unexpected tensors: {extras[:8]}")
    expected = {
        "encoder_Wemb": (32000, 384),
        "decoder_Wemb": (32000, 384),
        "decoder_ff_logit_out_b": (1, 32000),
    }
    for name, shape in expected.items():
        if tensors[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {tensors[name].shape}")
    for name, tensor in tensors.items():
        if not np.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinity")
        if name.endswith(("_Wq", "_Wk", "_Wv", "_Wo", "_rnn_W", "_rnn_Wf")):
            if tensor.shape != (384, 384):
                raise ValueError(f"{name}: expected (384, 384), got {tensor.shape}")
        if name.endswith("_ffn_W1") and tensor.shape != (384, 1536):
            raise ValueError(f"{name}: expected (384, 1536), got {tensor.shape}")
        if name.endswith("_ffn_W2") and tensor.shape != (1536, 384):
            raise ValueError(f"{name}: expected (1536, 384), got {tensor.shape}")


def copy_asset(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)
    return sha256(destination)


def canonicalize_safetensors_header(path: Path) -> None:
    """Rewrite the JSON header in a byte-stable order.

    safetensors preserves tensor bytes, but its metadata map can be serialized in
    a different key order between processes. Canonical JSON keeps the complete
    file checksum reproducible without changing any tensor offsets or payloads.
    """
    temporary: Path | None = None
    try:
        with path.open("rb") as source:
            size_bytes = source.read(8)
            if len(size_bytes) != 8:
                raise ValueError("safetensors file has a truncated header length")
            (header_size,) = struct.unpack("<Q", size_bytes)
            if header_size == 0 or header_size > 100 * 1024 * 1024:
                raise ValueError(f"invalid safetensors header size: {header_size}")
            header_bytes = source.read(header_size)
            if len(header_bytes) != header_size:
                raise ValueError("safetensors file has a truncated JSON header")
            header = json.loads(header_bytes)
            if not isinstance(header, dict):
                raise ValueError("safetensors header is not a JSON object")

            canonical = json.dumps(
                header,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            canonical += b" " * (-len(canonical) % 8)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".canonical",
                delete=False,
            ) as destination:
                temporary = Path(destination.name)
                destination.write(struct.pack("<Q", len(canonical)))
                destination.write(canonical)
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        if temporary is None:
            raise RuntimeError("failed to create a canonical safetensors file")
        shutil.copymode(path, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    model_hash = sha256(args.model)
    if model_hash != EXPECTED_MODEL_SHA256 and not args.allow_unverified_model:
        raise ValueError(
            f"unexpected NPZ sha256 {model_hash}; expected {EXPECTED_MODEL_SHA256}. "
            "Pass --allow-unverified-model only for a graph you audited."
        )
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.model, allow_pickle=False) as archive:
        if "special:model.yml" not in archive:
            raise ValueError("NPZ has no embedded Marian graph config")
        graph_config = validate_graph_config(archive["special:model.yml"].tobytes())
        tensors = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
            if name != "special:model.yml"
        }
    validate_shapes(tensors)

    dtype = np.float32 if args.dtype == "fp32" else np.float16
    converted = {name: tensor.astype(dtype, copy=False) for name, tensor in tensors.items()}
    weights_name = f"model.{args.dtype}.safetensors"
    weights_path = args.output / weights_name
    save_file(
        converted,
        str(weights_path),
        metadata={
            "format": WEIGHTS_FORMAT,
            "source_npz_sha256": model_hash,
            "marian_config": graph_config,
        },
    )
    canonicalize_safetensors_header(weights_path)

    source_name = "source.spm"
    target_name = "target.spm"
    source_hash = copy_asset(args.source_vocab, args.output / source_name)
    target_hash = copy_asset(args.target_vocab, args.output / target_name)
    shortlist_name = None
    shortlist_hash = None
    if args.shortlist:
        shortlist_name = "shortlist.bin"
        shortlist_hash = copy_asset(args.shortlist, args.output / shortlist_name)

    manifest = {
        "format": MANIFEST_FORMAT,
        "model_id": args.model_id,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "weights": weights_name,
        "source_vocab": source_name,
        "target_vocab": target_name,
        "shortlist": shortlist_name,
        "precision": args.dtype,
        "architecture": {
            "model_dim": 384,
            "attention_heads": 8,
            "encoder_layers": 6,
            "decoder_layers": 4,
            "ffn_dim": 1536,
            "source_vocab_size": 32000,
            "target_vocab_size": 32000,
            "eos_id": 0,
            "unk_id": 1,
            "max_length_factor": 3,
        },
        "checksums": {
            "source_npz_sha256": model_hash,
            "weights_sha256": sha256(weights_path),
            "source_vocab_sha256": source_hash,
            "target_vocab_sha256": target_hash,
            "shortlist_sha256": shortlist_hash,
        },
        "upstream": {
            "project": "Mozilla Firefox Translations",
            "registry": "https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with safe_open(weights_path, framework="np") as loaded:
        if set(loaded.keys()) != set(converted):
            raise RuntimeError("safetensors round-trip tensor names differ")
        for name in ("encoder_Wemb", "decoder_Wemb", "decoder_l4_rnn_Wf"):
            if loaded.get_tensor(name).shape != converted[name].shape:
                raise RuntimeError(f"safetensors round-trip shape differs for {name}")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "weights": str(weights_path),
                "weights_sha256": manifest["checksums"]["weights_sha256"],
                "dtype": args.dtype,
                "tensors": len(converted),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
