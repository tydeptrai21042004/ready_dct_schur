from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from dct_schur.config import SchurConfig
from dct_schur.philosophy import philosophy_summary
from dct_schur.provenance.keys import save_private_key, save_public_key
from pipelines.audit import write_audit
from pipelines.batch import embed_folder_payload
from pipelines.baseline import embed_baseline_image, extract_baseline_image
from pipelines.benchmark import run_comparison_benchmark
from pipelines.checkpointed import run_checkpointed_benchmark
from pipelines.data import embed_payload_file, extract_payload_file
from pipelines.document import embed_document_pages, verify_document_pages
from pipelines.image import embed_image, extract_image
from pipelines.provenance import embed_provenance, verify_provenance
from pipelines.video import embed_video_provenance, verify_video_provenance
from benchmarking.registry import all_method_specs
from dataclasses import asdict


def _config(path: str | None) -> SchurConfig | None:
    if path is None:
        return None
    return SchurConfig.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def _print(value: Any) -> None:
    if hasattr(value, "__dict__"):
        value = value.__dict__
    print(json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dct-schur",
        description="Invariant-relational DCT-Schur watermarking pipelines",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("philosophy", help="Show the mathematical core philosophy")
    sub.add_parser("list-methods", help="List the DCT-Schur proposal and all published baselines")

    keygen = sub.add_parser("keygen", help="Generate an Ed25519 provenance key pair")
    keygen.add_argument("--private", required=True)
    keygen.add_argument("--public", required=True)

    image_embed = sub.add_parser("image-embed")
    image_embed.add_argument("--host", required=True)
    image_embed.add_argument("--payload", required=True)
    image_embed.add_argument("--output", required=True)
    image_embed.add_argument("--key", required=True)
    image_embed.add_argument("--config")

    image_extract = sub.add_parser("image-extract")
    image_extract.add_argument("--image", required=True)
    image_extract.add_argument("--key", required=True)
    image_extract.add_argument("--output", required=True)

    baseline_embed = sub.add_parser("baseline-embed", help="Embed a binary watermark with one baseline")
    baseline_embed.add_argument("--method", required=True)
    baseline_embed.add_argument("--host", required=True)
    baseline_embed.add_argument("--payload", required=True)
    baseline_embed.add_argument("--output", required=True)
    baseline_embed.add_argument("--key", required=True)
    baseline_embed.add_argument("--seed", type=int, default=2026)
    baseline_embed.add_argument("--repeat", default="1")
    baseline_embed.add_argument("--step", type=float)

    baseline_extract = sub.add_parser("baseline-extract", help="Extract a baseline watermark from a saved key bundle")
    baseline_extract.add_argument("--image", required=True)
    baseline_extract.add_argument("--key", required=True)
    baseline_extract.add_argument("--output", required=True)

    data_embed = sub.add_parser("data-embed")
    data_embed.add_argument("--host", required=True)
    data_embed.add_argument("--payload", required=True)
    data_embed.add_argument("--output", required=True)
    data_embed.add_argument("--key", required=True)
    data_embed.add_argument("--config")

    data_extract = sub.add_parser("data-extract")
    data_extract.add_argument("--image", required=True)
    data_extract.add_argument("--key", required=True)
    data_extract.add_argument("--output", required=True)

    provenance_embed = sub.add_parser("provenance-embed")
    provenance_embed.add_argument("--host", required=True)
    provenance_embed.add_argument("--private-key", required=True)
    provenance_embed.add_argument("--output", required=True)
    provenance_embed.add_argument("--watermark-key", required=True)
    provenance_embed.add_argument("--record", required=True)
    provenance_embed.add_argument("--manifest")
    provenance_embed.add_argument("--config")

    provenance_verify = sub.add_parser("provenance-verify")
    provenance_verify.add_argument("--image", required=True)
    provenance_verify.add_argument("--watermark-key", required=True)
    provenance_verify.add_argument("--public-key", required=True)

    batch = sub.add_parser("batch-data")
    batch.add_argument("--input-folder", required=True)
    batch.add_argument("--payload", required=True)
    batch.add_argument("--output-folder", required=True)
    batch.add_argument("--config")


    document_embed = sub.add_parser("document-embed")
    document_embed.add_argument("--pages", required=True)
    document_embed.add_argument("--output-folder", required=True)
    document_embed.add_argument("--private-key", required=True)
    document_embed.add_argument("--manifest")
    document_embed.add_argument("--config")

    document_verify = sub.add_parser("document-verify")
    document_verify.add_argument("--bundle", required=True)
    document_verify.add_argument("--public-key", required=True)

    video_embed = sub.add_parser("video-embed")
    video_embed.add_argument("--input", required=True)
    video_embed.add_argument("--output", required=True)
    video_embed.add_argument("--private-key", required=True)
    video_embed.add_argument("--key-bundle", required=True)
    video_embed.add_argument("--stride", type=int, default=1)
    video_embed.add_argument("--manifest")
    video_embed.add_argument("--config")

    video_verify = sub.add_parser("video-verify")
    video_verify.add_argument("--video", required=True)
    video_verify.add_argument("--key-bundle", required=True)
    video_verify.add_argument("--public-key", required=True)

    benchmark = sub.add_parser("benchmark", help="Compare DCT-Schur with selectable published baselines")
    benchmark.add_argument("--host", required=True, help="Host image or folder")
    benchmark.add_argument("--payload", required=True, help="Payload image or folder")
    benchmark.add_argument("--methods", default="all", help="all, baselines, blind, semi_blind, key_assisted, non_blind, dct_schur, or comma-separated IDs")
    benchmark.add_argument("--output-json", required=True)
    benchmark.add_argument("--output-summary-csv")
    benchmark.add_argument("--output-attack-csv")
    benchmark.add_argument("--suite", default="extended")
    benchmark.add_argument("--config")
    benchmark.add_argument("--baseline-parameters", default="configs/baseline_parameters.json")
    benchmark.add_argument("--seed", type=int, default=2026)
    benchmark.add_argument("--strict", action="store_true", help="Stop on the first method or attack error")

    checkpointed = sub.add_parser("benchmark-checkpointed", help="Run and resume one benchmark checkpoint per method")
    checkpointed.add_argument("--host", required=True)
    checkpointed.add_argument("--payload", required=True)
    checkpointed.add_argument("--methods", default="all")
    checkpointed.add_argument("--suite", default="extended")
    checkpointed.add_argument("--output-directory", required=True)
    checkpointed.add_argument("--config")
    checkpointed.add_argument("--baseline-parameters", default="configs/baseline_parameters.json")
    checkpointed.add_argument("--seed", type=int, default=2026)
    checkpointed.add_argument("--no-resume", action="store_true")
    checkpointed.add_argument("--strict", action="store_true")

    audit = sub.add_parser("audit")
    audit.add_argument("--root", default=".")
    audit.add_argument("--output", default="results/repository_audit.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "philosophy":
        _print(philosophy_summary())
    elif args.command == "list-methods":
        _print([asdict(spec) for spec in all_method_specs()])
    elif args.command == "keygen":
        private_key = Ed25519PrivateKey.generate()
        save_private_key(args.private, private_key)
        save_public_key(args.public, private_key.public_key())
        _print({"private": args.private, "public": args.public})
    elif args.command == "image-embed":
        _print(embed_image(args.host, args.payload, args.output, args.key, config=_config(args.config)))
    elif args.command == "image-extract":
        _print({"output": extract_image(args.image, args.key, args.output)})
    elif args.command == "baseline-embed":
        repeat = args.repeat if args.repeat == "auto" else int(args.repeat)
        _print(embed_baseline_image(
            args.method, args.host, args.payload, args.output, args.key,
            seed=args.seed, repeat=repeat, step=args.step,
        ))
    elif args.command == "baseline-extract":
        _print({"output": extract_baseline_image(args.image, args.key, args.output)})
    elif args.command == "data-embed":
        _print(embed_payload_file(args.host, args.payload, args.output, args.key, config=_config(args.config)))
    elif args.command == "data-extract":
        _print({"output": extract_payload_file(args.image, args.key, args.output)})
    elif args.command == "provenance-embed":
        _print(embed_provenance(
            args.host, args.private_key, args.output, args.watermark_key,
            args.record, manifest_path=args.manifest, config=_config(args.config),
        ))
    elif args.command == "provenance-verify":
        _print(verify_provenance(args.image, args.watermark_key, args.public_key))
    elif args.command == "batch-data":
        _print({"manifest": embed_folder_payload(
            args.input_folder, args.payload, args.output_folder,
            config=_config(args.config),
        )})
    elif args.command == "document-embed":
        _print({"bundle": embed_document_pages(
            args.pages, args.output_folder, args.private_key,
            manifest_path=args.manifest, config=_config(args.config),
        )})
    elif args.command == "document-verify":
        _print(verify_document_pages(args.bundle, args.public_key))
    elif args.command == "video-embed":
        _print({"key_bundle": embed_video_provenance(
            args.input, args.output, args.private_key, args.key_bundle,
            frame_stride=args.stride, manifest_path=args.manifest,
            config=_config(args.config),
        )})
    elif args.command == "video-verify":
        _print(verify_video_provenance(args.video, args.key_bundle, args.public_key))
    elif args.command == "benchmark":
        _print(run_comparison_benchmark(
            args.host, args.payload, args.output_json,
            methods=args.methods,
            output_summary_csv=args.output_summary_csv,
            output_attack_csv=args.output_attack_csv,
            attack_suite=args.suite,
            config=_config(args.config),
            baseline_parameters_path=args.baseline_parameters,
            seed=args.seed,
            continue_on_error=not args.strict,
        ))
    elif args.command == "benchmark-checkpointed":
        _print(run_checkpointed_benchmark(
            args.host, args.payload, args.output_directory,
            methods=args.methods, attack_suite=args.suite,
            config=_config(args.config),
            baseline_parameters_path=args.baseline_parameters,
            seed=args.seed, resume=not args.no_resume,
            continue_on_error=not args.strict,
        ))
    elif args.command == "audit":
        _print({"output": write_audit(args.root, args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
