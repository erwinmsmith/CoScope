"""
Run a full CoScope build job and export shareable reasoning-trace artifacts.

This script is intended for "run it all, save the traces, and send them" jobs.
It wraps three steps into one command:

1. Build JSONL shards with `main.py`
2. Export one markdown trace per episode plus index/combined markdown
3. Optionally copy source shards into the bundle and zip everything

Example
-------
Run the full MuSiQue dev CoT job with DashScope/Qwen and export a shareable bundle:

    python -m scripts.run_trace_job \
        --dataset musique \
        --split dev \
        --reasoning-path-type cot \
        --graph-types LINEAR \
        --llm dashscope \
        --llm-model qwen-plus \
        --processed-dir data/processed \
        --export-root data/exports \
        --combined-file all_traces.md \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from main import main as run_main
from scripts.export_trace_bundle import export_trace_bundle


logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    out: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch.lower())
        else:
            out.append("_")
    return "".join(out).strip("_") or "job"


def _default_bundle_name(args: argparse.Namespace) -> str:
    graph_part = "-".join(_slugify(item) for item in args.graph_types)
    llm_part = _slugify(args.llm_model if args.llm != "template" else "template")
    return (
        f"{_slugify(args.dataset)}_{_slugify(args.split)}_"
        f"{_slugify(args.reasoning_path_type)}_{graph_part}_{llm_part}_trace_bundle"
    )


def _bundle_dir(args: argparse.Namespace) -> Path:
    name = args.bundle_name or _default_bundle_name(args)
    return Path(args.export_root) / name


def _trace_input_dir(args: argparse.Namespace) -> Path:
    return Path(args.processed_dir) / args.reasoning_path_type / args.dataset / args.split


def _build_main_argv(args: argparse.Namespace) -> List[str]:
    argv: List[str] = [
        "--datasets", args.dataset,
        "--splits", args.split,
        "--reasoning-path-type", args.reasoning_path_type,
        "--graph-types", *args.graph_types,
        "--llm", args.llm,
        "--llm-model", args.llm_model,
        "--embedder", args.embedder,
        "--embedder-model", args.embedder_model,
        "--embedder-dim", str(args.embedder_dim),
        "--processed-dir", args.processed_dir,
        "--data-dir", args.data_dir,
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
        "--log-level", args.log_level,
    ]
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    return argv


def _ensure_output_target(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Output path already exists: {path}. Pass --overwrite to replace it."
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _copy_jsonl_shards(input_dir: Path, bundle_dir: Path) -> List[str]:
    shard_dir = bundle_dir / "jsonl_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for shard in sorted(input_dir.rglob("*.jsonl")):
        target = shard_dir / shard.name
        shutil.copy2(shard, target)
        copied.append(str(target))
    return copied


def _build_bundle_summary(
    *,
    args: argparse.Namespace,
    build_rc: int,
    trace_input_dir: Path,
    bundle_dir: Path,
    export_summary: Dict[str, Any],
    zip_path: Path | None,
    copied_shards: List[str],
) -> Dict[str, Any]:
    return {
        "dataset": args.dataset,
        "split": args.split,
        "reasoning_path_type": args.reasoning_path_type,
        "graph_types": args.graph_types,
        "llm": args.llm,
        "llm_model": args.llm_model,
        "processed_dir": str(trace_input_dir),
        "bundle_dir": str(bundle_dir),
        "zip_path": str(zip_path) if zip_path else None,
        "episodes_exported": export_summary["episodes_exported"],
        "jsonl_files": export_summary["jsonl_files"],
        "copied_shards": copied_shards,
        "index_path": export_summary["index_path"],
        "combined_path": export_summary["combined_path"],
        "build_return_code": build_rc,
    }


def _write_bundle_summary(bundle_dir: Path, summary: Dict[str, Any]) -> Path:
    path = bundle_dir / "bundle_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_trace_job(args: argparse.Namespace) -> int:
    load_dotenv()

    trace_input_dir = _trace_input_dir(args)
    bundle_dir = _bundle_dir(args)
    zip_path = bundle_dir.with_suffix(".zip") if args.make_zip else None

    _ensure_output_target(bundle_dir, overwrite=args.overwrite)
    if zip_path:
        _ensure_output_target(zip_path, overwrite=args.overwrite)

    build_rc = 0
    if not args.skip_build:
        logger.info("Starting build job for %s/%s", args.dataset, args.split)
        build_rc = run_main(_build_main_argv(args))
        if build_rc != 0:
            logger.error("Build failed with return code %d", build_rc)
            return build_rc
    else:
        logger.info("Skipping build step and reusing existing processed output.")

    if not trace_input_dir.exists():
        raise FileNotFoundError(f"Processed output directory not found: {trace_input_dir}")

    export_summary = export_trace_bundle(
        input_path=trace_input_dir,
        output_dir=bundle_dir,
        combined_file=args.combined_file,
    )

    copied_shards: List[str] = []
    if args.include_shards:
        copied_shards = _copy_jsonl_shards(trace_input_dir, bundle_dir)

    summary = _build_bundle_summary(
        args=args,
        build_rc=build_rc,
        trace_input_dir=trace_input_dir,
        bundle_dir=bundle_dir,
        export_summary=export_summary,
        zip_path=zip_path,
        copied_shards=copied_shards,
    )
    summary_path = _write_bundle_summary(bundle_dir, summary)

    if zip_path:
        archive_base = str(bundle_dir)
        shutil.make_archive(archive_base, "zip", root_dir=bundle_dir)

    print("Trace job completed.")
    print(f"Processed output: {trace_input_dir}")
    print(f"Trace bundle: {bundle_dir}")
    print(f"Summary: {summary_path}")
    print(f"Index: {export_summary['index_path']}")
    if export_summary["combined_path"]:
        print(f"Combined markdown: {export_summary['combined_path']}")
    if zip_path:
        print(f"Zip archive: {zip_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a full CoScope build and export a shareable trace bundle."
    )
    parser.add_argument("--dataset", required=True,
                        choices=["musique", "2wikimhqa", "hotpotqa", "gsm8k", "math"])
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--reasoning-path-type", default="cot", choices=["got", "cot", "tot"])
    parser.add_argument("--graph-types", nargs="+", default=["LINEAR"])

    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--export-root", default="data/exports")
    parser.add_argument("--bundle-name", default=None)
    parser.add_argument("--combined-file", default="all_traces.md")

    parser.add_argument("--llm", default="dashscope", choices=["template", "dashscope"])
    parser.add_argument("--llm-model", default="qwen-plus")
    parser.add_argument("--embedder", default="none", choices=["none", "dashscope"])
    parser.add_argument("--embedder-model", default="text-embedding-v3")
    parser.add_argument("--embedder-dim", type=int, default=1024)

    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--include-shards", action="store_true", default=True)
    parser.add_argument("--no-include-shards", dest="include_shards", action="store_false")
    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-make-zip", dest="make_zip", action="store_false")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_trace_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
