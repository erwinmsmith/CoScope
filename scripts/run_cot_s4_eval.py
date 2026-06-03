"""
Run the CoT S4 workflow end-to-end.

This helper is intentionally narrow: it targets the CoT `POLICY_ISOLATED`
branch, which is the current project path for S4 / verifier / safety analysis.

It wraps three steps:

1. build `POLICY_ISOLATED` CoT episodes with `main.py`
2. evaluate the resulting `s4_policy_isolated.jsonl`
3. render a compact markdown brief from the evaluation summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from main import main as run_main
from scripts.eval_jsonl import main as run_eval_main
from scripts.render_eval_brief import main as run_brief_main


logger = logging.getLogger(__name__)


def _build_argv(args: argparse.Namespace) -> List[str]:
    argv: List[str] = [
        "--datasets", args.dataset,
        "--splits", args.split,
        "--reasoning-path-type", "cot",
        "--graph-types", "POLICY_ISOLATED",
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


def _s4_jsonl_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.processed_dir)
        / "cot"
        / args.dataset
        / args.split
        / "s4_policy_isolated.jsonl"
    )


def _eval_argv(args: argparse.Namespace, s4_jsonl: Path) -> List[str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--shards", str(s4_jsonl),
        "--embedding-dim", str(args.query_dim),
        "--embedder", args.eval_embedder,
        "--variants", *args.variants,
        "--output", str(output_dir / "summary.json"),
        "--log-level", args.eval_log_level,
    ]
    if args.eval_embedder == "dashscope":
        argv.extend([
            "--dashscope-model", args.embedder_model,
            "--dashscope-dim", str(args.embedder_dim),
        ])
    return argv


def _brief_argv(args: argparse.Namespace) -> List[str]:
    summary_json = Path(args.output_dir) / "summary.json"
    brief_md = Path(args.output_dir) / "brief.md"
    return [
        "--summary-json", str(summary_json),
        "--title", args.title,
        "--output", str(brief_md),
    ]


def _call_cli_main(main_func, argv: List[str]) -> int:
    """Call a no-argv CLI main by temporarily replacing sys.argv."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return main_func()
    finally:
        sys.argv = old_argv


def run_workflow(args: argparse.Namespace) -> int:
    load_dotenv()

    if not args.skip_build:
        logger.info("Building/resuming CoT POLICY_ISOLATED episodes for S4.")
        rc = run_main(_build_argv(args))
        if rc != 0:
            logger.error("Build step failed with return code %d", rc)
            return rc
    else:
        logger.info("Skipping build step and reusing existing S4 shard.")

    s4_jsonl = _s4_jsonl_path(args)
    if not s4_jsonl.exists():
        raise FileNotFoundError(
            f"S4 shard not found: {s4_jsonl}. Run the build step first."
        )

    logger.info("Evaluating CoT S4 retrieval variants from %s", s4_jsonl)
    rc = _call_cli_main(run_eval_main, _eval_argv(args, s4_jsonl))
    if rc != 0:
        logger.error("Evaluation step failed with return code %d", rc)
        return rc

    logger.info("Rendering markdown brief.")
    rc = run_brief_main(_brief_argv(args))
    if rc != 0:
        logger.error("Brief rendering failed with return code %d", rc)
        return rc

    print("CoT S4 workflow completed.")
    print(f"S4 shard: {s4_jsonl}")
    print(f"Evaluation directory: {args.output_dir}")
    print(f"Summary: {Path(args.output_dir) / 'summary.json'}")
    print(f"Brief: {Path(args.output_dir) / 'brief.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the CoT POLICY_ISOLATED / S4 workflow."
    )
    parser.add_argument("--dataset", default="musique",
                        choices=["musique", "2wikimhqa", "hotpotqa", "gsm8k", "math"])
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])

    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/eval/cot_s4_eval")

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

    parser.add_argument("--skip-build", action="store_true")

    parser.add_argument("--query-dim", type=int, default=256)
    parser.add_argument(
        "--eval-embedder",
        default="random",
        choices=["random", "st", "dashscope", "default"],
        help="Embedder used by scripts.eval_jsonl during retrieval evaluation.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["a1", "a3", "a4", "a4_nofb", "a4_norerank", "a5", "a5_noproj", "a5_norerank", "a6", "a7", "a8"],
    )
    parser.add_argument(
        "--title",
        default="MuSiQue CoT S4 Retrieval Ablation",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--eval-log-level", default="ERROR",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
