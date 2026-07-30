"""Inspect or export a durable cloud experiment checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from coscope.evaluation.checkpoint_store import ExperimentCheckpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dir")
    parser.add_argument("--watch", type=float)
    parser.add_argument("--export-results")
    args = parser.parse_args()
    if args.watch is not None and args.watch <= 0:
        parser.error("--watch must be positive")

    state_dir = Path(args.state_dir).expanduser().resolve()
    database = state_dir / "checkpoint.sqlite3"
    if not database.exists():
        completed_path = state_dir / "COMPLETED.json"
        status_path = state_dir / "status.json"
        if completed_path.exists() and status_path.exists():
            if args.export_results:
                parser.error(
                    "detailed results were purged after successful metric "
                    "finalization; inspect final_metrics.json"
                )
            payload = {
                "completion": json.loads(
                    completed_path.read_text(encoding="utf-8")
                ),
                "status": json.loads(
                    status_path.read_text(encoding="utf-8")
                ),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        parser.error(f"checkpoint not found: {database}")
    checkpoint = ExperimentCheckpoint(database)
    if args.export_results:
        count = checkpoint.export_results_jsonl(args.export_results)
        print(
            json.dumps(
                {"output": args.export_results, "records": count},
                ensure_ascii=False,
            )
        )
        return 0

    while True:
        print(
            json.dumps(
                checkpoint.status_snapshot(),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if args.watch is None:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
