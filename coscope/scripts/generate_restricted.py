"""
Offline generator for the restricted memory layer.

Reads raw items for a dataset and writes
`data/interim/restricted/{dataset}_restricted.jsonl`, one line per original_id.
Each line has shape:

    {
        "original_id": "<id>",
        "entries": [
            {
                "content": "...",
                "confidence": 1.0,
                "source": "<paragraph_id or step key>",
                "metadata": { ... optional extra fields ... }
            },
            ...
        ]
    }

QA datasets generate credibility scores per `supporting_facts` sentence.
Math datasets generate per-step numerical-consistency records.

Usage:
    python -m coscope.scripts.generate_restricted \\
        --dataset musique --split train --data-dir data/raw/musique
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from coscope.utils.loaders import get_loader


def _qa_entries(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    sf = raw.get("supporting_facts", []) or []
    gold_ids = {(f.get("paragraph_id"), f.get("sentence_idx")) for f in sf}
    for para in raw.get("supporting_paragraphs", []) or []:
        pid = para.get("paragraph_id")
        entries.append(
            {
                "content": f"{pid} credibility: 1.0 (gold evidence)",
                "confidence": 1.0,
                "source": pid,
                "metadata": {"source_paragraph_id": pid, "kind": "gold_fact"},
            }
        )
    for para in raw.get("distractor_paragraphs", []) or []:
        pid = para.get("paragraph_id")
        entries.append(
            {
                "content": f"{pid} credibility: 0.0 (distractor)",
                "confidence": 0.0,
                "source": pid,
                "metadata": {"source_paragraph_id": pid, "kind": "distractor"},
            }
        )
    # Sentence-level entries when facts are present.
    for fact in sf:
        pid = fact.get("paragraph_id")
        sent_idx = fact.get("sentence_idx", 0)
        entries.append(
            {
                "content": f"{pid} sentence {sent_idx}: credibility=1.0",
                "confidence": 1.0,
                "source": f"{pid}:{sent_idx}",
                "metadata": {
                    "source_paragraph_id": pid,
                    "sentence_idx": int(sent_idx),
                    "kind": "gold_sentence",
                },
            }
        )
    return entries


def _math_entries(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for k, step in enumerate(raw.get("solution_steps", []) or [], start=1):
        entries.append(
            {
                "content": f"step_{k} numerical check: pass, value=<oracle>",
                "confidence": 1.0,
                "source": f"step_{k}",
                "metadata": {"step_index": k, "kind": "numerical_check"},
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate restricted layer for a dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", required=True, help="Path to the dataset's raw root.")
    parser.add_argument(
        "--interim-dir",
        default="data/interim/restricted",
        help="Output directory for <dataset>_restricted.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    loader = get_loader(args.dataset, data_dir=args.data_dir, split=args.split)
    raws = loader.load()
    if args.limit is not None:
        raws = raws[: args.limit]

    out_dir = Path(args.interim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_restricted.jsonl"

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for raw in raws:
            if raw.get("dataset_type") == "math":
                entries = _math_entries(raw)
            else:
                entries = _qa_entries(raw)
            if not entries:
                continue
            f.write(
                json.dumps(
                    {"original_id": raw.get("original_id", ""), "entries": entries},
                    ensure_ascii=False,
                )
            )
            f.write("\n")
            count += 1
    print(f"Wrote {count} restricted records to {out_path}")


if __name__ == "__main__":
    main()
