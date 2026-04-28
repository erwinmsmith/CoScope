"""
Batch-export report-friendly reasoning traces from CoScope JSONL shards.

This script turns one JSONL shard or a directory of shards into:

1. One markdown trace per episode
2. An index markdown file with links / metadata
3. An optional combined markdown file with all traces appended

Examples
--------
Export one shard:

    python -m coscope.scripts.export_trace_bundle \
        --input coscope/data/processed_qwen_smoke/cot/musique/dev/s1_linear.jsonl \
        --output-dir coscope/data/exports/musique_dev_traces

Export a whole split directory and also emit a combined markdown file:

    python -m coscope.scripts.export_trace_bundle \
        --input coscope/data/processed_qwen_smoke/cot/musique/dev \
        --output-dir coscope/data/exports/musique_dev_traces \
        --combined-file all_traces.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from coscope.scripts.render_trace import render_episode_markdown


def _discover_jsonl_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.rglob("*.jsonl"))
        if not files:
            raise ValueError(f"No JSONL files found under: {input_path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _load_episodes(jsonl_path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            episodes.append(json.loads(line))
    return episodes


def _safe_filename(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "episode"


def _write_episode_trace(
    *,
    episode: Dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_id = str(episode.get("episode_id", "episode"))
    path = output_dir / f"{_safe_filename(episode_id)}.md"
    path.write_text(render_episode_markdown(episode), encoding="utf-8")
    return path


def _episode_summary_row(
    *,
    episode: Dict[str, Any],
    shard_path: Path,
    trace_path: Path,
    bundle_root: Path,
) -> str:
    rel_trace = trace_path.relative_to(bundle_root).as_posix()
    question = str(episode.get("question", "")).replace("\n", " ").replace("|", "\\|")
    question = question[:120] + ("..." if len(question) > 120 else "")
    return (
        f"| `{episode.get('episode_id', '')}` "
        f"| `{episode.get('original_id', '')}` "
        f"| `{episode.get('reasoning_path_type', '')}` "
        f"| `{episode.get('graph_type', '')}` "
        f"| `{episode.get('rho_subset', '')}` "
        f"| `{episode.get('rho', '')}` "
        f"| `{shard_path.name}` "
        f"| [trace]({rel_trace}) "
        f"| {question} |"
    )


def _build_index_markdown(rows: Iterable[str]) -> str:
    lines = [
        "# Trace Bundle Index",
        "",
        "| Episode ID | Original ID | Reasoning | Graph | Subset | Rho | Source Shard | Trace File | Question |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(rows)
    lines.append("")
    return "\n".join(lines)


def _build_combined_markdown(entries: Iterable[tuple[Path, Dict[str, Any]]]) -> str:
    chunks: List[str] = ["# Combined Trace Bundle", ""]
    for shard_path, episode in entries:
        chunks.append(
            f"---\n\n"
            f"Source shard: `{shard_path}`\n"
        )
        chunks.append(render_episode_markdown(episode).rstrip())
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def export_trace_bundle(
    *,
    input_path: Path,
    output_dir: Path,
    combined_file: str | None = None,
) -> Dict[str, Any]:
    jsonl_files = _discover_jsonl_files(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[str] = []
    combined_entries: List[tuple[Path, Dict[str, Any]]] = []
    exported = 0

    for jsonl_path in jsonl_files:
        shard_stem = _safe_filename(jsonl_path.stem)
        shard_output_dir = output_dir / shard_stem
        episodes = _load_episodes(jsonl_path)
        for episode in episodes:
            trace_path = _write_episode_trace(
                episode=episode,
                output_dir=shard_output_dir,
            )
            index_rows.append(
                _episode_summary_row(
                    episode=episode,
                    shard_path=jsonl_path,
                    trace_path=trace_path,
                    bundle_root=output_dir,
                )
            )
            combined_entries.append((jsonl_path, episode))
            exported += 1

    index_path = output_dir / "index.md"
    index_path.write_text(_build_index_markdown(index_rows), encoding="utf-8")

    combined_path = None
    if combined_file:
        combined_path = output_dir / combined_file
        combined_path.write_text(
            _build_combined_markdown(combined_entries),
            encoding="utf-8",
        )

    return {
        "jsonl_files": [str(path) for path in jsonl_files],
        "episodes_exported": exported,
        "index_path": str(index_path),
        "combined_path": str(combined_path) if combined_path else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export markdown trace bundle from JSONL shard(s)")
    parser.add_argument(
        "--input",
        required=True,
        help="A JSONL shard file or a directory containing shard files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for markdown traces and index files.",
    )
    parser.add_argument(
        "--combined-file",
        default=None,
        help="Optional combined markdown filename, e.g. all_traces.md",
    )
    args = parser.parse_args()

    summary = export_trace_bundle(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        combined_file=args.combined_file,
    )

    print(f"Exported {summary['episodes_exported']} episode trace(s).")
    print(f"Index: {summary['index_path']}")
    if summary["combined_path"]:
        print(f"Combined: {summary['combined_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
