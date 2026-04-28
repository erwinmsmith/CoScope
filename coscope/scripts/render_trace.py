"""
Render one episode from a JSONL shard into a report-friendly Markdown trace.

Examples
--------
Render the first episode in a shard:

    python -m coscope.scripts.render_trace \
        --jsonl coscope/data/processed_qwen_smoke/cot/musique/dev/s1_linear.jsonl

Render by original_id and write to a markdown file:

    python -m coscope.scripts.render_trace \
        --jsonl coscope/data/processed_qwen_smoke/cot/musique/dev/s1_linear.jsonl \
        --original-id 2hop__460946_294723 \
        --output trace_sample.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_episodes(path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            episodes.append(json.loads(line))
    return episodes


def _pick_episode(
    episodes: List[Dict[str, Any]],
    *,
    original_id: Optional[str],
    episode_id: Optional[str],
    index: int,
) -> Dict[str, Any]:
    if original_id:
        for episode in episodes:
            if episode.get("original_id") == original_id:
                return episode
        raise ValueError(f"original_id not found: {original_id}")
    if episode_id:
        for episode in episodes:
            if episode.get("episode_id") == episode_id:
                return episode
        raise ValueError(f"episode_id not found: {episode_id}")
    if not (0 <= index < len(episodes)):
        raise IndexError(f"index {index} out of range for {len(episodes)} episodes")
    return episodes[index]


def _artifact_rows(memory_entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in memory_entries:
        meta = entry.get("metadata") or {}
        slot = meta.get("slot")
        if not slot:
            continue
        rows.append(
            {
                "memory_id": entry.get("memory_id"),
                "slot": slot,
                "node_id": meta.get("source_node_id"),
                "hop_index": meta.get("hop_index"),
                "topo_index": meta.get("topo_index", 0),
                "parents": meta.get("parent_artifact_ids") or [],
                "content": entry.get("content", ""),
            }
        )
    rows.sort(key=lambda row: (int(row.get("topo_index") or 0), str(row.get("node_id") or "")))
    return rows


def _format_multiline(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "(empty)"
    return text


def render_episode_markdown(episode: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append(f"# Trace Report: {episode.get('episode_id', '')}")
    lines.append("")
    lines.append("## Episode")
    lines.append(f"- Dataset: `{episode.get('dataset', '')}`")
    lines.append(f"- Split: `{episode.get('split', '')}`")
    lines.append(f"- Original ID: `{episode.get('original_id', '')}`")
    lines.append(f"- Reasoning Path: `{episode.get('reasoning_path_type', '')}`")
    lines.append(f"- Graph Type: `{episode.get('graph_type', '')}`")
    lines.append(f"- Rho: `{episode.get('rho', '')}`")
    lines.append(f"- Subset: `{episode.get('rho_subset', '')}`")
    lines.append("")

    lines.append("## Question")
    lines.append(_format_multiline(str(episode.get("question", ""))))
    lines.append("")

    answer = str(episode.get("answer", "")).strip()
    if answer:
        lines.append("## Gold Answer")
        lines.append(answer)
        lines.append("")

    lines.append("## Retrieval Requests")
    for request in episode.get("retrieval_requests", []) or []:
        lines.append(
            f"- `{request.get('agent_id', '')}` "
            f"({request.get('role', '')}): {_format_multiline(str(request.get('query', '')))}"
        )
    lines.append("")

    lines.append("## Graph")
    for node in episode.get("got_graph", {}).get("nodes", []) or []:
        parents = ", ".join(node.get("parent_node_ids", []) or []) or "-"
        children = ", ".join(node.get("child_node_ids", []) or []) or "-"
        hop = node.get("hop_index")
        lines.append(
            f"- `{node.get('node_id', '')}` "
            f"[{node.get('node_type', '')}, hop={hop}] "
            f"parents: {parents}; children: {children}"
        )
    lines.append("")

    lines.append("## Reasoning Trace")
    for row in _artifact_rows(episode.get("memory_entries", []) or []):
        parents = ", ".join(row["parents"]) if row["parents"] else "-"
        lines.append(
            f"### {row['topo_index']}. `{row['node_id']}` -> `{row['slot']}`"
        )
        lines.append(f"- Memory ID: `{row['memory_id']}`")
        lines.append(f"- Hop: `{row['hop_index']}`")
        lines.append(f"- Parents: {parents}")
        lines.append("- Content:")
        lines.append("")
        lines.append("```text")
        lines.append(_format_multiline(str(row["content"])))
        lines.append("```")
        lines.append("")

    rollout = (episode.get("meta") or {}).get("rollout") or {}
    if rollout:
        lines.append("## Rollout Meta")
        for key, value in rollout.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one JSONL episode as Markdown trace")
    parser.add_argument("--jsonl", required=True, help="Path to a shard JSONL file")
    parser.add_argument("--index", type=int, default=0, help="0-based episode index when no id is provided")
    parser.add_argument("--original-id", default=None, help="Select one episode by original_id")
    parser.add_argument("--episode-id", default=None, help="Select one episode by episode_id")
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    episodes = _load_episodes(jsonl_path)
    episode = _pick_episode(
        episodes,
        original_id=args.original_id,
        episode_id=args.episode_id,
        index=args.index,
    )
    report = render_episode_markdown(episode)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
