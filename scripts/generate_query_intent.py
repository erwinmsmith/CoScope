"""
Generate LLM-rewritten query_intent per retrieval_request (Step 2 of the
CoScope pipeline) and patch it back into JSONL episode shards so the A8
variant can replay retrieval with the rewritten query.

Usage:
    python -m scripts.generate_query_intent \
        --shards 'tests/tmp/processed/got/musique/test/s1*.jsonl' \
        --dataset musique --split test \
        --max-episodes 200 \
        --llm dashscope --model qwen-plus \
        --output-dir tests/tmp/processed/got/musique/test_a8

The script only issues LLM calls for the query_intent artifact (one call per
agent). Scratch / conclusion / plan artifacts are NOT generated here because
Mode B evaluation does not need them. Calls are issued sequentially with
retries; token cost for Qwen-plus is roughly 0.002-0.004 RMB per call.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types import (
    AgentRole,
    Episode,
    GoTGraph,
    ReasoningPathType,
    RetrievalRequest,
)
from rollout import prompt_assembly
from rollout import fallback_synth
from dataio.loaders import get_loader
from dataio.serializer import Serializer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------


def _make_llm(kind: str, model: Optional[str]) -> Any:
    """Instantiate the selected LLM client. Supported: dashscope, template."""
    kind = (kind or "template").lower()
    if kind == "dashscope":
        from llm.dashscope import DashScopeClient

        return DashScopeClient(model=model or "qwen-plus")
    if kind == "template":
        from llm.template import TemplateLLMClient

        return TemplateLLMClient()
    raise ValueError(f"Unsupported --llm kind: {kind}")


# ---------------------------------------------------------------------------
# Raw item lookup
# ---------------------------------------------------------------------------


def _load_raw_items(dataset: str, split: str, data_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load raw dataset items indexed by original_id (as string)."""
    loader = get_loader(dataset, data_dir=data_dir, split=split)
    items: Dict[str, Dict[str, Any]] = {}
    for item in loader.load():
        oid = str(item.get("original_id") or item.get("id") or "")
        if oid:
            items[oid] = item
    return items


def _original_id_from_episode(episode: Episode) -> Optional[str]:
    """Return the raw-item id stored directly on the episode."""
    return getattr(episode, "original_id", None) or None


# ---------------------------------------------------------------------------
# Core: per-agent query_intent generation
# ---------------------------------------------------------------------------


@dataclass
class GenStats:
    total_requests: int = 0
    generated: int = 0
    template_fallback: int = 0
    skipped_no_raw: int = 0
    elapsed_s: float = 0.0


def _call_llm_with_retry(llm: Any, prompt: str, max_retries: int = 2) -> Optional[str]:
    """Call the LLM returning stripped text, or None on persistent failure."""
    for attempt in range(max_retries + 1):
        try:
            resp = llm.generate(prompt, temperature=0.3, top_p=0.9, seed=42)
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(min(2 ** attempt, 6))
    return None


def _node_of_request(request: RetrievalRequest, got_graph: GoTGraph, episode: Episode):
    """Find the GoTNode associated with this retrieval request.

    Priority order:
      1. request.metadata['node_id']
      2. agent.config.metadata['node_id'] (via episode.agents lookup)
      3. Strip the ``<episode_id>_`` prefix from request.agent_id
    """
    node_id = (request.metadata or {}).get("node_id")
    if not node_id:
        for agent in episode.agents:
            if agent.agent_id == request.agent_id:
                node_id = (agent.config.metadata or {}).get("node_id")
                break
    if not node_id:
        prefix = f"{episode.episode_id}_"
        if request.agent_id.startswith(prefix):
            node_id = request.agent_id[len(prefix):]
    if not node_id:
        return None
    for node in got_graph.nodes:
        if node.node_id == node_id:
            return node
    return None


def _build_prior_conclusions(
    request: RetrievalRequest, got_graph: GoTGraph, episode: Episode
) -> List[str]:
    """Gather ancestor-solver conclusions from episode.memory_entries.

    We use task_shared conclusion content authored by each ancestor solver,
    matched via memory.metadata.source_node_id == ancestor.node_id.
    """
    node = _node_of_request(request, got_graph, episode)
    if node is None:
        return []
    ancestors = set(got_graph.get_all_ancestors(node.node_id))
    if not ancestors:
        return []

    priors: List[str] = []
    for mem in episode.memory_entries:
        meta = mem.metadata or {}
        slot = meta.get("slot", "")
        src = meta.get("source_node_id", "")
        if src in ancestors and slot in {"conclusion", "CONCLUSION"}:
            priors.append(mem.content)
    return priors


def _generate_intent_for_request(
    request: RetrievalRequest,
    raw_item: Dict[str, Any],
    got_graph: GoTGraph,
    episode: Episode,
    llm: Any,
    stats: GenStats,
) -> str:
    """Return query_intent text for one request. Falls back to template on LLM error."""
    role = request.role.value if hasattr(request.role, "value") else str(request.role)
    node = _node_of_request(request, got_graph, episode)

    if node is None:
        # Without a node we cannot reconstruct the Step-2 prompt faithfully,
        # so fall back to the original query text as the query_intent.
        stats.template_fallback += 1
        return request.query or ""

    prior = _build_prior_conclusions(request, got_graph, episode)

    prompt = prompt_assembly.build_query_intent_prompt(
        role=role,
        raw_item=raw_item,
        node=node,
        prior_conclusions=prior,
        reasoning_path_type=ReasoningPathType.GOT,
    )
    text = _call_llm_with_retry(llm, prompt)
    if text:
        stats.generated += 1
        return text

    # Template fallback mirrors the rollout engine's behaviour.
    stats.template_fallback += 1
    if role == "planner":
        return fallback_synth.synth_planner_query_intent(raw_item)
    if role == "solver":
        return fallback_synth.synth_solver_query_intent(raw_item, node)
    if role == "verifier":
        return fallback_synth.synth_verifier_query_intent(raw_item)
    return request.query or ""


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------


def _patch_episode(
    episode: Episode,
    raw_items: Dict[str, Dict[str, Any]],
    llm: Any,
    stats: GenStats,
) -> bool:
    """Mutate episode in-place so each request carries metadata['query_intent']."""
    oid = _original_id_from_episode(episode)
    raw_item = raw_items.get(oid) if oid else None
    if raw_item is None:
        stats.skipped_no_raw += len(episode.retrieval_requests or [])
        return False

    for request in episode.retrieval_requests:
        stats.total_requests += 1
        intent = _generate_intent_for_request(
            request=request,
            raw_item=raw_item,
            got_graph=episode.got_graph,
            episode=episode,
            llm=llm,
            stats=stats,
        )
        md = dict(request.metadata or {})
        md["query_intent"] = intent
        request.metadata = md
    return True


def _process_shards(
    shards: List[Path],
    raw_items: Dict[str, Dict[str, Any]],
    llm: Any,
    output_dir: Path,
    max_episodes: Optional[int],
) -> GenStats:
    stats = GenStats()
    serializer = Serializer()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    episode_count = 0
    for shard_path in shards:
        out_path = output_dir / shard_path.name
        episodes = serializer.read_jsonl(shard_path)
        with out_path.open("w", encoding="utf-8") as out_f:
            for episode in episodes:
                if max_episodes is not None and episode_count >= max_episodes:
                    out_f.write(
                        json.dumps(serializer.episode_to_dict(episode), ensure_ascii=False)
                        + "\n"
                    )
                    continue
                _patch_episode(episode, raw_items, llm, stats)
                out_f.write(
                    json.dumps(serializer.episode_to_dict(episode), ensure_ascii=False)
                    + "\n"
                )
                episode_count += 1
                if episode_count % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = episode_count / max(elapsed, 1e-3)
                    print(
                        f"[generate_query_intent] episodes={episode_count} "
                        f"requests={stats.total_requests} "
                        f"llm_ok={stats.generated} fb={stats.template_fallback} "
                        f"rate={rate:.2f} ep/s",
                        flush=True,
                    )
        print(f"[generate_query_intent] wrote {out_path}")

    stats.elapsed_s = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--data-dir",
        default=None,
        help="Raw dataset dir, defaults to data/raw/<dataset>",
    )
    p.add_argument("--llm", choices=["dashscope", "template"], default="template")
    p.add_argument("--model", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit total episodes patched across all shards (for a small-scale run).",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    shard_paths: List[Path] = []
    for pattern in args.shards:
        matched = sorted(Path().glob(pattern))
        if not matched and Path(pattern).exists():
            matched = [Path(pattern)]
        shard_paths.extend(matched)
    if not shard_paths:
        print("No shards matched; aborting.", file=sys.stderr)
        sys.exit(2)

    data_dir = args.data_dir or f"data/raw/{args.dataset}"
    raw_items = _load_raw_items(args.dataset, args.split, data_dir)
    print(f"[generate_query_intent] loaded {len(raw_items)} raw items from {data_dir}/{args.split}")

    llm = _make_llm(args.llm, args.model)
    print(f"[generate_query_intent] llm={getattr(llm, 'name', args.llm)}")

    stats = _process_shards(
        shards=shard_paths,
        raw_items=raw_items,
        llm=llm,
        output_dir=Path(args.output_dir),
        max_episodes=args.max_episodes,
    )
    print(
        f"\n[summary] episodes_patched_requests={stats.total_requests} "
        f"llm_ok={stats.generated} template_fb={stats.template_fallback} "
        f"skipped={stats.skipped_no_raw} elapsed={stats.elapsed_s:.1f}s"
    )


if __name__ == "__main__":
    main()
