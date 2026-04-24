"""
Standalone smoke test for `coscope.data`.

This script:
  1. Installs a stub for `coscope.retrieval.pipeline.RetrievalPipeline` so the
     broken top-level `coscope/__init__.py` import chain does not crash.
     (The retrieval pipeline is unrelated to data construction.)
  2. Builds a synthetic MuSiQue-like raw_item in memory.
  3. Runs EpisodeBuilder for each graph type and prints key invariants.
  4. Round-trips via Serializer and verifies the data survives intact.

Usage:
    python -m coscope.scripts.smoke_test
"""

from __future__ import annotations

import sys
import types


def _install_retrieval_pipeline_stub() -> None:
    """Create a minimal `coscope.retrieval.pipeline` module with RetrievalPipeline."""
    if "coscope.retrieval.pipeline" in sys.modules:
        return
    pkg_name = "coscope.retrieval.pipeline"
    stub = types.ModuleType(pkg_name)

    class RetrievalPipeline:  # noqa: D401
        """Placeholder RetrievalPipeline (smoke-test stub)."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("RetrievalPipeline stub invoked - runtime not wired")

    stub.RetrievalPipeline = RetrievalPipeline
    sys.modules[pkg_name] = stub


_install_retrieval_pipeline_stub()


# Deferred imports (after the stub).
from coscope import build_episode            # noqa: E402
from coscope.core.types import GraphType     # noqa: E402
from coscope.utils.output.serializer import Serializer  # noqa: E402


def _make_raw_item() -> dict:
    """Minimal MuSiQue-ish 2-hop raw_item."""
    return {
        "original_id": "smoke_0001",
        "question": "Who is the current mayor of the birthplace of Albert Einstein?",
        "answer": "Gunter Czisch",
        "hop_count": 2,
        "sub_questions": [
            "Where was Albert Einstein born?",
            "Who is the mayor of that city?",
        ],
        "solution_steps": [],
        "supporting_paragraphs": [
            {
                "paragraph_id": "para_000",
                "text": "Albert Einstein was born in Ulm, Kingdom of Wuerttemberg, German Empire.",
                "title": "Albert Einstein",
                "is_gold": True,
            },
            {
                "paragraph_id": "para_001",
                "text": "Gunter Czisch serves as the mayor of Ulm.",
                "title": "Gunter Czisch",
                "is_gold": True,
            },
        ],
        "distractor_paragraphs": [
            {
                "paragraph_id": "para_002",
                "text": "Berlin is the capital of Germany.",
                "title": "Berlin",
                "is_gold": False,
            },
        ],
        "supporting_facts": [
            {"paragraph_id": "para_000", "sentence_idx": 0},
            {"paragraph_id": "para_001", "sentence_idx": 0},
        ],
        "dataset_type": "qa",
        "math_category": None,
        "task_shared_items": [
            {
                "hop_index": 1,
                "text": "Albert Einstein was born in Ulm",
                "source_paragraph_id": "para_000",
            },
            {
                "hop_index": 2,
                "text": "Gunter Czisch serves as the mayor of Ulm",
                "source_paragraph_id": "para_001",
            },
        ],
        "qa_type": "bridge",
    }


def main() -> int:
    raw = _make_raw_item()
    graph_types = [
        GraphType.LINEAR,
        GraphType.FORK,
        GraphType.FORK_MERGE,
        GraphType.INDEPENDENT,
        GraphType.POLICY_ISOLATED,
    ]
    serializer = Serializer()
    failures = 0
    for gt in graph_types:
        print(f"--- {gt.value} ---")
        ep = build_episode(raw, dataset="musique", split="dev", target_graph_type=gt)
        if ep is None:
            print("  build returned None")
            failures += 1
            continue
        print(
            f"  episode_id={ep.episode_id}  agents={len(ep.agents)}  "
            f"memory={len(ep.memory_entries)}  rho={ep.rho}  "
            f"subset={ep.rho_subset.value}  s4_eligible={ep.s4_eligible}  "
            f"policy_conflict={ep.policy_conflict}"
        )
        # Roundtrip check.
        d = serializer.episode_to_dict(ep)
        ep2 = serializer.episode_from_dict(d)
        assert ep2.episode_id == ep.episode_id
        assert len(ep2.memory_entries) == len(ep.memory_entries)
        assert len(ep2.agents) == len(ep.agents)
        assert ep2.got_graph.graph_type == ep.got_graph.graph_type
        print("  roundtrip OK")
        # to_runtime_objects()
        agents, mem, reqs = ep.to_runtime_objects()
        assert len(agents) == len(ep.agents)
        assert len(mem) == len(ep.memory_entries)
        assert len(reqs) == len(ep.retrieval_requests)
        print(f"  to_runtime_objects OK: {len(agents)} agents / {len(mem)} entries / {len(reqs)} requests")
    if failures:
        print(f"FAIL: {failures} graph types failed")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
