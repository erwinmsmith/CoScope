"""Small structural comparison for GoT vs CoT.

Builds a handful of raw-item-shaped cases in memory and prints a compact table
showing how reasoning_path_type and graph_type affect:

- node / agent counts
- verifier presence
- rho and rho_subset
- restricted-layer availability for S4

Usage:
    python -m coscope.examples.compare_reasoning_structures
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from coscope.auth.access_controller import AccessController
from coscope.construction.episode_builder import EpisodeBuilder
from coscope.core.types import AgentRole, Episode, GraphType


def _make_cases() -> List[Dict[str, object]]:
    return [
        {
            "original_id": "cmp_2hop",
            "question": "Who is the mother of the husband of Queen Victoria?",
            "answer": "Princess Victoria of Saxe-Coburg-Saalfeld",
            "hop_count": 2,
            "sub_questions": [
                "Who was Queen Victoria married to?",
                "Who was the mother of Prince Albert?",
            ],
            "supporting_paragraphs": [
                {
                    "paragraph_id": "p1",
                    "title": "Queen Victoria",
                    "text": "Queen Victoria was married to Prince Albert.",
                    "hop_index": 1,
                },
                {
                    "paragraph_id": "p2",
                    "title": "Prince Albert",
                    "text": "Prince Albert was the son of Princess Victoria of Saxe-Coburg-Saalfeld.",
                    "hop_index": 2,
                },
            ],
            "distractor_paragraphs": [
                {
                    "paragraph_id": "d1",
                    "title": "Other",
                    "text": "A distractor paragraph.",
                }
            ],
            "task_shared_items": [
                {
                    "hop_index": 1,
                    "text": "Queen Victoria was married to Prince Albert.",
                    "source_paragraph_id": "p1",
                },
                {
                    "hop_index": 2,
                    "text": "Prince Albert was the son of Princess Victoria of Saxe-Coburg-Saalfeld.",
                    "source_paragraph_id": "p2",
                },
            ],
            "dataset_type": "qa",
        },
        {
            "original_id": "cmp_3hop",
            "question": "Which city is the birthplace of the mother of the inventor of relativity?",
            "answer": "Canstatt",
            "hop_count": 3,
            "sub_questions": [
                "Who invented relativity?",
                "Who was the mother of Albert Einstein?",
                "Where was Pauline Koch born?",
            ],
            "supporting_paragraphs": [
                {
                    "paragraph_id": "p11",
                    "title": "Relativity",
                    "text": "Albert Einstein developed the theory of relativity.",
                    "hop_index": 1,
                },
                {
                    "paragraph_id": "p12",
                    "title": "Pauline Koch",
                    "text": "Pauline Koch was the mother of Albert Einstein.",
                    "hop_index": 2,
                },
                {
                    "paragraph_id": "p13",
                    "title": "Canstatt",
                    "text": "Pauline Koch was born in Canstatt.",
                    "hop_index": 3,
                },
            ],
            "distractor_paragraphs": [
                {
                    "paragraph_id": "d11",
                    "title": "Distractor",
                    "text": "Another unrelated city paragraph.",
                }
            ],
            "task_shared_items": [
                {
                    "hop_index": 1,
                    "text": "Albert Einstein developed the theory of relativity.",
                    "source_paragraph_id": "p11",
                },
                {
                    "hop_index": 2,
                    "text": "Pauline Koch was the mother of Albert Einstein.",
                    "source_paragraph_id": "p12",
                },
                {
                    "hop_index": 3,
                    "text": "Pauline Koch was born in Canstatt.",
                    "source_paragraph_id": "p13",
                },
            ],
            "dataset_type": "qa",
        },
    ]


def _summarize_episode(ep: Episode) -> Dict[str, object]:
    controller = AccessController()
    restricted_count = len(
        [
            m
            for m in ep.memory_entries
            if (m.metadata or {}).get("scope_layer") == "restricted"
        ]
    )
    verifier = next(
        (a for a in ep.agents if a.config.role == AgentRole.VERIFIER),
        None,
    )
    verifier_visible = 0
    if verifier is not None:
        verifier_visible = len(
            [
                m
                for m in controller.filter_accessible_entries(
                    verifier, ep.memory_entries, ep.got_graph
                )
                if (m.metadata or {}).get("scope_layer") == "restricted"
            ]
        )
    return {
        "episode_id": ep.episode_id,
        "reasoning": ep.reasoning_path_type.value,
        "graph": ep.graph_type.value,
        "nodes": len(ep.got_graph.nodes),
        "solvers": len(ep.got_graph.solver_nodes()),
        "agents": len(ep.agents),
        "verifier": "Y" if verifier is not None else "N",
        "rho": f"{ep.rho:.4f}",
        "subset": ep.rho_subset.value,
        "s4": "Y" if ep.s4_eligible else "N",
        "restricted": restricted_count,
        "verifier_visible": verifier_visible,
    }


def _format_table(rows: Sequence[Dict[str, object]]) -> str:
    columns: List[Tuple[str, str]] = [
        ("episode_id", "Episode"),
        ("reasoning", "Reasoning"),
        ("graph", "Graph"),
        ("nodes", "Nodes"),
        ("solvers", "Solvers"),
        ("agents", "Agents"),
        ("verifier", "Verifier"),
        ("rho", "Rho"),
        ("subset", "Subset"),
        ("s4", "S4"),
        ("restricted", "Restricted"),
        ("verifier_visible", "VerifierSees"),
    ]
    widths = {
        key: max(len(label), *(len(str(row[key])) for row in rows))
        for key, label in columns
    }
    header = " | ".join(label.ljust(widths[key]) for key, label in columns)
    sep = "-+-".join("-" * widths[key] for key, _ in columns)
    lines = [header, sep]
    for row in rows:
        lines.append(
            " | ".join(str(row[key]).ljust(widths[key]) for key, _ in columns)
        )
    return "\n".join(lines)


def main() -> None:
    cases = _make_cases()
    rows: List[Dict[str, object]] = []

    got_graph_types = [
        GraphType.LINEAR,
        GraphType.FORK,
        GraphType.FORK_MERGE,
        GraphType.INDEPENDENT,
        GraphType.POLICY_ISOLATED,
    ]
    cot_graph_types = [
        GraphType.LINEAR,
        GraphType.POLICY_ISOLATED,
    ]

    for raw in cases:
        for graph_type in got_graph_types:
            ep = EpisodeBuilder(reasoning_path_type="got").build_episode(
                raw, dataset="musique", split="test", target_graph_type=graph_type, seed=42
            )
            if ep is not None:
                rows.append(_summarize_episode(ep))

        for graph_type in cot_graph_types:
            ep = EpisodeBuilder(reasoning_path_type="cot").build_episode(
                raw, dataset="musique", split="test", target_graph_type=graph_type, seed=42
            )
            if ep is not None:
                rows.append(_summarize_episode(ep))

    print(_format_table(rows))


if __name__ == "__main__":
    main()
