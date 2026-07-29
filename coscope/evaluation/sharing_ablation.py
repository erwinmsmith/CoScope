"""Paired CoScope, full-sharing, and no-sharing MAS ablation."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from enum import Enum
from statistics import mean
from typing import Any, cast

from coscope import AgentClass, CoScopeRuntime, MemoryEntry
from coscope.config import CoScopeSettings
from coscope.context import ContextPacket
from coscope.core import Artifact, ArtifactState, ArtifactType
from coscope.evaluation.benchmark_suite import (
    extract_benchmark_answer,
    score_benchmark_answer,
)
from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.code_benchmark import (
    CodeEvaluator,
    apply_mbpp_plus_scores,
)
from coscope.evaluation.context_metrics import (
    context_pollution,
    duplicate_evidence_rate,
)
from coscope.evaluation.mas_experiment import (
    AGENT_CLASSES,
    AGENTS,
    ROLE_KNOWLEDGE,
)
from coscope.evaluation.math_grader import extract_math_answer
from coscope.evaluation.retrieval_metrics import mrr_at_k, recall_at_k
from coscope.reasoning import ReasoningConfig
from coscope.retrieval import RetrievalRequest
from coscope.scope import Permission, ScopeDescriptor, Visibility


class SharingArm(str, Enum):
    COSCOPE = "coscope"
    FULL_SHARING = "full_sharing"
    NO_SHARING = "no_sharing"


def reaggregate_sharing_report(
    report: dict[str, Any],
    *,
    bootstrap_samples: int = 1_000,
) -> dict[str, Any]:
    """Recompute overall metrics after merging disjoint benchmark shards."""
    combined = deepcopy(report)
    tasks_by_arm: dict[SharingArm, list[dict[str, Any]]] = {
        arm: [] for arm in SharingArm
    }
    output_caps = (
        combined.get("models", {}).get("max_output_tokens_by_benchmark", {})
    )
    for benchmark, benchmark_report in combined["benchmarks"].items():
        for arm in SharingArm:
            tasks = benchmark_report[arm.value]["tasks"]
            for task in tasks:
                _ensure_per_agent_llm_usage(task)
                raw_output_cap = output_caps.get(benchmark)
                _ensure_generation_budget(
                    task,
                    max_output_tokens=(
                        int(raw_output_cap)
                        if raw_output_cap is not None
                        else None
                    ),
                )
            tasks_by_arm[arm].extend(tasks)
            benchmark_report[arm.value]["aggregate"] = _aggregate(
                tasks
            )
    combined["overall"] = {
        arm.value: _aggregate(tasks)
        for arm, tasks in tasks_by_arm.items()
    }
    combined["comparisons"] = _comparisons(
        tasks_by_arm,
        combined["overall"],
        bootstrap_samples=bootstrap_samples,
    )
    return combined


def run_sharing_ablation(
    examples_by_benchmark: dict[str, list[BenchmarkExample]],
    base_settings: CoScopeSettings,
    *,
    threshold: float = 0.75,
    max_output_tokens: int | None = None,
    max_output_tokens_by_benchmark: dict[str, int | None] | None = None,
    bootstrap_samples: int = 1_000,
    code_evaluator: CodeEvaluator | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Run all arms on identical examples in paired order."""
    settings = replace(
        base_settings,
        retrieval=replace(
            base_settings.retrieval,
            medoid_threshold=threshold,
            minimum_pairwise_similarity=threshold,
        ),
    )
    output_caps: dict[str, int | None] = {}
    for benchmark in examples_by_benchmark:
        cap = (max_output_tokens_by_benchmark or {}).get(
            benchmark,
            max_output_tokens,
        )
        output_caps[benchmark] = (
            base_settings.llm.max_tokens if cap is None else cap
        )
    if any(cap is not None and cap <= 0 for cap in output_caps.values()):
        raise ValueError("max output token caps must be positive when set")
    reports: dict[str, dict[str, Any]] = {}
    all_tasks: dict[SharingArm, list[dict[str, Any]]] = {
        arm: [] for arm in SharingArm
    }
    for benchmark, examples in examples_by_benchmark.items():
        benchmark_settings = replace(
            settings,
            llm=replace(
                settings.llm,
                max_tokens=output_caps[benchmark],
            ),
        )
        arm_tasks: dict[SharingArm, list[dict[str, Any]]] = {
            arm: [] for arm in SharingArm
        }
        for example in examples:
            for arm in SharingArm:
                task = _run_arm_task(example, benchmark_settings, arm)
                arm_tasks[arm].append(task)
                all_tasks[arm].append(task)
                if progress is not None:
                    progress(
                        {
                            "benchmark": benchmark,
                            "example_id": example.example_id,
                            "arm": arm.value,
                            "answer_f1": task["scores"]["f1"],
                            "pollution_rate": task["context"]["pollution_rate"],
                            "end_to_end_seconds": task["latency"][
                                "end_to_end_seconds"
                            ],
                        }
                    )
        if benchmark == "mbpp_plus":
            if code_evaluator is None:
                raise ValueError(
                    "MBPP-Plus requires the Docker EvalPlus code evaluator"
                )
            for arm in SharingArm:
                apply_mbpp_plus_scores(
                    arm_tasks[arm],
                    examples,
                    code_evaluator,
                    label=f"sharing_ablation_mbpp_plus_{arm.value}",
                )
        reports[benchmark] = {
            arm.value: {
                "aggregate": _aggregate(tasks),
                "tasks": tasks,
            }
            for arm, tasks in arm_tasks.items()
        }

    overall = {
        arm.value: _aggregate(tasks) for arm, tasks in all_tasks.items()
    }
    return {
        "models": {
            "llm": settings.llm.model,
            "llm_temperature": settings.llm.temperature,
            "embedding": settings.embedding.model,
            "embedding_dimension": settings.embedding.dimension,
            "max_output_tokens_by_benchmark": output_caps,
        },
        "design": {
            "paired_examples": True,
            "agent_count": len(AGENTS),
            "retrieval_threshold": threshold,
            "arms": {
                SharingArm.COSCOPE.value: (
                    "shared task evidence, role-private knowledge, private raw "
                    "working state, explicit downstream summaries"
                ),
                SharingArm.FULL_SHARING.value: (
                    "all role knowledge and raw upstream working text are team shared"
                ),
                SharingArm.NO_SHARING.value: (
                    "task evidence is duplicated into agent-private corpora and "
                    "no runtime state crosses agents"
                ),
            },
            "gold_available_to_runtime": False,
            "final_answer_rule": "verifier output after arm-specific information flow",
        },
        "benchmarks": reports,
        "overall": overall,
        "comparisons": _comparisons(
            all_tasks,
            overall,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def _run_arm_task(
    example: BenchmarkExample,
    settings: CoScopeSettings,
    arm: SharingArm,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = CoScopeRuntime.from_settings(settings)
    run_id = f"ablation_{arm.value}_{example.benchmark}_{example.example_id}"
    runtime.create_run(example.benchmark, run_id=run_id)
    nodes = {}
    for agent_id, mode, _ in AGENTS:
        agent_class = _agent_class_for_arm(agent_id, arm)
        runtime.register_agent(agent_class.instantiate(agent_id))
        _, node = runtime.start_reasoning(agent_id, ReasoningConfig(mode))
        nodes[agent_id] = node

    gold_by_agent = _ingest_arm_memories(runtime, run_id, example, arm)
    requests = _make_requests(runtime, run_id, nodes, example, arm)
    retrieval_started = time.perf_counter()
    results = runtime.retrieve_batch(requests)
    retrieval_seconds = time.perf_counter() - retrieval_started
    executor = runtime.live_executor(
        system_instructions=_answer_instructions(example.benchmark)
    )

    outputs = {}
    packets: list[ContextPacket] = []
    data_flow = {}
    generation_started = time.perf_counter()
    for (agent_id, _, _), request in zip(AGENTS, requests, strict=True):
        runtime.refresh_context_view(request)
        packet = runtime.assemble_context(
            request,
            results[request.request_id],
        )
        packets.append(packet)
        output = executor.invoke(
            runtime.topology.agents[agent_id],
            nodes[agent_id],
            packet,
        )
        outputs[agent_id] = output
        candidate = extract_benchmark_answer(example.benchmark, output.text)
        data_flow[agent_id] = _write_arm_thinking(
            runtime,
            arm,
            agent_id=agent_id,
            reasoning_node_id=nodes[agent_id].node_id,
            raw_output=output.text,
            candidate=candidate,
        )
    generation_seconds = time.perf_counter() - generation_started

    answers = {
        agent_id: extract_benchmark_answer(example.benchmark, output.text)
        for agent_id, output in outputs.items()
    }
    prediction = answers["verifier"]
    scores = score_benchmark_answer(example, prediction)
    gold_by_request = {
        request.request_id: gold_by_agent[request.agent_id]
        for request in requests
    }
    pollution_reports = {
        agent_id: context_pollution(packet, agent_id=agent_id)
        for (agent_id, _, _), packet in zip(AGENTS, packets, strict=True)
    }
    unauthorized = sum(
        entry.scope.scope_id not in request.effective_view.scope_ids
        for request, packet in zip(requests, packets, strict=True)
        for entry in packet.all_memories
    )
    usage = runtime.usage.summary()
    per_agent_usage = {
        agent_id: cast(dict[str, Any], output.metadata.get("usage", {}))
        for agent_id, output in outputs.items()
    }
    store_queries = (
        runtime.retrieval.stats.shared_store_queries
        + runtime.retrieval.stats.private_store_queries
    )
    return {
        "benchmark": example.benchmark,
        "example_id": example.example_id,
        "arm": arm.value,
        "prediction": prediction,
        "reference": (
            extract_math_answer(example.reference)
            if example.benchmark == "math"
            else (
                "[hidden EvalPlus base and plus tests]"
                if example.benchmark == "mbpp_plus"
                else example.reference
            )
        ),
        "scores": scores,
        "task_success": bool(scores["success"]),
        "agent_answers": answers,
        "per_agent_llm_usage": per_agent_usage,
        "generation_budget": _generation_budget(
            per_agent_usage,
            max_output_tokens=settings.llm.max_tokens,
        ),
        "provider_usage": usage,
        "retrieval": {
            "groups": runtime.retrieval.stats.groups,
            "shared_store_queries": runtime.retrieval.stats.shared_store_queries,
            "private_store_queries": runtime.retrieval.stats.private_store_queries,
            "store_queries": store_queries,
            "fallback_triggers": runtime.retrieval.stats.fallback_triggers,
            "recall_at_10": recall_at_k(results, gold_by_request, k=10),
            "mrr_at_10": mrr_at_k(results, gold_by_request, k=10),
            "latency_seconds": retrieval_seconds,
        },
        "context": {
            "selected_items": sum(len(packet.all_memories) for packet in packets),
            "mean_selected_items": mean(
                len(packet.all_memories) for packet in packets
            ),
            "mean_duplicate_evidence_rate": mean(
                duplicate_evidence_rate(packet) for packet in packets
            ),
            "polluted_items": sum(
                report.polluted_items for report in pollution_reports.values()
            ),
            "pollution_rate": (
                sum(
                    report.polluted_items
                    for report in pollution_reports.values()
                )
                / max(1, sum(len(packet.all_memories) for packet in packets))
            ),
            "private_thinking_exposure": sum(
                report.private_thinking_exposure
                for report in pollution_reports.values()
            ),
            "cross_role_knowledge_exposure": sum(
                report.cross_role_knowledge_exposure
                for report in pollution_reports.values()
            ),
            "per_agent": {
                agent_id: {
                    "selected_items": report.selected_items,
                    "polluted_items": report.polluted_items,
                    "pollution_rate": report.pollution_rate,
                    "source_ids": [
                        entry.source_id for entry in packet.all_memories
                    ],
                }
                for (agent_id, _, _), packet, report in zip(
                    AGENTS,
                    packets,
                    pollution_reports.values(),
                    strict=True,
                )
            },
        },
        "safety": {"unauthorized_context_exposure": unauthorized},
        "data_flow": data_flow,
        "latency": {
            "generation_seconds": generation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        },
    }


def _agent_class_for_arm(agent_id: str, arm: SharingArm) -> AgentClass:
    base = AGENT_CLASSES[agent_id]
    if arm == SharingArm.COSCOPE:
        return base
    return AgentClass(
        f"{base.class_id}_{arm.value}",
        base.role,
        base.knowledge_permissions,
        replace(base.context_policy, channel_budgets={}),
        base.capabilities,
        base.tool_permissions,
    )


def _ingest_arm_memories(
    runtime: CoScopeRuntime,
    run_id: str,
    example: BenchmarkExample,
    arm: SharingArm,
) -> dict[str, set[str]]:
    role_domains = {
        "planner": "planning_knowledge",
        "solver": "solving_knowledge",
        "verifier": "verification_knowledge",
    }
    role_scopes = {
        agent_id: ScopeDescriptor(
            frozenset({domain}),
            f"{run_id}/knowledge/{agent_id}",
            Visibility.RESTRICTED,
            permissions={
                f"role:{agent_id}": frozenset({Permission.READ})
            },
            memory_types=frozenset(
                {"benchmark_evidence", "role_knowledge"}
            ),
            trust_level="benchmark_specialist",
        )
        for agent_id, domain in role_domains.items()
    }
    common_scope = ScopeDescriptor(
        frozenset({"benchmark_common"}),
        f"{run_id}/task",
        Visibility.TEAM_SHARED,
        memory_types=frozenset(
            {"benchmark_evidence", "role_knowledge"}
        ),
        trust_level="benchmark_common",
    )
    raw: list[
        tuple[str, str, ScopeDescriptor, str, dict[str, object]]
    ] = []
    source_map: dict[str, dict[str, str]] = {
        agent_id: {} for agent_id, _, _ in AGENTS
    }
    task_items = [
        ("question", example.question, "user_input", False),
        *[
            (
                context.source_id,
                context.content,
                "benchmark_context",
                True,
            )
            for context in example.context
        ],
    ]
    if arm == SharingArm.NO_SHARING:
        for agent_id, _, _ in AGENTS:
            for source_id, content, source_type, retrieval_only in task_items:
                raw.append(
                    (
                        f"{source_id}@{agent_id}",
                        content,
                        role_scopes[agent_id],
                        source_type,
                        {
                            "benchmark": example.benchmark,
                            "retrieval_only": retrieval_only,
                            "original_source_id": source_id,
                            "intended_agents": [agent_id],
                        },
                    )
                )
            raw.append(
                (
                    f"{agent_id}_role_knowledge",
                    ROLE_KNOWLEDGE[agent_id],
                    role_scopes[agent_id],
                    "role_knowledge",
                    {
                        "knowledge_class": agent_id,
                        "intended_agents": [agent_id],
                    },
                )
            )
    else:
        for source_id, content, source_type, retrieval_only in task_items:
            raw.append(
                (
                    source_id,
                    content,
                    common_scope,
                    source_type,
                    {
                        "benchmark": example.benchmark,
                        "retrieval_only": retrieval_only,
                        "intended_agents": [
                            agent_id for agent_id, _, _ in AGENTS
                        ],
                    },
                )
            )
        for agent_id, _, _ in AGENTS:
            raw.append(
                (
                    f"{agent_id}_role_knowledge",
                    ROLE_KNOWLEDGE[agent_id],
                    (
                        common_scope
                        if arm == SharingArm.FULL_SHARING
                        else role_scopes[agent_id]
                    ),
                    "role_knowledge",
                    {
                        "knowledge_class": agent_id,
                        "intended_agents": [agent_id],
                    },
                )
            )

    batches = (
        [
            [
                row
                for row in raw
                if row[4].get("intended_agents") == [agent_id]
            ]
            for agent_id, _, _ in AGENTS
        ]
        if arm == SharingArm.NO_SHARING
        else [raw]
    )
    for batch in batches:
        vectors = runtime.embedder.embed_many(
            [content for _, content, _, _, _ in batch]
        )
        for (source_id, content, scope, source_type, metadata), vector in zip(
            batch,
            vectors,
            strict=True,
        ):
            entry = runtime.ingest_memory(
                MemoryEntry(
                    content,
                    scope,
                    source_type,
                    source_id,
                    vector=vector,
                    metadata=metadata,
                )
            )
            original = str(metadata.get("original_source_id", source_id))
            if arm == SharingArm.NO_SHARING:
                intended = metadata.get("intended_agents", [])
                if isinstance(intended, list) and intended:
                    source_map[str(intended[0])][original] = entry.memory_id
            else:
                for agent_id, _, _ in AGENTS:
                    source_map[agent_id][original] = entry.memory_id

    gold: dict[str, set[str]] = {}
    for agent_id, _, _ in AGENTS:
        ids = {
            source_map[agent_id][source_id]
            for source_id in example.supporting_source_ids
            if source_id in source_map[agent_id]
        }
        if not ids:
            ids = {source_map[agent_id]["question"]}
        gold[agent_id] = ids
    return gold


def _make_requests(
    runtime: CoScopeRuntime,
    run_id: str,
    nodes: dict[str, Any],
    example: BenchmarkExample,
    arm: SharingArm,
) -> list[RetrievalRequest]:
    objectives = {
        "planner": "identify evidence and a solution plan",
        "solver": "derive the exact final answer",
        "verifier": "independently verify evidence and candidate answers",
    }
    return [
        runtime.create_request(
            run_id=run_id,
            agent_id=agent_id,
            reasoning_node_id=nodes[agent_id].node_id,
            full_query=f"{example.question}\nRole objective: {objectives[agent_id]}",
            public_intent=f"{example.question} {objectives[agent_id]}",
            private_intent=(
                f"{agent_id} private task evidence"
                if arm == SharingArm.NO_SHARING
                else None
            ),
            full_query_is_public=True,
            retrieval_budget=10,
        )
        for agent_id, _, _ in AGENTS
    ]


def _write_arm_thinking(
    runtime: CoScopeRuntime,
    arm: SharingArm,
    *,
    agent_id: str,
    reasoning_node_id: str,
    raw_output: str,
    candidate: str,
) -> dict[str, object]:
    if arm == SharingArm.NO_SHARING:
        private_entry, _ = runtime.record_thinking(
            actor_id=agent_id,
            reasoning_node_id=reasoning_node_id,
            private_content=raw_output,
        )
        return {
            "private_memory_id": private_entry.memory_id,
            "published_memory_id": None,
            "published_to": [],
        }
    if arm == SharingArm.COSCOPE:
        recipients = (
            None
            if agent_id in {"planner", "verifier"}
            else frozenset({"verifier"})
        )
        summary = _extract_public_summary(
            raw_output,
            fallback_answer=candidate,
            agent_id=agent_id,
        )
        private_entry, published_entry = runtime.record_thinking(
            actor_id=agent_id,
            reasoning_node_id=reasoning_node_id,
            private_content=raw_output,
            public_summary=summary,
            recipients=recipients,
        )
        return {
            "private_memory_id": private_entry.memory_id,
            "published_memory_id": (
                published_entry.memory_id if published_entry else None
            ),
            "published_to": (
                "team" if recipients is None else sorted(recipients)
            ),
        }

    actor = runtime.topology.agents[agent_id]
    node = runtime.nodes[(agent_id, reasoning_node_id)]
    scope = ScopeDescriptor(
        frozenset({"runtime_thinking"}),
        f"{node.runtime_region}/full_shared",
        Visibility.TEAM_SHARED,
        owner_id=agent_id,
        permissions={
            agent_id: frozenset(
                {Permission.READ, Permission.WRITE, Permission.SHARE}
            )
        },
        memory_types=frozenset({"thinking"}),
        trust_level="unfiltered_working_text",
    )
    entry = runtime.write_artifact(
        agent_id,
        Artifact(
            raw_output,
            ArtifactType.REASONING_SUMMARY,
            "full_shared_thinking",
            reasoning_node_id,
            actor.agent_id,
            state=ArtifactState.COMMITTED,
            metadata={
                "context_flow": "full_shared_raw",
                "privacy": "private",
                "owner_agent_id": agent_id,
                "intended_agents": [agent_id],
            },
        ),
        scope,
        runtime_region=node.runtime_region,
        searchable=False,
    )
    return {
        "private_memory_id": None,
        "published_memory_id": entry.memory_id,
        "published_to": "team_raw",
    }


def _answer_instructions(benchmark: str) -> tuple[str, ...]:
    common = (
        "Answer using only the authorized context.",
        (
            "Before the final answer, include exactly one line formatted as "
            "PUBLIC_SUMMARY: <a concise share-safe result>."
        ),
    )
    if benchmark == "gsm8k":
        return (*common, "End with exactly #### <number> and no text after it.")
    if benchmark == "mbpp_plus":
        return (
            *common,
            "Return a self-contained Python solution defining the requested function.",
            "Do not include tests, shell commands, file access, or network access.",
            "End with FINAL_CODE: followed by one fenced Python code block.",
            "Do not place text after the closing code fence.",
        )
    return (
        *common,
        "End with exactly FINAL_ANSWER: <answer> and no text after it.",
        (
            "For MATH, put only the final mathematical expression after the "
            "marker; for AIME, put only an integer from 000 through 999."
        ),
    )


def _extract_public_summary(
    output: str,
    *,
    fallback_answer: str,
    agent_id: str,
) -> str:
    match = re.search(r"(?im)^PUBLIC_SUMMARY:\s*(.+?)\s*$", output)
    if match:
        return match.group(1).strip()
    return f"{agent_id} candidate answer: {fallback_answer}"


def _aggregate(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    llm = _sum_usage(tasks, "llm")
    embedding = _sum_usage(tasks, "embedding")
    total_seconds = sum(
        float(task["latency"]["end_to_end_seconds"]) for task in tasks
    )
    total_wall_clock_seconds = sum(
        float(
            task["latency"].get(
                "wall_clock_seconds",
                task["latency"]["end_to_end_seconds"],
            )
        )
        for task in tasks
    )
    total_embedding_seconds = sum(
        float(task["latency"].get("embedding_seconds", 0.0))
        for task in tasks
    )
    total_f1 = sum(float(task["scores"]["f1"]) for task in tasks)
    total_provider_tokens = llm["total_tokens"] + embedding["total_tokens"]
    selected = sum(int(task["context"]["selected_items"]) for task in tasks)
    polluted = sum(int(task["context"]["polluted_items"]) for task in tasks)
    aggregate = {
        "tasks": len(tasks),
        "answer_em": mean(float(task["scores"]["em"]) for task in tasks),
        "answer_f1": mean(float(task["scores"]["f1"]) for task in tasks),
        "official_accuracy": mean(
            float(task["scores"]["accuracy"]) for task in tasks
        ),
        "task_success_rate": mean(
            float(task["scores"]["success"]) for task in tasks
        ),
        "llm_calls": llm["calls"],
        "llm_total_tokens": llm["total_tokens"],
        "llm_prompt_tokens": llm["prompt_tokens"],
        "llm_completion_tokens": llm["completion_tokens"],
        "llm_reasoning_tokens": llm["reasoning_tokens"],
        "llm_cached_tokens": llm["cached_tokens"],
        "embedding_calls": embedding["calls"],
        "embedding_tokens": embedding["total_tokens"],
        "embedding_input_tokens": embedding["prompt_tokens"],
        "total_provider_tokens": total_provider_tokens,
        "store_queries": sum(
            int(task["retrieval"]["store_queries"]) for task in tasks
        ),
        "shared_store_queries": sum(
            int(task["retrieval"]["shared_store_queries"]) for task in tasks
        ),
        "private_store_queries": sum(
            int(task["retrieval"]["private_store_queries"]) for task in tasks
        ),
        "mean_recall_at_10": mean(
            float(task["retrieval"]["recall_at_10"]) for task in tasks
        ),
        "mean_mrr_at_10": mean(
            float(task["retrieval"]["mrr_at_10"]) for task in tasks
        ),
        "mean_retrieval_groups": mean(
            int(task["retrieval"]["groups"]) for task in tasks
        ),
        "mean_selected_context_items": selected / len(tasks),
        "mean_duplicate_evidence_rate": mean(
            float(task["context"]["mean_duplicate_evidence_rate"])
            for task in tasks
        ),
        "polluted_context_items": polluted,
        "context_pollution_rate": polluted / selected if selected else 0.0,
        "private_thinking_exposure": sum(
            int(task["context"]["private_thinking_exposure"])
            for task in tasks
        ),
        "cross_role_knowledge_exposure": sum(
            int(task["context"]["cross_role_knowledge_exposure"])
            for task in tasks
        ),
        "unauthorized_context_exposure": sum(
            int(task["safety"]["unauthorized_context_exposure"])
            for task in tasks
        ),
        "private_thinking_memories": sum(
            1
            for task in tasks
            for flow in task["data_flow"].values()
            if flow["private_memory_id"]
        ),
        "published_memories": sum(
            1
            for task in tasks
            for flow in task["data_flow"].values()
            if flow["published_memory_id"]
        ),
        "output_cap_hits": sum(
            int(task.get("generation_budget", {}).get("cap_hits", 0))
            for task in tasks
        ),
        "tasks_with_output_cap_hit": sum(
            bool(task.get("generation_budget", {}).get("agents_at_cap"))
            for task in tasks
        ),
        "mean_end_to_end_seconds": total_seconds / len(tasks),
        "mean_wall_clock_seconds": (
            total_wall_clock_seconds / len(tasks)
        ),
        "mean_embedding_seconds": total_embedding_seconds / len(tasks),
        "canonical_latency_excludes_embedding": True,
        "quality_per_1k_llm_tokens": (
            total_f1 / (llm["total_tokens"] / 1_000)
            if llm["total_tokens"]
            else 0.0
        ),
        "quality_per_1k_provider_tokens": (
            total_f1 / (total_provider_tokens / 1_000)
            if total_provider_tokens
            else 0.0
        ),
        "quality_per_second": (
            total_f1 / total_seconds if total_seconds else 0.0
        ),
    }
    if tasks and "base_pass" in tasks[0]["scores"]:
        aggregate["base_pass_at_1"] = mean(
            float(task["scores"]["base_pass"]) for task in tasks
        )
        aggregate["plus_pass_at_1"] = mean(
            float(task["scores"]["plus_pass"]) for task in tasks
        )
    return aggregate


def _ensure_per_agent_llm_usage(task: dict[str, Any]) -> None:
    """Backfill older reports from their ordered provider usage events."""
    if "per_agent_llm_usage" in task:
        return
    events = [
        event
        for event in task["provider_usage"].get("events", [])
        if event.get("category") == "llm"
    ]
    if len(events) != len(AGENTS):
        return
    usage_keys = (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
    )
    task["per_agent_llm_usage"] = {
        agent_id: {key: int(event.get(key, 0)) for key in usage_keys}
        for (agent_id, _, _), event in zip(AGENTS, events, strict=True)
    }


def _generation_budget(
    per_agent_usage: dict[str, dict[str, Any]],
    *,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    agents_at_cap = [
        agent_id
        for agent_id, usage in per_agent_usage.items()
        if max_output_tokens is not None
        and int(usage.get("completion_tokens", 0)) >= max_output_tokens
    ]
    return {
        "max_output_tokens": max_output_tokens,
        "agents_at_cap": agents_at_cap,
        "cap_hits": len(agents_at_cap),
    }


def _ensure_generation_budget(
    task: dict[str, Any],
    *,
    max_output_tokens: int | None,
) -> None:
    if "generation_budget" not in task and "per_agent_llm_usage" in task:
        task["generation_budget"] = _generation_budget(
            task["per_agent_llm_usage"],
            max_output_tokens=max_output_tokens,
        )


def _sum_usage(
    tasks: list[dict[str, Any]],
    category: str,
) -> dict[str, int]:
    keys = (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
    )
    return {
        key: sum(
            int(
                task["provider_usage"]["by_category"]
                .get(category, {})
                .get(key, 0)
            )
            for task in tasks
        )
        for key in keys
    }


def _comparisons(
    tasks: dict[SharingArm, list[dict[str, Any]]],
    overall: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    scoped = overall[SharingArm.COSCOPE.value]
    baselines = (
        SharingArm.FULL_SHARING,
        SharingArm.NO_SHARING,
    )
    comparisons = {}
    for index, baseline in enumerate(baselines):
        base = overall[baseline.value]
        comparisons[f"coscope_vs_{baseline.value}"] = {
            "answer_f1_delta": scoped["answer_f1"] - base["answer_f1"],
            "task_success_delta": (
                scoped["task_success_rate"] - base["task_success_rate"]
            ),
            "store_query_savings": _savings(
                scoped["store_queries"], base["store_queries"]
            ),
            "embedding_token_savings": _savings(
                scoped["embedding_tokens"], base["embedding_tokens"]
            ),
            "llm_token_savings": _savings(
                scoped["llm_total_tokens"], base["llm_total_tokens"]
            ),
            "provider_token_savings": _savings(
                scoped["total_provider_tokens"],
                base["total_provider_tokens"],
            ),
            "end_to_end_latency_improvement": _savings(
                scoped["mean_end_to_end_seconds"],
                base["mean_end_to_end_seconds"],
            ),
            "pollution_rate_reduction": (
                base["context_pollution_rate"]
                - scoped["context_pollution_rate"]
            ),
            "quality_per_1k_token_delta": (
                scoped["quality_per_1k_provider_tokens"]
                - base["quality_per_1k_provider_tokens"]
            ),
            "paired_bootstrap": _bootstrap_deltas(
                tasks[SharingArm.COSCOPE],
                tasks[baseline],
                samples=bootstrap_samples,
                seed=20260729 + index,
            ),
        }
    comparisons["acceptance"] = {
        "retrieval_savings_vs_no_sharing": (
            comparisons["coscope_vs_no_sharing"]["store_query_savings"] > 0
        ),
        "zero_context_pollution": scoped["context_pollution_rate"] == 0,
        "zero_private_thinking_exposure": (
            scoped["private_thinking_exposure"] == 0
        ),
        "quality_not_below_full_sharing": (
            scoped["answer_f1"]
            >= overall[SharingArm.FULL_SHARING.value]["answer_f1"]
        ),
        "quality_not_below_no_sharing": (
            scoped["answer_f1"]
            >= overall[SharingArm.NO_SHARING.value]["answer_f1"]
        ),
        "quality_efficiency_above_both_controls": all(
            scoped["quality_per_1k_provider_tokens"]
            >= overall[baseline.value]["quality_per_1k_provider_tokens"]
            for baseline in baselines
        ),
        "latency_below_both_controls": all(
            scoped["mean_end_to_end_seconds"]
            <= overall[baseline.value]["mean_end_to_end_seconds"]
            for baseline in baselines
        ),
    }
    return comparisons


def _bootstrap_deltas(
    scoped: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if len(scoped) != len(baseline):
        raise ValueError("paired arms must contain the same number of tasks")
    metric_paths = {
        "answer_f1": ("scores", "f1"),
        "task_success": ("scores", "success"),
        "llm_tokens": ("provider_usage", "by_category", "llm", "total_tokens"),
        "pollution_rate": ("context", "pollution_rate"),
        "end_to_end_seconds": ("latency", "end_to_end_seconds"),
    }
    rng = random.Random(seed)
    report = {}
    for metric, path in metric_paths.items():
        deltas = [
            float(_nested(left, path)) - float(_nested(right, path))
            for left, right in zip(scoped, baseline, strict=True)
        ]
        means = []
        for _ in range(samples):
            means.append(
                mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            )
        means.sort()
        report[metric] = {
            "mean_delta": mean(deltas),
            "ci95_low": means[int(0.025 * (samples - 1))],
            "ci95_high": means[int(0.975 * (samples - 1))],
        }
    return report


def _nested(item: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = item
    for key in path:
        value = value[key]
    return value


def _savings(value: int | float, baseline: int | float) -> float:
    return 1.0 - value / baseline if baseline else 0.0
