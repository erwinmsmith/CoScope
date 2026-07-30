"""Live multi-agent benchmark and retrieval-threshold sweep."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from statistics import mean
from typing import Any

from coscope import (
    AgentClass,
    AgentContextPolicy,
    CoScopeRuntime,
    MemoryEntry,
)
from coscope.config import CoScopeSettings
from coscope.context import ContextPacket
from coscope.core import ArtifactType
from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.context_metrics import duplicate_evidence_rate
from coscope.evaluation.retrieval_metrics import mrr_at_k, recall_at_k
from coscope.evaluation.runtime_metrics import runtime_report
from coscope.evaluation.task_metrics import (
    INVALID_ANSWER,
    extract_gsm8k_answer,
    gsm8k_is_correct,
    task_success_rate,
)
from coscope.reasoning import ReasoningConfig, ReasoningMode
from coscope.retrieval import RetrievalRequest
from coscope.scope import Permission, ScopeDescriptor, Visibility


@dataclass(frozen=True)
class ThresholdPoint:
    medoid_threshold: float
    minimum_pairwise_similarity: float


AGENTS = (
    (
        "planner",
        ReasoningMode.COT,
        "derive a step-by-step arithmetic solution",
    ),
    (
        "solver",
        ReasoningMode.TOT,
        "calculate the final numeric answer",
    ),
    (
        "verifier",
        ReasoningMode.COT,
        "verify the arithmetic and final numeric answer",
    ),
)

AGENT_CLASSES = {
    "planner": AgentClass(
        "benchmark_planner",
        "planner",
        frozenset(
            {"benchmark_common", "planning_knowledge", "runtime_thinking"}
        ),
        AgentContextPolicy(
            readable_artifact_types=frozenset(
                {ArtifactType.PLAN, ArtifactType.REASONING_SUMMARY}
            ),
            publishable_artifact_types=frozenset(
                {ArtifactType.PLAN, ArtifactType.REASONING_SUMMARY}
            ),
            allow_public_thinking=True,
            default_context_budget=8_000,
            channel_budgets={
                "task_shared": 1_500,
                "agent_private": 1_000,
                "branch_local": 1_000,
                "retrieved_evidence": 4_500,
            },
        ),
    ),
    "solver": AgentClass(
        "benchmark_solver",
        "solver",
        frozenset(
            {"benchmark_common", "solving_knowledge", "runtime_thinking"}
        ),
        AgentContextPolicy(
            readable_artifact_types=frozenset(
                {
                    ArtifactType.PLAN,
                    ArtifactType.REASONING_SUMMARY,
                    ArtifactType.DECISION,
                }
            ),
            publishable_artifact_types=frozenset(
                {ArtifactType.DECISION, ArtifactType.REASONING_SUMMARY}
            ),
            allow_public_thinking=True,
            default_context_budget=8_000,
            channel_budgets={
                "task_shared": 2_000,
                "agent_private": 1_000,
                "branch_local": 1_000,
                "retrieved_evidence": 5_000,
            },
        ),
    ),
    "verifier": AgentClass(
        "benchmark_verifier",
        "verifier",
        frozenset(
            {
                "benchmark_common",
                "verification_knowledge",
                "runtime_thinking",
            }
        ),
        AgentContextPolicy(
            readable_artifact_types=frozenset(
                {
                    ArtifactType.PLAN,
                    ArtifactType.REASONING_SUMMARY,
                    ArtifactType.DECISION,
                    ArtifactType.VERIFICATION_RESULT,
                }
            ),
            publishable_artifact_types=frozenset(
                {
                    ArtifactType.REASONING_SUMMARY,
                    ArtifactType.VERIFICATION_RESULT,
                }
            ),
            allow_public_thinking=True,
            default_context_budget=8_000,
            channel_budgets={
                "task_shared": 2_500,
                "agent_private": 1_000,
                "branch_local": 1_000,
                "retrieved_evidence": 4_500,
            },
        ),
    ),
}

ROLE_KNOWLEDGE = {
    "planner": (
        "Private planner protocol: identify required evidence, decompose the "
        "task, and publish only a concise plan or evidence summary."
    ),
    "solver": (
        "Private solver protocol: combine authorized evidence with published "
        "plans, calculate precisely, and publish only a concise result summary."
    ),
    "verifier": (
        "Private verifier protocol: independently check evidence and candidate "
        "answers, resolve disagreements, and reject unsupported conclusions."
    ),
}


def run_gsm8k_threshold_sweep(
    examples: list[BenchmarkExample],
    base_settings: CoScopeSettings,
    threshold_points: list[ThresholdPoint],
    *,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if not base_settings.live:
        raise ValueError("the MAS benchmark requires live provider mode")
    if not examples or not threshold_points:
        raise ValueError("examples and threshold_points cannot be empty")

    points: list[dict[str, Any]] = []
    for point in threshold_points:
        retrieval = replace(
            base_settings.retrieval,
            medoid_threshold=point.medoid_threshold,
            minimum_pairwise_similarity=point.minimum_pairwise_similarity,
        )
        settings = replace(
            base_settings,
            llm=replace(
                base_settings.llm,
                max_tokens=(
                    base_settings.llm.max_tokens
                    if max_output_tokens is None
                    else max_output_tokens
                ),
            ),
            retrieval=retrieval,
        )
        task_results = [_run_gsm8k_task(example, settings) for example in examples]
        points.append(
            {
                "thresholds": asdict(point),
                "aggregate": _aggregate(task_results),
                "tasks": task_results,
            }
        )

    return {
        "benchmark": {
            "name": "GSM8K",
            "split": "test",
            "official_metric": "accuracy",
            "official_answer_extraction": "numeric value immediately after ####",
            "sample_size": len(examples),
            "example_ids": [example.example_id for example in examples],
        },
        "models": {
            "llm": base_settings.llm.model,
            "llm_temperature": base_settings.llm.temperature,
            "max_output_tokens": max_output_tokens,
            "embedding": base_settings.embedding.model,
            "embedding_dimension": base_settings.embedding.dimension,
        },
        "mas": {
            "agents": [agent_id for agent_id, _, _ in AGENTS],
            "agent_classes": {
                agent_id: agent_class.class_id
                for agent_id, agent_class in AGENT_CLASSES.items()
            },
            "data_flow": (
                "private raw thinking; explicit planner team summary; "
                "directed solver-to-verifier summary"
            ),
            "final_answer_rule": (
                "sequential summaries; majority vote; verifier tie-break"
            ),
            "gold_available_to_runtime": False,
        },
        "threshold_points": points,
    }


def _run_gsm8k_task(
    example: BenchmarkExample,
    settings: CoScopeSettings,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = CoScopeRuntime.from_settings(settings)
    run_id = f"run_{example.example_id}"
    runtime.create_run("gsm8k", run_id=run_id)
    nodes = {}
    for agent_id, reasoning_mode, _ in AGENTS:
        runtime.register_agent(AGENT_CLASSES[agent_id].instantiate(agent_id))
        _, node = runtime.start_reasoning(
            agent_id,
            ReasoningConfig(reasoning_mode),
        )
        nodes[agent_id] = node

    scope = ScopeDescriptor(
        frozenset({"benchmark_common"}),
        f"{run_id}/task",
        Visibility.TEAM_SHARED,
        memory_types=frozenset({"benchmark_evidence"}),
        trust_level="benchmark_common",
    )
    question_memory = runtime.ingest_memory(
        MemoryEntry(
            example.question,
            scope,
            "user_input",
            example.example_id,
        )
    )
    specialist_domains = {
        "planner": "planning_knowledge",
        "solver": "solving_knowledge",
        "verifier": "verification_knowledge",
    }
    for agent_id, domain in specialist_domains.items():
        runtime.ingest_memory(
            MemoryEntry(
                ROLE_KNOWLEDGE[agent_id],
                ScopeDescriptor(
                    frozenset({domain}),
                    f"{run_id}/knowledge/{agent_id}",
                    Visibility.RESTRICTED,
                    permissions={
                        f"role:{agent_id}": frozenset({Permission.READ})
                    },
                    memory_types=frozenset({"role_knowledge"}),
                    trust_level="benchmark_specialist",
                ),
                "role_knowledge",
                f"{agent_id}_role_knowledge",
            )
        )

    requests: list[RetrievalRequest] = []
    for agent_id, _, public_intent in AGENTS:
        requests.append(
            runtime.create_request(
                run_id=run_id,
                agent_id=agent_id,
                reasoning_node_id=nodes[agent_id].node_id,
                full_query=(
                    f"Solve this GSM8K problem as the {agent_id}. "
                    "Use the authorized problem statement in context."
                ),
                public_intent=public_intent,
                private_intent=f"{agent_id} specialist knowledge",
                full_query_is_public=True,
                context_budget=2_000,
                retrieval_budget=5,
            )
        )

    retrieval_started = time.perf_counter()
    results = runtime.retrieve_batch(requests)
    retrieval_seconds = time.perf_counter() - retrieval_started
    executor = runtime.live_executor(
        system_instructions=(
            "Solve the arithmetic word problem independently.",
            (
                "Before the final answer, include exactly one line formatted as "
                "PUBLIC_SUMMARY: <a concise share-safe result for downstream agents>."
            ),
            "End with one final line formatted exactly as: #### <number>",
            "Do not place any text after that final line.",
        )
    )

    outputs: dict[str, Any] = {}
    packets = []
    context_flow = {}
    published_summaries = {}
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
        candidate = extract_gsm8k_answer(output.text)
        recipients = (
            None
            if agent_id in {"planner", "verifier"}
            else frozenset({"verifier"})
        )
        public_summary = _extract_public_summary(
            output.text,
            fallback_answer=candidate,
            agent_id=agent_id,
        )
        private_entry, public_entry = runtime.record_thinking(
            actor_id=agent_id,
            reasoning_node_id=nodes[agent_id].node_id,
            private_content=output.text,
            public_summary=public_summary,
            recipients=recipients,
        )
        published_summaries[agent_id] = public_summary
        context_flow[agent_id] = {
            "class_id": runtime.topology.agents[agent_id].agent_class_id,
            "context_source_ids": [
                entry.source_id for entry in packet.all_memories
            ],
            "private_thinking_memory_id": private_entry.memory_id,
            "published_summary_memory_id": (
                public_entry.memory_id if public_entry else None
            ),
            "published_to": (
                "team" if recipients is None else sorted(recipients)
            ),
        }
    generation_seconds = time.perf_counter() - generation_started

    extracted = {
        agent_id: extract_gsm8k_answer(output.text)
        for agent_id, output in outputs.items()
    }
    final_answer = _majority_answer(extracted)
    final_completion = f"#### {final_answer}"
    success = (
        final_answer != INVALID_ANSWER
        and gsm8k_is_correct(final_completion, example.reference)
    )
    gold = extract_gsm8k_answer(example.reference)
    gold_by_request = {
        request.request_id: {question_memory.memory_id} for request in requests
    }
    unauthorized = _unauthorized_context_count(requests, packets)
    per_agent_usage = {
        agent_id: dict(output.metadata.get("usage", {}))
        for agent_id, output in outputs.items()
    }
    llm_tokens = sum(
        int(usage.get("total_tokens", 0)) for usage in per_agent_usage.values()
    )
    single_agent_tokens = int(
        per_agent_usage.get("solver", {}).get("total_tokens", 0)
    )
    usage = runtime.usage.summary()

    return {
        "example_id": example.example_id,
        "official_prediction": final_answer,
        "official_reference": gold,
        "task_success": success,
        "agent_answers": extracted,
        "published_summaries": published_summaries,
        "per_agent_llm_usage": per_agent_usage,
        "mas_llm_tokens": llm_tokens,
        "single_solver_llm_tokens": single_agent_tokens,
        "mas_llm_token_overhead_ratio": (
            llm_tokens / single_agent_tokens - 1 if single_agent_tokens else 0.0
        ),
        "provider_usage": usage,
        "retrieval": {
            **runtime_report(runtime.retrieval.stats),
            "recall_at_5": recall_at_k(results, gold_by_request, k=5),
            "mrr_at_5": mrr_at_k(results, gold_by_request, k=5),
            "latency_seconds": retrieval_seconds,
            "group_ids": sorted(
                {
                    result.group_id
                    for result in results.values()
                    if result.group_id is not None
                }
            ),
        },
        "context": {
            "mean_duplicate_evidence_rate": mean(
                duplicate_evidence_rate(packet) for packet in packets
            ),
            "mean_selected_items": mean(len(packet.all_memories) for packet in packets),
            "per_agent_selected_items": {
                agent_id: len(packet.all_memories)
                for (agent_id, _, _), packet in zip(
                    AGENTS, packets, strict=True
                )
            },
        },
        "data_flow": context_flow,
        "safety": {
            "unauthorized_context_exposure": unauthorized,
        },
        "latency": {
            "generation_seconds": generation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        },
    }


def _majority_answer(extracted: dict[str, str]) -> str:
    valid = [answer for answer in extracted.values() if answer != INVALID_ANSWER]
    if not valid:
        return INVALID_ANSWER
    counts = Counter(valid)
    highest = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == highest}
    verifier = extracted.get("verifier", INVALID_ANSWER)
    if verifier in winners:
        return verifier
    solver = extracted.get("solver", INVALID_ANSWER)
    if solver in winners:
        return solver
    return sorted(winners)[0]


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


def _unauthorized_context_count(
    requests: list[RetrievalRequest],
    packets: list[ContextPacket],
) -> int:
    return sum(
        entry.scope.scope_id not in request.effective_view.scope_ids
        for request, packet in zip(requests, packets, strict=True)
        for entry in packet.all_memories
    )


def _aggregate(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(task["task_success"]) for task in tasks]
    retrieval = [task["retrieval"] for task in tasks]
    context = [task["context"] for task in tasks]
    safety = [task["safety"] for task in tasks]
    latency = [task["latency"] for task in tasks]
    usage = [task["provider_usage"] for task in tasks]

    total_provider_tokens = sum(
        int(item["all"]["total_tokens"]) for item in usage
    )
    total_embedding_tokens = sum(
        int(item["by_category"].get("embedding", {}).get("total_tokens", 0))
        for item in usage
    )
    llm_usage = _sum_usage_category(usage, "llm")
    embedding_usage = _sum_usage_category(usage, "embedding")
    total_llm_tokens = sum(int(task["mas_llm_tokens"]) for task in tasks)
    return {
        "tasks": len(tasks),
        "official_accuracy": task_success_rate(successes),
        "task_success_rate": task_success_rate(successes),
        "mas_total_provider_tokens": total_provider_tokens,
        "mas_total_llm_tokens": total_llm_tokens,
        "mas_total_embedding_tokens": total_embedding_tokens,
        "llm_calls": llm_usage["calls"],
        "llm_prompt_tokens": llm_usage["prompt_tokens"],
        "llm_completion_tokens": llm_usage["completion_tokens"],
        "llm_reasoning_tokens": llm_usage["reasoning_tokens"],
        "llm_cached_tokens": llm_usage["cached_tokens"],
        "embedding_calls": embedding_usage["calls"],
        "embedding_prompt_tokens": embedding_usage["prompt_tokens"],
        "mean_provider_tokens_per_task": total_provider_tokens / len(tasks),
        "mean_llm_token_overhead_ratio": mean(
            float(task["mas_llm_token_overhead_ratio"]) for task in tasks
        ),
        "mean_retrieval_groups": mean(
            int(item["groups"]) for item in retrieval
        ),
        "mean_retrieval_savings": mean(
            float(item["retrieval_savings"]) for item in retrieval
        ),
        "shared_store_queries": sum(
            int(item["shared_store_queries"]) for item in retrieval
        ),
        "private_store_queries": sum(
            int(item["private_store_queries"]) for item in retrieval
        ),
        "fallback_triggers": sum(
            int(item["fallback_triggers"]) for item in retrieval
        ),
        "mean_recall_at_5": mean(
            float(item["recall_at_5"]) for item in retrieval
        ),
        "mean_mrr_at_5": mean(
            float(item["mrr_at_5"]) for item in retrieval
        ),
        "mean_duplicate_evidence_rate": mean(
            float(item["mean_duplicate_evidence_rate"])
            for item in context
        ),
        "mean_selected_context_items": mean(
            float(item["mean_selected_items"]) for item in context
        ),
        "mean_selected_context_items_by_agent": {
            agent_id: mean(
                int(
                    task["context"]["per_agent_selected_items"][agent_id]
                )
                for task in tasks
            )
            for agent_id, _, _ in AGENTS
        },
        "private_thinking_memories": sum(
            1
            for task in tasks
            for flow in task["data_flow"].values()
            if flow["private_thinking_memory_id"]
        ),
        "published_summaries": sum(
            1
            for task in tasks
            for flow in task["data_flow"].values()
            if flow["published_summary_memory_id"]
        ),
        "unauthorized_context_exposure": sum(
            int(item["unauthorized_context_exposure"])
            for item in safety
        ),
        "mean_end_to_end_seconds": mean(
            float(item["end_to_end_seconds"]) for item in latency
        ),
        "mean_retrieval_seconds": mean(
            float(item["latency_seconds"]) for item in retrieval
        ),
        "mean_generation_seconds": mean(
            float(item["generation_seconds"]) for item in latency
        ),
    }


def _sum_usage_category(
    usage_reports: list[dict[str, Any]],
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
            int(report["by_category"].get(category, {}).get(key, 0))
            for report in usage_reports
        )
        for key in keys
    }
