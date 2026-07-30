"""Live MAS evaluation across QA and mathematical reasoning benchmarks."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import replace
from statistics import mean
from typing import Any

from coscope import CoScopeRuntime, MemoryEntry
from coscope.config import CoScopeSettings
from coscope.context import ContextPacket
from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.code_benchmark import (
    CodeEvaluator,
    apply_mbpp_plus_scores,
    extract_python_solution,
    pending_code_score,
)
from coscope.evaluation.context_metrics import duplicate_evidence_rate
from coscope.evaluation.mas_experiment import (
    AGENT_CLASSES,
    AGENTS,
    ROLE_KNOWLEDGE,
)
from coscope.evaluation.math_grader import (
    extract_math_answer,
    grade_math_answer,
    normalize_math_answer,
)
from coscope.evaluation.retrieval_metrics import mrr_at_k, recall_at_k
from coscope.evaluation.runtime_metrics import runtime_report
from coscope.evaluation.task_metrics import (
    INVALID_ANSWER,
    aime_is_correct,
    extract_aime_answer,
    extract_final_answer,
    qa_answer_metrics,
)
from coscope.reasoning import ReasoningConfig
from coscope.retrieval import RetrievalRequest
from coscope.scope import Permission, ScopeDescriptor, Visibility


def run_benchmark_suite(
    examples_by_benchmark: dict[str, list[BenchmarkExample]],
    base_settings: CoScopeSettings,
    *,
    threshold: float = 0.82,
    max_output_tokens: int | None = None,
    max_output_tokens_by_benchmark: dict[str, int | None] | None = None,
    code_evaluator: CodeEvaluator | None = None,
) -> dict[str, Any]:
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
    reports = {}
    for benchmark, examples in examples_by_benchmark.items():
        benchmark_settings = replace(
            settings,
            llm=replace(
                settings.llm,
                max_tokens=output_caps[benchmark],
            ),
        )
        tasks = [
            _run_task(example, benchmark_settings) for example in examples
        ]
        if benchmark == "mbpp_plus":
            if code_evaluator is None:
                raise ValueError(
                    "MBPP-Plus requires the Docker EvalPlus code evaluator"
                )
            apply_mbpp_plus_scores(
                tasks,
                examples,
                code_evaluator,
                label="benchmark_suite_mbpp_plus",
            )
        reports[benchmark] = {
            "aggregate": _aggregate(tasks),
            "tasks": tasks,
        }
    return {
        "models": {
            "llm": settings.llm.model,
            "llm_temperature": settings.llm.temperature,
            "embedding": settings.embedding.model,
            "embedding_dimension": settings.embedding.dimension,
            "max_output_tokens_by_benchmark": output_caps,
        },
        "mas": {
            "agents": [agent_id for agent_id, _, _ in AGENTS],
            "agent_classes": {
                agent_id: {
                    "class_id": agent_class.class_id,
                    "knowledge_permissions": sorted(
                        agent_class.knowledge_permissions
                    ),
                    "readable_artifact_types": sorted(
                        item.value
                        for item in agent_class.context_policy.readable_artifact_types
                    ),
                }
                for agent_id, agent_class in AGENT_CLASSES.items()
            },
            "data_flow": (
                "private raw thinking; explicit planner team summary; "
                "directed solver-to-verifier summary"
            ),
            "knowledge_layout": (
                "common official task evidence plus role-restricted "
                "planner/solver/verifier knowledge"
            ),
            "final_answer_rule": (
                "sequential summaries; normalized majority; verifier tie-break"
            ),
            "gold_available_to_runtime": False,
        },
        "retrieval_threshold": threshold,
        "benchmarks": reports,
    }


def _run_task(
    example: BenchmarkExample,
    settings: CoScopeSettings,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = CoScopeRuntime.from_settings(settings)
    run_id = f"run_{example.benchmark}_{example.example_id}"
    runtime.create_run(example.benchmark, run_id=run_id)
    nodes = {}
    for agent_id, mode, _ in AGENTS:
        runtime.register_agent(AGENT_CLASSES[agent_id].instantiate(agent_id))
        _, node = runtime.start_reasoning(agent_id, ReasoningConfig(mode))
        nodes[agent_id] = node

    common_scope = ScopeDescriptor(
        frozenset({"benchmark_common"}),
        f"{run_id}/task",
        Visibility.TEAM_SHARED,
        memory_types=frozenset({"benchmark_evidence"}),
        trust_level="benchmark_common",
    )
    specialist_domains = {
        "planner": "planning_knowledge",
        "solver": "solving_knowledge",
        "verifier": "verification_knowledge",
    }
    specialist_scopes = {
        agent_id: ScopeDescriptor(
            frozenset({domain}),
            f"{run_id}/knowledge/{agent_id}",
            Visibility.RESTRICTED,
            permissions={
                f"role:{agent_id}": frozenset({Permission.READ}),
            },
            memory_types=frozenset({"benchmark_evidence", "role_knowledge"}),
            trust_level="benchmark_specialist",
        )
        for agent_id, domain in specialist_domains.items()
    }
    raw_memories: list[
        tuple[str, str, dict[str, object], ScopeDescriptor, str]
    ] = [
        (
            "question",
            example.question,
            {"benchmark": example.benchmark},
            common_scope,
            "user_input",
        ),
        *[
            (
                context.source_id,
                context.content,
                dict[str, object](
                    benchmark=example.benchmark,
                    retrieval_only=True,
                ),
                common_scope,
                "benchmark_context",
            )
            for context in example.context
        ],
        *[
            (
                f"{agent_id}_role_knowledge",
                ROLE_KNOWLEDGE[agent_id],
                dict[str, object](
                    benchmark=example.benchmark,
                    knowledge_class=agent_id,
                ),
                specialist_scopes[agent_id],
                "role_knowledge",
            )
            for agent_id, _, _ in AGENTS
        ],
    ]
    vectors = runtime.embedder.embed_many(
        [content for _, content, _, _, _ in raw_memories]
    )
    memory_by_source = {}
    for (source_id, content, metadata, scope, source_type), vector in zip(
        raw_memories, vectors, strict=True
    ):
        memory = runtime.ingest_memory(
            MemoryEntry(
                content,
                scope,
                source_type,
                source_id,
                vector=vector,
                metadata=metadata,
            )
        )
        memory_by_source[source_id] = memory.memory_id

    requests = _make_requests(runtime, run_id, nodes, example)
    retrieval_started = time.perf_counter()
    results = runtime.retrieve_batch(requests)
    retrieval_seconds = time.perf_counter() - retrieval_started
    executor = runtime.live_executor(
        system_instructions=_answer_instructions(example.benchmark)
    )
    generation_started = time.perf_counter()
    outputs = {}
    packets = []
    context_flow = {}
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
        recipients = (
            None
            if agent_id in {"planner", "verifier"}
            else frozenset({"verifier"})
        )
        private_entry, published_entry = runtime.record_thinking(
            actor_id=agent_id,
            reasoning_node_id=nodes[agent_id].node_id,
            private_content=output.text,
            public_summary=_extract_public_summary(
                output.text,
                fallback_answer=candidate,
                agent_id=agent_id,
            ),
            recipients=recipients,
        )
        context_flow[agent_id] = {
            "class_id": runtime.topology.agents[agent_id].agent_class_id,
            "effective_scope_ids": sorted(request.effective_view.scope_ids),
            "context_source_ids": [
                entry.source_id for entry in packet.all_memories
            ],
            "private_thinking_memory_id": private_entry.memory_id,
            "published_summary_memory_id": (
                published_entry.memory_id if published_entry else None
            ),
            "published_to": (
                "team" if recipients is None else sorted(recipients)
            ),
        }
    generation_seconds = time.perf_counter() - generation_started
    agent_answers = {
        agent_id: extract_benchmark_answer(example.benchmark, output.text)
        for agent_id, output in outputs.items()
    }
    prediction = _majority_answer(example.benchmark, agent_answers)
    score = score_benchmark_answer(example, prediction)

    supporting_ids = {
        memory_by_source[source_id]
        for source_id in example.supporting_source_ids
        if source_id in memory_by_source
    }
    if not supporting_ids:
        supporting_ids = {memory_by_source["question"]}
    gold_by_request = {
        request.request_id: supporting_ids for request in requests
    }
    unauthorized = _unauthorized_context_count(requests, packets)
    usage = runtime.usage.summary()
    return {
        "example_id": example.example_id,
        "prediction": prediction,
        "reference": _reference_answer(example),
        "scores": score,
        "task_success": bool(score["success"]),
        "agent_answers": agent_answers,
        "per_agent_llm_usage": {
            agent_id: output.metadata.get("usage", {})
            for agent_id, output in outputs.items()
        },
        "provider_usage": usage,
        "retrieval": {
            **runtime_report(runtime.retrieval.stats),
            "recall_at_10": recall_at_k(results, gold_by_request, k=10),
            "mrr_at_10": mrr_at_k(results, gold_by_request, k=10),
            "latency_seconds": retrieval_seconds,
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
        "safety": {"unauthorized_context_exposure": unauthorized},
        "latency": {
            "generation_seconds": generation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        },
    }


def _make_requests(
    runtime: CoScopeRuntime,
    run_id: str,
    nodes: dict[str, Any],
    example: BenchmarkExample,
) -> list[RetrievalRequest]:
    suffixes = {
        "planner": "identify the evidence and reasoning steps",
        "solver": "derive the exact final answer",
        "verifier": "verify evidence and check the final answer",
    }
    return [
        runtime.create_request(
            run_id=run_id,
            agent_id=agent_id,
            reasoning_node_id=nodes[agent_id].node_id,
            full_query=f"{example.question}\nRole objective: {suffixes[agent_id]}",
            public_intent=f"{example.question} {suffixes[agent_id]}",
            private_intent=f"{agent_id} specialist knowledge",
            full_query_is_public=True,
            retrieval_budget=10,
        )
        for agent_id, _, _ in AGENTS
    ]


def _answer_instructions(benchmark: str) -> tuple[str, ...]:
    common = (
        "Answer the benchmark question using the authorized context.",
        "Do not mention or guess the hidden reference answer.",
        (
            "Before the final answer, include exactly one line formatted as "
            "PUBLIC_SUMMARY: <a concise share-safe result for downstream agents>."
        ),
    )
    if benchmark == "gsm8k":
        return (
            *common,
            "End with exactly: #### <number>",
            "Do not place text after that line.",
        )
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
        "End with exactly: FINAL_ANSWER: <answer>",
        (
            "For MATH, put only the final mathematical expression after the "
            "marker; for AIME, put only an integer from 000 through 999."
        ),
        "Do not place text after that line.",
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


def extract_benchmark_answer(benchmark: str, output: str) -> str:
    if benchmark == "gsm8k":
        from coscope.evaluation.task_metrics import extract_gsm8k_answer

        return extract_gsm8k_answer(output)
    if benchmark == "math":
        return extract_math_answer(output) or INVALID_ANSWER
    if benchmark in {"aime2024", "aime2025"}:
        return extract_aime_answer(output)
    if benchmark == "mbpp_plus":
        return extract_python_solution(output)
    return extract_final_answer(output)


def _majority_answer(benchmark: str, answers: dict[str, str]) -> str:
    valid = {
        agent: answer
        for agent, answer in answers.items()
        if answer != INVALID_ANSWER
    }
    if not valid:
        return INVALID_ANSWER
    normalized = {
        agent: (
            normalize_math_answer(answer)
            if benchmark in {"math", "aime2024", "aime2025"}
            else answer.casefold().strip()
        )
        for agent, answer in valid.items()
    }
    counts = Counter(normalized.values())
    highest = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == highest}
    for preferred in ("verifier", "solver", "planner"):
        if preferred in normalized and normalized[preferred] in winners:
            return valid[preferred]
    return next(iter(valid.values()))


def score_benchmark_answer(
    example: BenchmarkExample,
    prediction: str,
) -> dict[str, float]:
    if prediction == INVALID_ANSWER:
        return {"em": 0.0, "f1": 0.0, "accuracy": 0.0, "success": 0.0}
    if example.benchmark == "math":
        accuracy = float(grade_math_answer(prediction, example.reference))
        return {
            "em": accuracy,
            "f1": accuracy,
            "accuracy": accuracy,
            "success": accuracy,
        }
    if example.benchmark == "gsm8k":
        from coscope.evaluation.task_metrics import gsm8k_is_correct

        accuracy = float(
            gsm8k_is_correct(f"#### {prediction}", example.reference)
        )
        return {
            "em": accuracy,
            "f1": accuracy,
            "accuracy": accuracy,
            "success": accuracy,
        }
    if example.benchmark in {"aime2024", "aime2025"}:
        accuracy = float(aime_is_correct(prediction, example.reference))
        return {
            "em": accuracy,
            "f1": accuracy,
            "accuracy": accuracy,
            "success": accuracy,
        }
    if example.benchmark == "mbpp_plus":
        return pending_code_score()
    qa = qa_answer_metrics(
        prediction,
        (example.reference, *example.aliases),
    )
    return {
        **qa,
        "accuracy": qa["em"],
        "success": qa["em"],
    }


def _reference_answer(example: BenchmarkExample) -> str:
    if example.benchmark == "math":
        return extract_math_answer(example.reference) or INVALID_ANSWER
    if example.benchmark == "mbpp_plus":
        return "[hidden EvalPlus base and plus tests]"
    return example.reference


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
    usage = [task["provider_usage"] for task in tasks]
    llm = _sum_usage(usage, "llm")
    embedding = _sum_usage(usage, "embedding")
    retrieval = [task["retrieval"] for task in tasks]
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
        "embedding_calls": embedding["calls"],
        "embedding_tokens": embedding["total_tokens"],
        "mean_retrieval_groups": mean(
            int(item["groups"]) for item in retrieval
        ),
        "mean_retrieval_savings": mean(
            float(item["retrieval_savings"]) for item in retrieval
        ),
        "mean_recall_at_10": mean(
            float(item["recall_at_10"]) for item in retrieval
        ),
        "mean_mrr_at_10": mean(
            float(item["mrr_at_10"]) for item in retrieval
        ),
        "unauthorized_context_exposure": sum(
            int(task["safety"]["unauthorized_context_exposure"])
            for task in tasks
        ),
        "mean_duplicate_evidence_rate": mean(
            float(task["context"]["mean_duplicate_evidence_rate"])
            for task in tasks
        ),
        "mean_selected_context_items_by_agent": {
            agent_id: mean(
                int(task["context"]["per_agent_selected_items"][agent_id])
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
        "mean_end_to_end_seconds": mean(
            float(task["latency"]["end_to_end_seconds"]) for task in tasks
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


def _sum_usage(
    reports: list[dict[str, Any]],
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
            for report in reports
        )
        for key in keys
    }
