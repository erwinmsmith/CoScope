"""Compare CoT with retrieval-aware ToT on identical benchmark samples."""

from __future__ import annotations

import time
from dataclasses import replace
from statistics import mean
from typing import Any, cast

from coscope import AgentInstance, CoScopeRuntime, MemoryEntry
from coscope.config import CoScopeSettings
from coscope.evaluation.benchmark_suite import (
    extract_benchmark_answer,
    score_benchmark_answer,
)
from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.code_benchmark import (
    CodeEvaluator,
    apply_mbpp_plus_scores,
)
from coscope.evaluation.math_grader import extract_math_answer
from coscope.reasoning import (
    ReasoningConfig,
    ReasoningMode,
    ToTRuntime,
    execute_live_strategy,
)
from coscope.reasoning.live_strategies import TOT_BRANCH_STRATEGIES
from coscope.scope import ScopeDescriptor, Visibility

ACTIVE_REASONING_MODES = (ReasoningMode.COT, ReasoningMode.TOT)


def run_reasoning_mode_comparison(
    examples_by_benchmark: dict[str, list[BenchmarkExample]],
    base_settings: CoScopeSettings,
    *,
    max_output_tokens: int | None = None,
    max_output_tokens_by_benchmark: dict[str, int | None] | None = None,
    code_evaluator: CodeEvaluator | None = None,
) -> dict[str, Any]:
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
    benchmarks = {}
    for benchmark, examples in examples_by_benchmark.items():
        settings = replace(
            base_settings,
            llm=replace(
                base_settings.llm,
                max_tokens=output_caps[benchmark],
            ),
        )
        mode_results = {}
        for mode in ACTIVE_REASONING_MODES:
            tasks = [_run_mode_task(example, mode, settings) for example in examples]
            if benchmark == "mbpp_plus":
                if code_evaluator is None:
                    raise ValueError(
                        "MBPP-Plus requires the Docker EvalPlus code evaluator"
                    )
                apply_mbpp_plus_scores(
                    tasks,
                    examples,
                    code_evaluator,
                    label=f"reasoning_modes_mbpp_plus_{mode.value}",
                )
            mode_results[mode.value] = {
                "aggregate": _aggregate(tasks),
                "tasks": tasks,
            }
        benchmarks[benchmark] = mode_results
    return {
        "models": {
            "llm": settings.llm.model,
            "llm_temperature": settings.llm.temperature,
            "embedding": settings.embedding.model,
            "max_output_tokens_by_benchmark": output_caps,
        },
        "method_alignment": {
            "cot": "single chain generation",
            "tot": (
                "three branch-specific retrieval requests batched through "
                "CoScope, then three thoughts plus value/vote selection"
            ),
        },
        "gold_available_to_runtime": False,
        "benchmarks": benchmarks,
    }


def _run_mode_task(
    example: BenchmarkExample,
    mode: ReasoningMode,
    settings: CoScopeSettings,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = CoScopeRuntime.from_settings(settings)
    run_id = f"mode_{mode.value}_{example.benchmark}_{example.example_id}"
    runtime.create_run(example.benchmark, run_id=run_id)
    runtime.register_agent(
        AgentInstance(
            "reasoner",
            f"{mode.value} reasoner",
            knowledge_permissions=frozenset({"benchmark_input"}),
        )
    )
    reasoning, root = runtime.start_reasoning(
        "reasoner",
        ReasoningConfig(mode, max_depth=4, branching_factor=3),
    )
    scope = ScopeDescriptor(
        frozenset({"benchmark_input"}),
        f"{run_id}/task",
        Visibility.TEAM_SHARED,
        trust_level="benchmark_input",
    )
    raw_memories: list[tuple[str, str, dict[str, object]]] = [
        ("question", example.question, {"benchmark": example.benchmark}),
        *[
            (
                item.source_id,
                item.content,
                {"benchmark": example.benchmark, "retrieval_only": True},
            )
            for item in example.context
        ],
    ]
    vectors = runtime.embedder.embed_many([item[1] for item in raw_memories])
    for (source_id, content, metadata), vector in zip(
        raw_memories, vectors, strict=True
    ):
        runtime.ingest_memory(
            MemoryEntry(
                content,
                scope,
                "user_input" if source_id == "question" else "benchmark_context",
                source_id,
                vector=vector,
                metadata=metadata,
            )
        )
    tot_branches = None
    if mode == ReasoningMode.TOT:
        if not isinstance(reasoning, ToTRuntime):
            raise TypeError("ToT mode requires ToTRuntime")
        tot_branches = reasoning.branch(
            root.node_id,
            reasoning.config.branching_factor,
        )
        runtime.refresh_reasoning_nodes("reasoner", reasoning)
        requests = [
            runtime.create_request(
                run_id=run_id,
                agent_id="reasoner",
                reasoning_node_id=branch.node_id,
                full_query=f"{example.question}\nRetrieval strategy: {strategy}",
                public_intent=f"{example.question} find public evidence and solve",
                private_intent=f"branch-local strategy: {strategy}",
                context_budget=8_000,
                retrieval_budget=10,
            )
            for branch, strategy in zip(
                tot_branches,
                TOT_BRANCH_STRATEGIES,
                strict=True,
            )
        ]
    else:
        requests = [
            runtime.create_request(
                run_id=run_id,
                agent_id="reasoner",
                reasoning_node_id=root.node_id,
                full_query=example.question,
                public_intent=f"{example.question} find evidence and solve",
                full_query_is_public=True,
                context_budget=8_000,
                retrieval_budget=10,
            )
        ]
    results = runtime.retrieve_batch(requests)
    packets = {
        request.reasoning_node_id: runtime.assemble_context(
            request,
            results[request.request_id],
        )
        for request in requests
    }
    contexts_by_node = {
        node_id: "\n\n".join(
            f"[{entry.source_id}] {entry.content}"
            for entry in packet.all_memories
        )
        for node_id, packet in packets.items()
    }
    authorized_context = contexts_by_node.get(root.node_id, "")
    generation_started = time.perf_counter()
    outcome = execute_live_strategy(
        mode,
        reasoning,  # type: ignore[arg-type]
        root,
        runtime.llm,  # type: ignore[arg-type]
        question=example.question,
        authorized_context=authorized_context,
        answer_instruction=_answer_instruction(example.benchmark),
        tot_branches=tot_branches,
        tot_branch_contexts=contexts_by_node if tot_branches else None,
    )
    generation_seconds = time.perf_counter() - generation_started
    prediction = extract_benchmark_answer(example.benchmark, outcome.output)
    scores = score_benchmark_answer(example, prediction)
    usage = runtime.usage.summary()
    by_category = cast(dict[str, dict[str, int]], usage["by_category"])
    llm_usage = by_category.get("llm", {})
    return {
        "example_id": example.example_id,
        "mode": mode.value,
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
        "reasoning": {
            "node_count": outcome.node_count,
            "generated_thoughts": outcome.generated_thoughts,
            "llm_calls": outcome.llm_calls,
            "metadata": outcome.metadata,
        },
        "provider_usage": usage,
        "llm_tokens": int(llm_usage.get("total_tokens", 0)),
        "embedding_tokens": int(
            by_category.get("embedding", {}).get("total_tokens", 0)
        ),
        "retrieval": {
            "requests": len(requests),
            "groups": runtime.retrieval.stats.groups,
            "shared_store_queries": runtime.retrieval.stats.shared_store_queries,
            "private_store_queries": runtime.retrieval.stats.private_store_queries,
            "query_savings": (
                1.0
                - (
                    runtime.retrieval.stats.shared_store_queries
                    + runtime.retrieval.stats.private_store_queries
                )
                / len(requests)
            ),
            "candidates": sum(
                len(results[request.request_id].candidates)
                for request in requests
            ),
            "group_ids": {
                request.reasoning_node_id: results[
                    request.request_id
                ].group_id
                for request in requests
            },
            "public_queries": {
                request.reasoning_node_id: request.public_intent
                for request in requests
            },
            "context_source_ids": {
                node_id: [entry.source_id for entry in packet.all_memories]
                for node_id, packet in packets.items()
            },
        },
        "safety": {
            "unauthorized_context_exposure": sum(
                entry.scope.scope_id not in request.effective_view.scope_ids
                for request in requests
                for entry in packets[request.reasoning_node_id].all_memories
            )
        },
        "latency": {
            "generation_seconds": generation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        },
    }


def _answer_instruction(benchmark: str) -> str:
    if benchmark == "gsm8k":
        return "End with exactly #### <number> and no text after it."
    if benchmark == "mbpp_plus":
        return (
            "Return a self-contained Python solution defining the requested "
            "function. End with FINAL_CODE: followed by one fenced Python "
            "code block and no text after the closing fence."
        )
    return (
        "End with exactly FINAL_ANSWER: <answer> and no text after it. "
        "For MATH, provide only the final mathematical expression; for AIME, "
        "provide only an integer from 000 through 999."
    )


def _aggregate(tasks: list[dict[str, Any]]) -> dict[str, int | float]:
    aggregate: dict[str, int | float] = {
        "tasks": len(tasks),
        "answer_em": mean(float(task["scores"]["em"]) for task in tasks),
        "answer_f1": mean(float(task["scores"]["f1"]) for task in tasks),
        "task_success_rate": mean(
            float(task["scores"]["success"]) for task in tasks
        ),
        "llm_calls": sum(
            int(task["reasoning"]["llm_calls"]) for task in tasks
        ),
        "mean_reasoning_nodes": mean(
            int(task["reasoning"]["node_count"]) for task in tasks
        ),
        "mean_generated_thoughts": mean(
            int(task["reasoning"]["generated_thoughts"]) for task in tasks
        ),
        "llm_tokens": sum(int(task["llm_tokens"]) for task in tasks),
        "embedding_tokens": sum(
            int(task["embedding_tokens"]) for task in tasks
        ),
        "retrieval_requests": sum(
            int(task["retrieval"]["requests"]) for task in tasks
        ),
        "shared_store_queries": sum(
            int(task["retrieval"]["shared_store_queries"]) for task in tasks
        ),
        "private_store_queries": sum(
            int(task["retrieval"]["private_store_queries"]) for task in tasks
        ),
        "mean_retrieval_groups": mean(
            int(task["retrieval"]["groups"]) for task in tasks
        ),
        "mean_query_savings": mean(
            float(task["retrieval"]["query_savings"]) for task in tasks
        ),
        "mean_generation_seconds": mean(
            float(task["latency"]["generation_seconds"]) for task in tasks
        ),
        "mean_end_to_end_seconds": mean(
            float(task["latency"]["end_to_end_seconds"]) for task in tasks
        ),
        "unauthorized_context_exposure": sum(
            int(task["safety"]["unauthorized_context_exposure"])
            for task in tasks
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
