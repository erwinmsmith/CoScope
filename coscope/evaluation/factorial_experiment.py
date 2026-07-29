"""Uniform reasoning-mode x sharing-policy multi-agent experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from statistics import mean
from typing import Any, cast

from coscope import CoScopeRuntime
from coscope.config import CoScopeSettings
from coscope.context import ContextPacket
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
from coscope.evaluation.mas_experiment import AGENTS
from coscope.evaluation.math_grader import extract_math_answer
from coscope.evaluation.retrieval_metrics import mrr_at_k, recall_at_k
from coscope.evaluation.sharing_ablation import (
    SharingArm,
    _agent_class_for_arm,
    _aggregate,
    _answer_instructions,
    _comparisons,
    _ingest_arm_memories,
    _write_arm_thinking,
)
from coscope.reasoning import (
    GoTRuntime,
    ReasoningConfig,
    ReasoningMode,
    ToTRuntime,
    execute_live_strategy,
)
from coscope.reasoning.live_strategies import (
    GOT_SOURCE_OPERATIONS,
    TOT_BRANCH_STRATEGIES,
    ReasoningOutcome,
)
from coscope.reasoning.node import ReasoningNode
from coscope.retrieval import RetrievalRequest


@dataclass(frozen=True)
class FactorialModeSpec:
    mode: ReasoningMode
    retrieval_operations: tuple[str, ...]
    llm_calls_per_agent: int
    default_enabled: bool


FACTORIAL_MODE_SPECS: dict[ReasoningMode, FactorialModeSpec] = {
    ReasoningMode.COT: FactorialModeSpec(
        ReasoningMode.COT,
        ("follow one linear reasoning path",),
        1,
        True,
    ),
    ReasoningMode.TOT: FactorialModeSpec(
        ReasoningMode.TOT,
        TOT_BRANCH_STRATEGIES,
        len(TOT_BRANCH_STRATEGIES) + 1,
        True,
    ),
    ReasoningMode.GOT: FactorialModeSpec(
        ReasoningMode.GOT,
        GOT_SOURCE_OPERATIONS,
        len(GOT_SOURCE_OPERATIONS) + 1,
        False,
    ),
}
DEFAULT_FACTORIAL_MODES = tuple(
    spec.mode for spec in FACTORIAL_MODE_SPECS.values() if spec.default_enabled
)
DEFAULT_SHARING_ARMS = tuple(SharingArm)

ROLE_OBJECTIVES = {
    "planner": "identify the evidence requirements and construct a solution plan",
    "solver": "derive the exact final answer from authorized evidence",
    "verifier": "independently verify the evidence and final answer",
}


def parse_reasoning_modes(value: str) -> tuple[ReasoningMode, ...]:
    names = [item.strip().casefold() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one reasoning mode is required")
    try:
        modes = tuple(ReasoningMode(name) for name in names)
    except ValueError as error:
        raise ValueError("reasoning modes must be selected from cot, tot, got") from error
    if len(modes) != len(set(modes)):
        raise ValueError("reasoning modes cannot contain duplicates")
    return modes


def parse_sharing_arms(value: str) -> tuple[SharingArm, ...]:
    names = [item.strip().casefold() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one sharing policy is required")
    try:
        arms = tuple(SharingArm(name) for name in names)
    except ValueError as error:
        raise ValueError(
            "sharing policies must be selected from coscope, full_sharing, no_sharing"
        ) from error
    if len(arms) != len(set(arms)):
        raise ValueError("sharing policies cannot contain duplicates")
    return arms


def run_factorial_experiment(
    examples_by_benchmark: dict[str, list[BenchmarkExample]],
    base_settings: CoScopeSettings,
    *,
    reasoning_modes: tuple[ReasoningMode, ...] = DEFAULT_FACTORIAL_MODES,
    sharing_arms: tuple[SharingArm, ...] = DEFAULT_SHARING_ARMS,
    threshold: float = 0.75,
    max_output_tokens: int | None = None,
    max_output_tokens_by_benchmark: dict[str, int | None] | None = None,
    bootstrap_samples: int = 1_000,
    code_evaluator: CodeEvaluator | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Run identical examples under each uniform mode and sharing policy."""
    if not reasoning_modes or not sharing_arms:
        raise ValueError("reasoning modes and sharing arms cannot be empty")
    if len(reasoning_modes) != len(set(reasoning_modes)):
        raise ValueError("reasoning modes cannot contain duplicates")
    if len(sharing_arms) != len(set(sharing_arms)):
        raise ValueError("sharing arms cannot contain duplicates")
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
        output_caps[benchmark] = settings.llm.max_tokens if cap is None else cap
    if any(cap is not None and cap <= 0 for cap in output_caps.values()):
        raise ValueError("max output token caps must be positive when set")

    reports: dict[str, dict[str, Any]] = {}
    all_tasks: dict[
        ReasoningMode,
        dict[SharingArm, list[dict[str, Any]]],
    ] = {
        mode: {arm: [] for arm in sharing_arms}
        for mode in reasoning_modes
    }
    for benchmark, examples in examples_by_benchmark.items():
        benchmark_settings = replace(
            settings,
            llm=replace(settings.llm, max_tokens=output_caps[benchmark]),
        )
        condition_tasks: dict[
            ReasoningMode,
            dict[SharingArm, list[dict[str, Any]]],
        ] = {
            mode: {arm: [] for arm in sharing_arms}
            for mode in reasoning_modes
        }
        for example in examples:
            for mode in reasoning_modes:
                for arm in sharing_arms:
                    task = _run_factorial_task(
                        example,
                        benchmark_settings,
                        mode,
                        arm,
                    )
                    condition_tasks[mode][arm].append(task)
                    all_tasks[mode][arm].append(task)
                    if progress is not None:
                        progress(
                            {
                                "benchmark": benchmark,
                                "example_id": example.example_id,
                                "reasoning_mode": mode.value,
                                "sharing_policy": arm.value,
                                "answer_f1": task["scores"]["f1"],
                                "retrieval_requests": task["retrieval"]["requests"],
                                "retrieval_groups": task["retrieval"]["groups"],
                                "store_queries": task["retrieval"]["store_queries"],
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
            for mode in reasoning_modes:
                for arm in sharing_arms:
                    apply_mbpp_plus_scores(
                        condition_tasks[mode][arm],
                        examples,
                        code_evaluator,
                        label=(
                            f"factorial_mbpp_plus_{mode.value}_{arm.value}"
                        ),
                    )
        reports[benchmark] = {
            mode.value: {
                arm.value: {
                    "aggregate": _factorial_aggregate(
                        condition_tasks[mode][arm]
                    ),
                    "tasks": condition_tasks[mode][arm],
                }
                for arm in sharing_arms
            }
            for mode in reasoning_modes
        }

    overall = {
        mode.value: {
            arm.value: _factorial_aggregate(all_tasks[mode][arm])
            for arm in sharing_arms
        }
        for mode in reasoning_modes
    }
    comparisons_by_mode = {
        mode.value: _comparisons(
            cast(dict[SharingArm, list[dict[str, Any]]], all_tasks[mode]),
            overall[mode.value],
            bootstrap_samples=bootstrap_samples,
        )
        for mode in reasoning_modes
        if set(sharing_arms) == set(DEFAULT_SHARING_ARMS)
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
            "factorial": True,
            "uniform_mode_across_agents": True,
            "reasoning_modes": [mode.value for mode in reasoning_modes],
            "sharing_policies": [arm.value for arm in sharing_arms],
            "agent_topology": "planner -> solver -> verifier",
            "agent_count": len(AGENTS),
            "mode_specs": {
                mode.value: {
                    "retrieval_requests_per_agent": len(
                        FACTORIAL_MODE_SPECS[mode].retrieval_operations
                    ),
                    "llm_calls_per_agent": FACTORIAL_MODE_SPECS[
                        mode
                    ].llm_calls_per_agent,
                }
                for mode in reasoning_modes
            },
            "got_extension_available": ReasoningMode.GOT in FACTORIAL_MODE_SPECS,
            "gold_available_to_runtime": False,
            "final_answer_rule": "verifier output for every factorial condition",
            "retrieval_batch_rule": (
                "all pending agent/node/branch requests in one scope-safe batch; "
                "shared recall, independent authorization/rerank/fallback/context"
            ),
        },
        "benchmarks": reports,
        "overall": overall,
        "comparisons_by_mode": comparisons_by_mode,
        "mode_comparisons_by_sharing_policy": _mode_comparisons(
            overall,
            reasoning_modes,
            sharing_arms,
        ),
    }


def run_factorial_task(
    example: BenchmarkExample,
    base_settings: CoScopeSettings,
    mode: ReasoningMode,
    arm: SharingArm,
    *,
    threshold: float = 0.75,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Run one independently checkpointable factorial condition."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max output tokens must be positive when set")
    settings = replace(
        base_settings,
        llm=replace(base_settings.llm, max_tokens=max_output_tokens),
        retrieval=replace(
            base_settings.retrieval,
            medoid_threshold=threshold,
            minimum_pairwise_similarity=threshold,
        ),
    )
    return _run_factorial_task(example, settings, mode, arm)


def _run_factorial_task(
    example: BenchmarkExample,
    settings: CoScopeSettings,
    mode: ReasoningMode,
    arm: SharingArm,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = CoScopeRuntime.from_settings(settings)
    run_id = f"factorial_{mode.value}_{arm.value}_{example.example_id}"
    runtime.create_run(example.benchmark, run_id=run_id)

    reasoning_by_agent: dict[str, Any] = {}
    roots: dict[str, ReasoningNode] = {}
    retrieval_nodes: dict[str, list[ReasoningNode]] = {}
    for agent_id, _, _ in AGENTS:
        runtime.register_agent(
            _agent_class_for_arm(agent_id, arm).instantiate(agent_id)
        )
        reasoning, root = runtime.start_reasoning(
            agent_id,
            ReasoningConfig(mode, max_depth=4, branching_factor=3),
        )
        reasoning_by_agent[agent_id] = reasoning
        roots[agent_id] = root
        retrieval_nodes[agent_id] = _expand_retrieval_nodes(
            runtime,
            agent_id,
            reasoning,
            root,
            mode,
        )

    gold_by_agent = _ingest_arm_memories(runtime, run_id, example, arm)
    requests = _make_factorial_requests(
        runtime,
        run_id,
        example,
        mode,
        retrieval_nodes,
    )
    retrieval_started = time.perf_counter()
    results = runtime.retrieve_batch(requests)
    retrieval_seconds = time.perf_counter() - retrieval_started

    requests_by_agent = {
        agent_id: [
            request for request in requests if request.agent_id == agent_id
        ]
        for agent_id, _, _ in AGENTS
    }
    packets_by_agent: dict[str, list[ContextPacket]] = {}
    outputs: dict[str, str] = {}
    outcomes: dict[str, ReasoningOutcome] = {}
    data_flow: dict[str, dict[str, object]] = {}
    per_agent_usage: dict[str, dict[str, int]] = {}
    generation_started = time.perf_counter()
    for agent_id, _, _ in AGENTS:
        agent_packets: list[ContextPacket] = []
        contexts_by_node: dict[str, str] = {}
        for request in requests_by_agent[agent_id]:
            runtime.refresh_context_view(request)
            packet = runtime.assemble_context(
                request,
                results[request.request_id],
            )
            agent_packets.append(packet)
            contexts_by_node[request.reasoning_node_id] = "\n\n".join(
                f"[{entry.source_id}] {entry.content}"
                for entry in packet.all_memories
            )
        packets_by_agent[agent_id] = agent_packets

        usage_before = _llm_usage(runtime)
        events_before = len(runtime.usage.events)
        reasoning = reasoning_by_agent[agent_id]
        root = roots[agent_id]
        nodes = retrieval_nodes[agent_id]
        outcome = execute_live_strategy(
            mode,
            reasoning,
            root,
            _require_llm(runtime),
            question=(
                f"{example.question}\n\nRole objective: "
                f"{ROLE_OBJECTIVES[agent_id]}"
            ),
            authorized_context=contexts_by_node.get(root.node_id, ""),
            answer_instruction=" ".join(_answer_instructions(example.benchmark)),
            tot_branches=nodes if mode == ReasoningMode.TOT else None,
            tot_branch_contexts=(
                contexts_by_node if mode == ReasoningMode.TOT else None
            ),
            got_sources=nodes if mode == ReasoningMode.GOT else None,
            got_source_contexts=(
                contexts_by_node if mode == ReasoningMode.GOT else None
            ),
        )
        runtime.refresh_reasoning_nodes(agent_id, reasoning)
        usage_after = _llm_usage(runtime)
        agent_usage = {
            key: usage_after[key] - usage_before[key]
            for key in usage_after
        }
        cap = settings.llm.max_tokens
        new_llm_events = [
            event
            for event in runtime.usage.events[events_before:]
            if event.category == "llm"
        ]
        agent_usage["cap_hit_calls"] = sum(
            cap is not None and event.completion_tokens >= cap
            for event in new_llm_events
        )
        per_agent_usage[agent_id] = agent_usage
        outputs[agent_id] = outcome.output
        outcomes[agent_id] = outcome
        candidate = extract_benchmark_answer(example.benchmark, outcome.output)
        data_flow[agent_id] = _write_arm_thinking(
            runtime,
            arm,
            agent_id=agent_id,
            reasoning_node_id=root.node_id,
            raw_output=outcome.output,
            candidate=candidate,
        )
    generation_seconds = time.perf_counter() - generation_started

    prediction = extract_benchmark_answer(
        example.benchmark,
        outputs["verifier"],
    )
    scores = score_benchmark_answer(example, prediction)
    flat_packets = [
        packet for packets in packets_by_agent.values() for packet in packets
    ]
    pollution_reports = {
        agent_id: [
            context_pollution(packet, agent_id=agent_id)
            for packet in packets_by_agent[agent_id]
        ]
        for agent_id, _, _ in AGENTS
    }
    gold_by_request = {
        request.request_id: gold_by_agent[request.agent_id]
        for request in requests
    }
    usage = runtime.usage.summary()
    store_queries = (
        runtime.retrieval.stats.shared_store_queries
        + runtime.retrieval.stats.private_store_queries
    )
    selected_items = sum(len(packet.all_memories) for packet in flat_packets)
    polluted_items = sum(
        report.polluted_items
        for reports in pollution_reports.values()
        for report in reports
    )
    return {
        "benchmark": example.benchmark,
        "example_id": example.example_id,
        "reasoning_mode": mode.value,
        "arm": arm.value,
        "prediction": prediction,
        "reference": _reference(example),
        "scores": scores,
        "task_success": bool(scores["success"]),
        "agent_answers": {
            agent_id: extract_benchmark_answer(example.benchmark, output)
            for agent_id, output in outputs.items()
        },
        "reasoning": {
            agent_id: {
                "node_count": outcome.node_count,
                "generated_thoughts": outcome.generated_thoughts,
                "llm_calls": outcome.llm_calls,
                "metadata": outcome.metadata,
            }
            for agent_id, outcome in outcomes.items()
        },
        "per_agent_llm_usage": per_agent_usage,
        "generation_budget": {
            "max_output_tokens": settings.llm.max_tokens,
            "cap_hits": sum(
                usage["cap_hit_calls"] for usage in per_agent_usage.values()
            ),
            "agents_at_cap": [
                agent_id
                for agent_id, agent_usage in per_agent_usage.items()
                if agent_usage["cap_hit_calls"]
            ],
        },
        "provider_usage": usage,
        "retrieval": {
            "requests": len(requests),
            "groups": runtime.retrieval.stats.groups,
            "shared_store_queries": runtime.retrieval.stats.shared_store_queries,
            "private_store_queries": runtime.retrieval.stats.private_store_queries,
            "store_queries": store_queries,
            "query_savings": (
                1.0
                - runtime.retrieval.stats.shared_store_queries / len(requests)
                if runtime.retrieval.stats.shared_store_queries
                else 0.0
            ),
            "shared_recall_savings": (
                1.0
                - runtime.retrieval.stats.shared_store_queries / len(requests)
                if runtime.retrieval.stats.shared_store_queries
                else 0.0
            ),
            "fallback_triggers": runtime.retrieval.stats.fallback_triggers,
            "recall_at_10": recall_at_k(results, gold_by_request, k=10),
            "mrr_at_10": mrr_at_k(results, gold_by_request, k=10),
            "latency_seconds": retrieval_seconds,
            "group_ids": {
                request.request_id: results[request.request_id].group_id
                for request in requests
            },
        },
        "context": {
            "selected_items": selected_items,
            "mean_selected_items": selected_items / len(flat_packets),
            "mean_duplicate_evidence_rate": mean(
                duplicate_evidence_rate(packet) for packet in flat_packets
            ),
            "polluted_items": polluted_items,
            "pollution_rate": (
                polluted_items / selected_items if selected_items else 0.0
            ),
            "private_thinking_exposure": sum(
                report.private_thinking_exposure
                for reports in pollution_reports.values()
                for report in reports
            ),
            "cross_role_knowledge_exposure": sum(
                report.cross_role_knowledge_exposure
                for reports in pollution_reports.values()
                for report in reports
            ),
            "per_agent": {
                agent_id: {
                    "request_count": len(packets_by_agent[agent_id]),
                    "selected_items": sum(
                        len(packet.all_memories)
                        for packet in packets_by_agent[agent_id]
                    ),
                    "polluted_items": sum(
                        report.polluted_items
                        for report in pollution_reports[agent_id]
                    ),
                    "source_ids_by_node": {
                        request.reasoning_node_id: [
                            entry.source_id
                            for entry in packet.all_memories
                        ]
                        for request, packet in zip(
                            requests_by_agent[agent_id],
                            packets_by_agent[agent_id],
                            strict=True,
                        )
                    },
                }
                for agent_id, _, _ in AGENTS
            },
        },
        "safety": {
            "unauthorized_context_exposure": sum(
                entry.scope.scope_id not in request.effective_view.scope_ids
                for request in requests
                for entry in next(
                    packet
                    for packet in packets_by_agent[request.agent_id]
                    if packet.reasoning_node_id == request.reasoning_node_id
                ).all_memories
            )
        },
        "data_flow": data_flow,
        "latency": {
            "generation_seconds": generation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        },
    }


def _expand_retrieval_nodes(
    runtime: CoScopeRuntime,
    agent_id: str,
    reasoning: Any,
    root: ReasoningNode,
    mode: ReasoningMode,
) -> list[ReasoningNode]:
    if mode == ReasoningMode.COT:
        return [root]
    if mode == ReasoningMode.TOT:
        if not isinstance(reasoning, ToTRuntime):
            raise TypeError("ToT mode requires ToTRuntime")
        nodes = reasoning.branch(root.node_id, reasoning.config.branching_factor)
    elif mode == ReasoningMode.GOT:
        if not isinstance(reasoning, GoTRuntime):
            raise TypeError("GoT mode requires GoTRuntime")
        nodes = [
            root,
            reasoning.add_node("evidence_path"),
            reasoning.add_node("countercheck_path"),
        ]
    else:
        raise ValueError(f"unsupported reasoning mode: {mode.value}")
    runtime.refresh_reasoning_nodes(agent_id, reasoning)
    return nodes


def _make_factorial_requests(
    runtime: CoScopeRuntime,
    run_id: str,
    example: BenchmarkExample,
    mode: ReasoningMode,
    nodes_by_agent: dict[str, list[ReasoningNode]],
) -> list[RetrievalRequest]:
    operations = FACTORIAL_MODE_SPECS[mode].retrieval_operations
    requests = []
    for agent_id, _, _ in AGENTS:
        nodes = nodes_by_agent[agent_id]
        if len(nodes) != len(operations):
            raise ValueError("retrieval nodes do not match the mode specification")
        for node, operation in zip(nodes, operations, strict=True):
            requests.append(
                runtime.create_request(
                    run_id=run_id,
                    agent_id=agent_id,
                    reasoning_node_id=node.node_id,
                    full_query=(
                        f"{example.question}\nRole: {agent_id}\n"
                        f"Reasoning operation: {operation}"
                    ),
                    public_intent=(
                        f"{example.question} find public evidence needed to "
                        "plan solve and verify"
                    ),
                    context_budget=8_000,
                    retrieval_budget=10,
                )
            )
    return requests


def _llm_usage(runtime: CoScopeRuntime) -> dict[str, int]:
    summary = runtime.usage.summary()
    categories = cast(dict[str, dict[str, int]], summary["by_category"])
    usage = categories.get("llm", {})
    return {
        key: int(usage.get(key, 0))
        for key in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "total_tokens",
        )
    }


def _require_llm(runtime: CoScopeRuntime) -> Any:
    if runtime.llm is None:
        raise RuntimeError("factorial experiment requires a live LLM")
    return runtime.llm


def _reference(example: BenchmarkExample) -> str:
    if example.benchmark == "math":
        return extract_math_answer(example.reference) or "[invalid]"
    if example.benchmark == "mbpp_plus":
        return "[hidden EvalPlus base and plus tests]"
    return example.reference


def _mode_comparisons(
    overall: dict[str, dict[str, dict[str, Any]]],
    modes: tuple[ReasoningMode, ...],
    arms: tuple[SharingArm, ...],
) -> dict[str, Any]:
    if ReasoningMode.COT not in modes or ReasoningMode.TOT not in modes:
        return {}
    report = {}
    for arm in arms:
        cot = overall[ReasoningMode.COT.value][arm.value]
        tot = overall[ReasoningMode.TOT.value][arm.value]
        report[arm.value] = {
            "tot_minus_cot_answer_f1": tot["answer_f1"] - cot["answer_f1"],
            "tot_minus_cot_task_success": (
                tot["task_success_rate"] - cot["task_success_rate"]
            ),
            "tot_to_cot_llm_token_ratio": (
                tot["llm_total_tokens"] / cot["llm_total_tokens"]
                if cot["llm_total_tokens"]
                else 0.0
            ),
            "tot_to_cot_latency_ratio": (
                tot["mean_end_to_end_seconds"] / cot["mean_end_to_end_seconds"]
                if cot["mean_end_to_end_seconds"]
                else 0.0
            ),
        }
    return report


def _factorial_aggregate(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate(tasks)
    aggregate.update(
        {
            "retrieval_requests": sum(
                int(task["retrieval"]["requests"]) for task in tasks
            ),
            "mean_retrieval_requests": mean(
                int(task["retrieval"]["requests"]) for task in tasks
            ),
            "mean_shared_recall_savings": mean(
                float(task["retrieval"]["shared_recall_savings"])
                for task in tasks
            ),
            "fallback_triggers": sum(
                int(task["retrieval"]["fallback_triggers"])
                for task in tasks
            ),
            "mean_reasoning_nodes_per_agent": mean(
                mean(
                    int(agent["node_count"])
                    for agent in task["reasoning"].values()
                )
                for task in tasks
            ),
            "mean_generated_thoughts_per_agent": mean(
                mean(
                    int(agent["generated_thoughts"])
                    for agent in task["reasoning"].values()
                )
                for task in tasks
            ),
        }
    )
    return aggregate
