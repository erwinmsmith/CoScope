from pathlib import Path

import pytest

from coscope import CoScopeRuntime
from coscope.evaluation.benchmarks import BenchmarkContext, BenchmarkExample
from coscope.evaluation.context_metrics import context_pollution
from coscope.evaluation.mas_experiment import AGENTS
from coscope.evaluation.sharing_ablation import (
    SharingArm,
    _agent_class_for_arm,
    _ensure_per_agent_llm_usage,
    _generation_budget,
    _ingest_arm_memories,
    _make_requests,
    _write_arm_thinking,
)
from coscope.reasoning import ReasoningConfig
from coscope.scripts.merge_sharing_ablation import _append_disjoint_shard


def _exercise_arm(arm: SharingArm):
    runtime = CoScopeRuntime()
    runtime.create_run("qa", run_id=f"run_{arm.value}")
    nodes = {}
    for agent_id, mode, _ in AGENTS:
        runtime.register_agent(
            _agent_class_for_arm(agent_id, arm).instantiate(agent_id)
        )
        _, node = runtime.start_reasoning(agent_id, ReasoningConfig(mode))
        nodes[agent_id] = node
    example = BenchmarkExample(
        "example",
        "question",
        "answer",
        benchmark="hotpotqa",
        context=(
            BenchmarkContext("source_a", "evidence a"),
            BenchmarkContext("source_b", "evidence b"),
        ),
        supporting_source_ids=frozenset({"source_a"}),
    )
    _ingest_arm_memories(runtime, f"run_{arm.value}", example, arm)
    requests = _make_requests(
        runtime,
        f"run_{arm.value}",
        nodes,
        example,
        arm,
    )
    results = runtime.retrieve_batch(requests)
    reports = {}
    for (agent_id, _, _), request in zip(AGENTS, requests, strict=True):
        runtime.refresh_context_view(request)
        packet = runtime.assemble_context(
            request,
            results[request.request_id],
        )
        reports[agent_id] = context_pollution(packet, agent_id=agent_id)
        _write_arm_thinking(
            runtime,
            arm,
            agent_id=agent_id,
            reasoning_node_id=nodes[agent_id].node_id,
            raw_output=(
                f"private working text from {agent_id}\n"
                f"PUBLIC_SUMMARY: summary from {agent_id}\n"
                "FINAL_ANSWER: answer"
            ),
            candidate="answer",
        )
    return runtime, reports


def test_full_sharing_exposes_private_thinking_and_cross_role_knowledge():
    runtime, reports = _exercise_arm(SharingArm.FULL_SHARING)
    assert reports["planner"].cross_role_knowledge_exposure == 2
    assert reports["solver"].private_thinking_exposure == 1
    assert reports["verifier"].private_thinking_exposure == 2
    assert all(
        entry.vector is None
        for entry in runtime.memory
        if entry.source_type == "full_shared_thinking"
    )


def test_coscope_and_no_sharing_have_zero_pollution_but_different_reuse():
    scoped_runtime, scoped_reports = _exercise_arm(SharingArm.COSCOPE)
    isolated_runtime, isolated_reports = _exercise_arm(SharingArm.NO_SHARING)
    assert all(report.polluted_items == 0 for report in scoped_reports.values())
    assert all(report.polluted_items == 0 for report in isolated_reports.values())
    assert isolated_runtime.retrieval.stats.shared_store_queries == 0
    assert isolated_runtime.retrieval.stats.private_store_queries == 3
    assert scoped_runtime.retrieval.stats.shared_store_queries >= 1
    assert scoped_runtime.retrieval.stats.private_store_queries == 0


def test_ablation_shards_append_only_when_example_ids_are_disjoint():
    base = _minimal_report("example_a")
    _append_disjoint_shard(
        base,
        _minimal_report("example_b"),
        source=Path("shard.json"),
    )
    assert len(base["benchmarks"]["math"]["coscope"]["tasks"]) == 2

    with pytest.raises(ValueError, match="repeats math/coscope"):
        _append_disjoint_shard(
            base,
            _minimal_report("example_b"),
            source=Path("duplicate.json"),
        )


def test_older_report_backfills_per_agent_usage_from_ordered_events():
    task = {
        "provider_usage": {
            "events": [
                {
                    "category": "llm",
                    "prompt_tokens": index,
                    "completion_tokens": 2,
                    "reasoning_tokens": 1,
                    "cached_tokens": 0,
                    "total_tokens": index + 2,
                }
                for index in (10, 20, 30)
            ]
        }
    }

    _ensure_per_agent_llm_usage(task)

    assert task["per_agent_llm_usage"]["planner"]["prompt_tokens"] == 10
    assert task["per_agent_llm_usage"]["solver"]["prompt_tokens"] == 20
    assert task["per_agent_llm_usage"]["verifier"]["prompt_tokens"] == 30


def test_generation_budget_records_agents_that_hit_the_cap():
    budget = _generation_budget(
        {
            "planner": {"completion_tokens": 100},
            "solver": {"completion_tokens": 99},
            "verifier": {"completion_tokens": 100},
        },
        max_output_tokens=100,
    )

    assert budget == {
        "max_output_tokens": 100,
        "agents_at_cap": ["planner", "verifier"],
        "cap_hits": 2,
    }


def _minimal_report(example_id: str):
    return {
        "benchmarks": {
            "math": {
                arm.value: {
                    "tasks": [{"example_id": example_id}],
                }
                for arm in SharingArm
            }
        }
    }
