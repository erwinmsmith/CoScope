from __future__ import annotations

from coscope import CoScopeRuntime
from coscope.adapters.llm import LLMOutput
from coscope.config import CoScopeSettings
from coscope.evaluation.benchmarks import BenchmarkContext, BenchmarkExample
from coscope.evaluation.factorial_experiment import (
    DEFAULT_BATCH_MODES,
    DEFAULT_FACTORIAL_MODES,
    DEFAULT_SHARING_ARMS,
    FACTORIAL_MODE_SPECS,
    BatchMode,
    parse_reasoning_modes,
    run_factorial_experiment,
    run_factorial_task,
)
from coscope.evaluation.sharing_ablation import SharingArm
from coscope.memory import QdrantMemoryStore
from coscope.reasoning import ReasoningMode


class _RepeatedLLM:
    model_version = "fake"

    def __init__(self, usage):
        self.usage = usage

    def invoke(self, messages):
        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "reasoning_tokens": 2,
            "cached_tokens": 0,
            "total_tokens": 15,
        }
        self.usage.record("llm", self.model_version, token_usage)
        return LLMOutput(
            "PUBLIC_SUMMARY: answer\n"
            "SELECTED_BRANCH: 1\n"
            "FINAL_ANSWER: answer",
            usage=token_usage,
        )


def _runtime_with_repeated_llm(
    settings: CoScopeSettings | None = None,
) -> CoScopeRuntime:
    runtime = CoScopeRuntime(settings=settings)
    runtime.llm = _RepeatedLLM(runtime.usage)
    return runtime


def test_factorial_defaults_use_uniform_cot_and_tot_with_three_sharing_arms(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        CoScopeRuntime,
        "from_settings",
        classmethod(
            lambda cls, settings, **kwargs: _runtime_with_repeated_llm(
                settings
            )
        ),
    )
    example = BenchmarkExample(
        "example",
        "question",
        "answer",
        benchmark="hotpotqa",
        context=(
            BenchmarkContext("evidence_a", "first evidence"),
            BenchmarkContext("evidence_b", "second evidence"),
        ),
        supporting_source_ids=frozenset({"evidence_a"}),
    )

    report = run_factorial_experiment(
        {"hotpotqa": [example]},
        CoScopeSettings.from_env(None, environ={}),
        threshold=0.0,
        bootstrap_samples=20,
    )

    assert report["design"]["reasoning_modes"] == ["cot", "tot"]
    assert report["design"]["sharing_policies"] == [
        "coscope",
        "full_sharing",
        "no_sharing",
    ]
    assert report["design"]["batch_modes"] == ["batched", "independent"]
    for mode in DEFAULT_FACTORIAL_MODES:
        for arm in DEFAULT_SHARING_ARMS:
            for batch in DEFAULT_BATCH_MODES:
                condition = report["benchmarks"]["hotpotqa"][mode.value][
                    arm.value
                ][batch.value]
                assert condition["aggregate"]["tasks"] == 1
                assert condition["tasks"][0]["task_success"]

    cot = report["benchmarks"]["hotpotqa"]["cot"]["coscope"]["batched"][
        "tasks"
    ][0]
    assert cot["retrieval"]["requests"] == 3
    assert cot["retrieval"]["shared_store_queries"] == 1
    assert {
        agent: metrics["llm_calls"]
        for agent, metrics in cot["reasoning"].items()
    } == {"planner": 1, "solver": 1, "verifier": 1}
    assert cot["latency"]["embedding_seconds"] == 0.0
    assert (
        cot["latency"]["end_to_end_seconds"]
        <= cot["latency"]["wall_clock_seconds"]
    )

    tot = report["benchmarks"]["hotpotqa"]["tot"]["coscope"]["batched"][
        "tasks"
    ][0]
    assert tot["retrieval"]["requests"] == 9
    assert tot["retrieval"]["groups"] == 1
    assert tot["retrieval"]["shared_store_queries"] == 1
    assert abs(tot["retrieval"]["query_savings"] - 8 / 9) < 1e-12
    assert {
        agent: metrics["llm_calls"]
        for agent, metrics in tot["reasoning"].items()
    } == {"planner": 4, "solver": 4, "verifier": 4}

    isolated = report["benchmarks"]["hotpotqa"]["tot"]["no_sharing"][
        "batched"
    ]["tasks"][0]
    assert isolated["retrieval"]["shared_store_queries"] == 0
    assert isolated["retrieval"]["private_store_queries"] == 9

    full_independent = report["benchmarks"]["hotpotqa"]["cot"][
        "full_sharing"
    ][BatchMode.INDEPENDENT.value]["tasks"][0]
    assert full_independent["retrieval"]["groups"] == 3
    assert full_independent["retrieval"]["shared_store_queries"] == 3
    assert full_independent["retrieval"]["query_reduction_rate"] == 0.0

    scoped_independent = report["benchmarks"]["hotpotqa"]["tot"]["coscope"][
        BatchMode.INDEPENDENT.value
    ]["tasks"][0]
    assert scoped_independent["retrieval"]["groups"] == 9
    assert scoped_independent["retrieval"]["shared_store_queries"] == 9


def test_got_remains_available_in_the_same_mode_registry() -> None:
    assert parse_reasoning_modes("got") == (ReasoningMode.GOT,)
    assert not FACTORIAL_MODE_SPECS[ReasoningMode.GOT].default_enabled
    assert FACTORIAL_MODE_SPECS[ReasoningMode.GOT].llm_calls_per_agent == 4


def test_factorial_task_routes_all_memory_through_qdrant_and_cleans_namespace(
    monkeypatch,
) -> None:
    original = CoScopeRuntime.from_settings.__func__
    runtimes: list[CoScopeRuntime] = []

    def build_runtime(cls, settings, **kwargs):
        runtime = original(cls, settings, **kwargs)
        runtime.llm = _RepeatedLLM(runtime.usage)
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(
        CoScopeRuntime,
        "from_settings",
        classmethod(build_runtime),
    )
    settings = CoScopeSettings.from_env(
        None,
        environ={
            "COSCOPE_MEMORY_PROVIDER": "qdrant",
            "COSCOPE_QDRANT_URL": ":memory:",
            "COSCOPE_QDRANT_COLLECTION": "factorial_test_memory",
        },
    )
    example = BenchmarkExample(
        "qdrant-example",
        "question",
        "answer",
        benchmark="hotpotqa",
        context=(BenchmarkContext("evidence", "evidence"),),
        supporting_source_ids=frozenset({"evidence"}),
    )

    result = run_factorial_task(
        example,
        settings,
        ReasoningMode.COT,
        SharingArm.COSCOPE,
        BatchMode.BATCHED,
        threshold=0.0,
    )

    assert result["retrieval"]["memory_store_backend"] == "qdrant"
    assert result["retrieval"]["shared_store_queries"] == 1
    assert result["retrieval"]["vector_store_query_seconds"] > 0
    assert isinstance(runtimes[0].memory, QdrantMemoryStore)
    assert len(runtimes[0].memory) == 0
