from coscope import CoScopeRuntime
from coscope.adapters.llm import LLMOutput
from coscope.config import CoScopeSettings
from coscope.evaluation.benchmarks import (
    BenchmarkContext,
    BenchmarkExample,
    load_2wikimultihopqa,
    load_aime2024,
    load_aime2025,
    load_hotpotqa,
    load_math,
    load_mbpp_plus,
    load_musique,
)
from coscope.evaluation.math_grader import (
    extract_math_answer,
    grade_math_answer,
)
from coscope.evaluation.mode_experiment import _run_mode_task
from coscope.evaluation.task_metrics import extract_final_answer, qa_answer_metrics
from coscope.reasoning import (
    CoTRuntime,
    GoTRuntime,
    ReasoningConfig,
    ReasoningMode,
    ToTRuntime,
    execute_live_strategy,
)


class _SequenceLLM:
    model_version = "fake"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return LLMOutput(next(self.outputs))


def test_all_local_benchmark_loaders_preserve_gold_outside_context():
    loaders = [
        (
            load_hotpotqa,
            "raw/hotpotqa/distractor_validation.parquet",
            "hotpotqa",
        ),
        (load_2wikimultihopqa, "raw/2wikimhqa/dev.json", "2wikimultihopqa"),
        (
            load_musique,
            "raw/musique/musique_ans_v1.0_dev.jsonl",
            "musique",
        ),
        (load_math, "raw/math/test.parquet", "math"),
    ]
    for loader, path, benchmark in loaders:
        example = loader(path, limit=1, seed=1)[0]
        assert example.benchmark == benchmark
        assert example.question
        assert example.reference
        assert "reference" not in example.metadata

    for loader, benchmark in (
        (load_aime2024, "aime2024"),
        (load_aime2025, "aime2025"),
    ):
        example = loader("raw/aime", limit=1, seed=1)[0]
        assert example.benchmark == benchmark
        assert example.reference.isdigit()

    code = load_mbpp_plus(
        "raw/mbpp_plus/MbppPlus.jsonl.gz",
        limit=1,
        seed=1,
    )[0]
    assert code.benchmark == "mbpp_plus"
    assert code.reference == "[hidden EvalPlus base and plus tests]"
    assert set(code.metadata) == {"task_id", "entry_point", "dataset_path"}
    assert "canonical_solution" not in code.question


def test_official_qa_alias_and_math_equivalence_scoring():
    assert extract_final_answer("reason\nFINAL_ANSWER: Eiffel Tower") == "Eiffel Tower"
    metrics = qa_answer_metrics("NYC", ("New York City", "NYC"))
    assert metrics == {"em": 1.0, "f1": 1.0}
    assert extract_math_answer(r"work \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert grade_math_answer("0.5", r"solution \boxed{\frac{1}{2}}")


def test_cot_executes_one_linear_thought():
    runtime = CoTRuntime("agent", ReasoningConfig(ReasoningMode.COT))
    root = runtime.create_root()
    llm = _SequenceLLM(["FINAL_ANSWER: answer"])
    outcome = execute_live_strategy(
        ReasoningMode.COT,
        runtime,
        root,
        llm,
        question="question",
        authorized_context="context",
        answer_instruction="FINAL_ANSWER",
    )
    assert outcome.llm_calls == 1
    assert outcome.node_count == 1
    assert llm.calls == 1


def test_tot_executes_generation_evaluation_and_pruning():
    runtime = ToTRuntime(
        "agent",
        ReasoningConfig(ReasoningMode.TOT, branching_factor=3),
    )
    root = runtime.create_root()
    llm = _SequenceLLM(
        [
            "candidate 1",
            "candidate 2",
            "candidate 3",
            "SELECTED_BRANCH: 2\nFINAL_ANSWER: answer",
        ]
    )
    outcome = execute_live_strategy(
        ReasoningMode.TOT,
        runtime,
        root,
        llm,
        question="question",
        authorized_context="context",
        answer_instruction="FINAL_ANSWER",
    )
    assert outcome.llm_calls == 4
    assert outcome.node_count == 4
    assert len(outcome.metadata["pruned_node_ids"]) == 2


def test_tot_batches_three_branch_retrieval_requests(monkeypatch):
    llm = _SequenceLLM(
        [
            "candidate 1",
            "candidate 2",
            "candidate 3",
            "SELECTED_BRANCH: 2\nFINAL_ANSWER: answer",
        ]
    )
    runtime = CoScopeRuntime(llm=llm)
    monkeypatch.setattr(
        CoScopeRuntime,
        "from_settings",
        classmethod(lambda cls, settings: runtime),
    )
    settings = CoScopeSettings.from_env(None, environ={})
    example = BenchmarkExample(
        "example",
        "question",
        "answer",
        benchmark="hotpotqa",
        context=(
            BenchmarkContext("evidence_a", "first evidence"),
            BenchmarkContext("evidence_b", "second evidence"),
        ),
    )

    task = _run_mode_task(example, ReasoningMode.TOT, settings)

    assert task["retrieval"]["requests"] == 3
    assert task["retrieval"]["groups"] == 1
    assert task["retrieval"]["shared_store_queries"] == 1
    assert abs(task["retrieval"]["query_savings"] - 2 / 3) < 1e-12
    assert len(task["retrieval"]["public_queries"]) == 3
    assert len(set(task["retrieval"]["public_queries"].values())) == 1
    assert len(task["retrieval"]["context_source_ids"]) == 3
    assert task["reasoning"]["llm_calls"] == 4
    assert task["safety"]["unauthorized_context_exposure"] == 0


def test_got_executes_three_sources_and_multi_parent_aggregate():
    runtime = GoTRuntime("agent", ReasoningConfig(ReasoningMode.GOT))
    root = runtime.add_node("root")
    llm = _SequenceLLM(
        ["source 1", "source 2", "source 3", "FINAL_ANSWER: answer"]
    )
    outcome = execute_live_strategy(
        ReasoningMode.GOT,
        runtime,
        root,
        llm,
        question="question",
        authorized_context="context",
        answer_instruction="FINAL_ANSWER",
    )
    aggregate = next(node for node in runtime.nodes() if node.node_id == "aggregate")
    assert outcome.llm_calls == 4
    assert outcome.node_count == 4
    assert len(aggregate.parent_ids) == 3
