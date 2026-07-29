from pathlib import Path

from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.code_benchmark import (
    CodeScore,
    EvalPlusDockerEvaluator,
    apply_mbpp_plus_scores,
    extract_python_solution,
)
from coscope.evaluation.task_metrics import INVALID_ANSWER


class _FakeCodeEvaluator:
    def evaluate(self, examples, predictions, *, label):
        assert label == "test"
        assert predictions == {"mbpp_2": "def answer():\n    return 1\n"}
        return {"mbpp_2": CodeScore(base_pass=True, plus_pass=False)}


def test_python_solution_extraction_is_strict_and_syntax_checked():
    output = (
        "PUBLIC_SUMMARY: defines answer\n"
        "FINAL_CODE:\n"
        "```python\n"
        "def answer():\n"
        "    return 1\n"
        "```"
    )
    assert extract_python_solution(output) == "def answer():\n    return 1\n"
    assert extract_python_solution("def answer(): return 1") == INVALID_ANSWER
    assert (
        extract_python_solution("FINAL_CODE:\n```python\ndef broken(\n```")
        == INVALID_ANSWER
    )


def test_evalplus_command_has_host_isolation_boundaries(tmp_path):
    evaluator = EvalPlusDockerEvaluator(
        tmp_path / "MbppPlus.jsonl.gz",
        artifact_root=tmp_path,
    )
    command = evaluator._docker_command(Path("/tmp/evalplus-run"))
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "evalplus.evaluate" in command


def test_external_code_scores_use_plus_tests_for_task_success():
    example = BenchmarkExample(
        "mbpp_2",
        "Write a function.",
        "[hidden]",
        benchmark="mbpp_plus",
        metadata={"task_id": "Mbpp/2"},
    )
    tasks = [
        {
            "example_id": "mbpp_2",
            "prediction": "def answer():\n    return 1\n",
        }
    ]
    apply_mbpp_plus_scores(
        tasks,
        [example],
        _FakeCodeEvaluator(),
        label="test",
    )
    assert tasks[0]["scores"]["base_pass"] == 1.0
    assert tasks[0]["scores"]["plus_pass"] == 0.0
    assert not tasks[0]["task_success"]
