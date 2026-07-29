"""Safe extraction and official EvalPlus scoring for code benchmarks."""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.task_metrics import INVALID_ANSWER

FINAL_CODE_MARKER = "FINAL_CODE:"
DEFAULT_EVALPLUS_IMAGE = (
    "ganler/evalplus@sha256:"
    "26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740"
)


@dataclass(frozen=True)
class CodeScore:
    base_pass: bool
    plus_pass: bool

    def metrics(self) -> dict[str, float]:
        base = float(self.base_pass)
        plus = float(self.base_pass and self.plus_pass)
        return {
            "em": plus,
            "f1": plus,
            "accuracy": plus,
            "success": plus,
            "base_pass": base,
            "plus_pass": plus,
        }


class CodeEvaluator(Protocol):
    def evaluate(
        self,
        examples: list[BenchmarkExample],
        predictions: dict[str, str],
        *,
        label: str,
    ) -> dict[str, CodeScore]:
        """Return an official base/plus score for every example ID."""


class EvalPlusDockerEvaluator:
    """Run generated Python only inside an isolated official EvalPlus image."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        artifact_root: str | Path = "new_results/evalplus",
        image: str = DEFAULT_EVALPLUS_IMAGE,
        parallel: int = 2,
        timeout_seconds: int = 3_600,
    ):
        self.dataset_path = Path(dataset_path).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.image = image
        self.parallel = parallel
        self.timeout_seconds = timeout_seconds
        if parallel <= 0:
            raise ValueError("EvalPlus parallelism must be positive")
        if timeout_seconds <= 0:
            raise ValueError("EvalPlus timeout must be positive")

    def evaluate(
        self,
        examples: list[BenchmarkExample],
        predictions: dict[str, str],
        *,
        label: str,
    ) -> dict[str, CodeScore]:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is required for MBPP-Plus scoring; generated code "
                "will not be executed on the host"
            )
        rows = _load_evalplus_rows(self.dataset_path)
        selected = []
        samples = []
        for example in examples:
            task_id = example.metadata.get("task_id")
            if not task_id or task_id not in rows:
                raise ValueError(
                    f"missing EvalPlus task for {example.example_id}"
                )
            selected.append(rows[task_id])
            solution = predictions.get(example.example_id, INVALID_ANSWER)
            if solution == INVALID_ANSWER:
                solution = "def __coscope_invalid_solution__():\n    pass\n"
            samples.append({"task_id": task_id, "solution": solution})

        self.artifact_root.mkdir(parents=True, exist_ok=True)
        prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("_")
        run_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{prefix or 'mbpp_plus'}_",
                dir=self.artifact_root,
            )
        )
        subset_path = run_dir / "MbppPlus-subset.jsonl.gz"
        samples_path = run_dir / "samples.jsonl"
        with gzip.open(subset_path, "wt", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        samples_path.write_text(
            "".join(
                json.dumps(sample, ensure_ascii=False) + "\n"
                for sample in samples
            ),
            encoding="utf-8",
        )

        command = self._docker_command(run_dir)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=_docker_environment(),
        )
        (run_dir / "docker.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (run_dir / "docker.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"EvalPlus container failed with exit code "
                f"{completed.returncode}; see {run_dir}"
            )
        result_candidates = (
            run_dir / "samples_eval_results.json",
            run_dir / "samples.eval_results.json",
        )
        result_path = next(
            (path for path in result_candidates if path.exists()),
            None,
        )
        if result_path is None:
            raise RuntimeError(
                f"EvalPlus did not write an evaluation result; see {run_dir}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        evaluated = payload.get("eval", {})
        scores = {}
        for example in examples:
            task_id = example.metadata["task_id"]
            task_results = evaluated.get(task_id, [])
            if len(task_results) != 1:
                raise RuntimeError(
                    f"EvalPlus returned {len(task_results)} results for "
                    f"{task_id}"
                )
            result = task_results[0]
            scores[example.example_id] = CodeScore(
                base_pass=result.get("base_status") == "pass",
                plus_pass=result.get("plus_status") == "pass",
            )
        return scores

    def _docker_command(self, run_dir: Path) -> list[str]:
        return [
            "docker",
            "run",
            "--platform=linux/amd64",
            "--rm",
            "--pull=missing",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=512",
            "--memory=4g",
            "--cpus=2",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=1g",
            "-e",
            "MBPP_OVERRIDE_PATH=/work/MbppPlus-subset.jsonl.gz",
            "-e",
            "XDG_CACHE_HOME=/work/cache",
            "-v",
            f"{run_dir}:/work",
            "-w",
            "/work",
            self.image,
            "evalplus.evaluate",
            "--dataset",
            "mbpp",
            "--samples",
            "/work/samples.jsonl",
            "--parallel",
            str(self.parallel),
        ]


def extract_python_solution(output: str) -> str:
    """Extract a final self-contained Python program without executing it."""
    marker = output.casefold().rfind(FINAL_CODE_MARKER.casefold())
    if marker < 0:
        return INVALID_ANSWER
    candidate = output[marker + len(FINAL_CODE_MARKER) :].strip()
    if candidate.startswith("```"):
        first_line, separator, remainder = candidate.partition("\n")
        if not separator or first_line.casefold() not in {"```", "```python"}:
            return INVALID_ANSWER
        closing = remainder.rfind("```")
        if closing < 0 or remainder[closing + 3 :].strip():
            return INVALID_ANSWER
        candidate = remainder[:closing].strip()
    if not candidate:
        return INVALID_ANSWER
    try:
        ast.parse(candidate)
    except SyntaxError:
        return INVALID_ANSWER
    return candidate.rstrip() + "\n"


def apply_mbpp_plus_scores(
    tasks: list[dict[str, Any]],
    examples: list[BenchmarkExample],
    evaluator: CodeEvaluator,
    *,
    label: str,
) -> None:
    predictions = {
        str(task["example_id"]): str(task["prediction"]) for task in tasks
    }
    scores = evaluator.evaluate(examples, predictions, label=label)
    for task in tasks:
        score = scores[str(task["example_id"])].metrics()
        task["scores"] = score
        task["task_success"] = bool(score["success"])
        task["external_evaluation"] = {
            "evaluator": "EvalPlus",
            "base_pass": bool(score["base_pass"]),
            "plus_pass": bool(score["plus_pass"]),
        }


def pending_code_score() -> dict[str, float]:
    return {
        "em": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "success": 0.0,
        "base_pass": 0.0,
        "plus_pass": 0.0,
    }


def _load_evalplus_rows(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {str(row["task_id"]): row for row in rows}


def _docker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    helper_dir = Path(
        "/Applications/Docker.app/Contents/Resources/bin"
    )
    if helper_dir.is_dir():
        current_path = environment.get("PATH", "")
        environment["PATH"] = f"{helper_dir}{os.pathsep}{current_path}"
    return environment
