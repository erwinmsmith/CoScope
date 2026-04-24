"""MATH loader (competition math, 5-9 step, per-category)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from coscope.utils.loaders.base_loader import BaseLoader


class MathLoader(BaseLoader):
    """Loader for the MATH dataset."""

    dataset_name = "math"
    dataset_type = "math"

    DEFAULT_TRAIN = "train.parquet"
    DEFAULT_TEST = "test.parquet"

    _FORMULA_RE = re.compile(r"\$[^$]+\$")
    _BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
    _SENT_RE = re.compile(r"\.\s+")

    def load(self) -> List[Dict[str, Any]]:
        default_dev = self.dataset_config.get("train_file", self.DEFAULT_TRAIN)
        path = self._resolve_split_file(self.DEFAULT_TRAIN, default_dev, self.DEFAULT_TEST)
        raw = self._load_any(path)
        return [self._normalize(row, idx) for idx, row in enumerate(raw)]

    # ------------------------------------------------------------------

    def _normalize(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        item = self._empty_raw_item()
        problem = row.get("problem", "") or row.get("question", "") or ""
        solution = row.get("solution", "") or ""
        item["question"] = problem
        category = self._infer_category(row)
        item["math_category"] = category
        # original_id: prefer explicit id, then file-based hint, fall back to index.
        if row.get("id"):
            item["original_id"] = str(row["id"])
        elif category:
            item["original_id"] = f"{category}_{idx:06d}"
        else:
            item["original_id"] = f"math_{self.split}_{idx:06d}"

        # Final answer -> \boxed{...}
        boxed = self._BOXED_RE.search(solution)
        item["answer"] = boxed.group(1).strip() if boxed else solution[-80:].strip()

        # Solution steps
        steps = self._split_steps(solution)
        item["solution_steps"] = steps
        item["hop_count"] = max(2, min(len(steps), 9))

        # Workspace: LaTeX formulas in the problem + per-step formulas.
        # v2.1: tag each entry with its hop_index so Solver-k gets its own
        # hop-level workspace content.
        formulas = self._FORMULA_RE.findall(problem)
        supporting: List[Dict[str, Any]] = []
        if formulas:
            for p_idx, f in enumerate(formulas):
                hop_for_formula = self._find_first_step_with(steps, f)
                supporting.append(
                    {
                        "paragraph_id": f"para_{p_idx:03d}",
                        "text": f,
                        "title": f"formula_{p_idx}",
                        "is_gold": True,
                        "hop_index": hop_for_formula,
                    }
                )
        else:
            supporting.append(
                {
                    "paragraph_id": "para_000",
                    "text": problem,
                    "title": "problem",
                    "is_gold": True,
                    "hop_index": None,
                }
            )
        # Per-step formula entries (one per hop) for hop-workspace differentiation.
        for k, step in enumerate(steps[: item["hop_count"]], start=1):
            step_formulas = self._FORMULA_RE.findall(step)
            text = " ; ".join(step_formulas) if step_formulas else step
            supporting.append(
                {
                    "paragraph_id": f"para_step_{k:03d}",
                    "text": f"Step-{k}: {text}",
                    "title": f"step_{k}",
                    "is_gold": True,
                    "hop_index": k,
                }
            )
        item["supporting_paragraphs"] = supporting
        item["distractor_paragraphs"] = []

        item["sub_questions"] = []  # Math: driven by solution_steps

        task_shared: List[Dict[str, Any]] = []
        for k, step in enumerate(steps[: item["hop_count"]], start=1):
            task_shared.append(
                {
                    "hop_index": k,
                    "text": step,
                    "source_paragraph_id": f"step_{k:03d}",
                }
            )
        item["task_shared_items"] = task_shared
        return item

    # ------------------------------------------------------------------

    def _split_steps(self, solution: str) -> List[str]:
        lines = [ln.strip() for ln in solution.split("\n") if ln.strip() and len(ln.strip()) >= 10]
        if len(lines) >= 2:
            return lines
        # Fallback: split on sentence boundaries
        sentences = [s.strip() for s in self._SENT_RE.split(solution) if s.strip()]
        return sentences or ([solution] if solution else [])

    def _infer_category(self, row: Dict[str, Any]) -> Optional[str]:
        # Try common fields
        for key in ("type", "category", "subject", "level"):
            val = row.get(key)
            if isinstance(val, str) and val:
                return self._slug(val)
        # Try inferring from `file` or `id` if present
        file_hint = row.get("file") or row.get("path")
        if isinstance(file_hint, str):
            try:
                return self._slug(Path(file_hint).parts[-2])
            except IndexError:
                return None
        return None

    @staticmethod
    def _slug(raw: str) -> str:
        return re.sub(r"\s+", "_", raw.strip().lower())

    @staticmethod
    def _find_first_step_with(steps: List[str], token: str) -> Optional[int]:
        for k, step in enumerate(steps, start=1):
            if token in step:
                return k
        return None
