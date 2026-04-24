"""GSM8K loader (math reasoning, 4-6 step)."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from coscope.utils.loaders.base_loader import BaseLoader


class GSM8KLoader(BaseLoader):
    """Loader for GSM8K (grade-school math word problems)."""

    dataset_name = "gsm8k"
    dataset_type = "math"

    DEFAULT_TRAIN = "train.parquet"
    DEFAULT_TEST = "test.parquet"

    _NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
    _FINAL_RE = re.compile(r"####\s*([\-0-9\.,]+)")

    def load(self) -> List[Dict[str, Any]]:
        # GSM8K has no official dev; dev split is produced by SplitManager.
        # For "dev" we load from the training pool and let the pipeline filter
        # by dev_ids, same pattern used in the legacy loader.
        default_dev = self.dataset_config.get("train_file", self.DEFAULT_TRAIN)
        path = self._resolve_split_file(self.DEFAULT_TRAIN, default_dev, self.DEFAULT_TEST)
        raw = self._load_any(path)
        return [self._normalize(row, idx) for idx, row in enumerate(raw)]

    # ------------------------------------------------------------------

    def _normalize(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        item = self._empty_raw_item()
        item["original_id"] = f"gsm8k_{self.split}_{idx:06d}"
        question = row.get("question", "") or ""
        answer_field = row.get("answer", "") or ""
        item["question"] = question

        # Final answer: "#### <value>"
        final_match = self._FINAL_RE.search(answer_field)
        item["answer"] = final_match.group(1).strip() if final_match else answer_field.strip()

        # Solution steps: split answer by newline, drop the "#### ..." line
        body = answer_field.split("####", 1)[0] if "####" in answer_field else answer_field
        steps = [ln.strip() for ln in body.split("\n") if ln.strip()]
        item["solution_steps"] = steps
        item["hop_count"] = max(2, min(len(steps), 9))

        # Workspace (semantic): known numeric values extracted from the question.
        # v2.1: tag each with the *first* solution_step it appears in to drive
        # hop-level workspace partitioning; otherwise leave hop_index=None.
        numbers = self._NUMBER_RE.findall(question)
        supporting: List[Dict[str, Any]] = []
        if numbers:
            for p_idx, num in enumerate(numbers):
                hop_for_num = self._find_first_step_with(steps, num)
                supporting.append(
                    {
                        "paragraph_id": f"para_{p_idx:03d}",
                        "text": f"Numeric value: {num}",
                        "title": f"number_{p_idx}",
                        "is_gold": True,
                        "hop_index": hop_for_num,
                    }
                )
        else:
            supporting.append(
                {
                    "paragraph_id": "para_000",
                    "text": question,
                    "title": "question",
                    "is_gold": True,
                    "hop_index": None,
                }
            )
        # Additionally, emit per-step "step formula" entries with hop_index=k
        # so each Solver-k has its own hop-workspace content (parallels the
        # QA loaders where gold paragraphs are hop-partitioned).
        base_offset = len(supporting)
        for k, step in enumerate(steps[: item["hop_count"]], start=1):
            supporting.append(
                {
                    "paragraph_id": f"para_step_{k:03d}",
                    "text": f"Step-{k} rationale: {step}",
                    "title": f"step_{k}",
                    "is_gold": True,
                    "hop_index": k,
                }
            )
        item["supporting_paragraphs"] = supporting
        item["distractor_paragraphs"] = []

        # sub_questions: one per step
        item["sub_questions"] = []  # Math: use solution_steps driven queries downstream

        # task_shared_items: oracle pre-write of gold step conclusions
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

    @staticmethod
    def _find_first_step_with(steps: List[str], number: str) -> int | None:
        for k, step in enumerate(steps, start=1):
            if number in step:
                return k
        return None
