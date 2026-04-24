"""MuSiQue loader (multi-hop QA, 2/3/4 hop)."""

from __future__ import annotations

from typing import Any, Dict, List

from coscope.utils.loaders.base_loader import BaseLoader


class MusiqueLoader(BaseLoader):
    """Loader for the MuSiQue-Ans dataset."""

    dataset_name = "musique"
    dataset_type = "qa"

    DEFAULT_TRAIN = "musique_ans_v1.0_train.jsonl"
    DEFAULT_DEV = "musique_ans_v1.0_dev.jsonl"

    def load(self) -> List[Dict[str, Any]]:
        path = self._resolve_split_file(self.DEFAULT_TRAIN, self.DEFAULT_DEV)
        raw = self._load_any(path)
        items: List[Dict[str, Any]] = []
        for idx, row in enumerate(raw):
            # MuSiQue-Ans: only keep answerable questions.
            if row.get("answerable") is False:
                continue
            items.append(self._normalize(row, idx))
        return items

    # ------------------------------------------------------------------

    def _normalize(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        item = self._empty_raw_item()
        raw_id = row.get("id") or f"idx_{idx:06d}"
        item["original_id"] = str(raw_id)
        item["question"] = row.get("question", "") or ""
        item["answer"] = str(row.get("answer", ""))

        # v2.1: derive per-paragraph hop_index from question_decomposition first,
        # so we can tag supporting paragraphs with the hop they belong to.
        decomp = row.get("question_decomposition", []) or []
        para_idx_to_hop: Dict[int, int] = {}
        for k, d in enumerate(decomp, start=1):
            para_idx = d.get("paragraph_support_idx")
            if isinstance(para_idx, int):
                # If a paragraph supports multiple hops, keep the earliest.
                para_idx_to_hop.setdefault(para_idx, k)

        # Paragraphs
        supporting: List[Dict[str, Any]] = []
        distractor: List[Dict[str, Any]] = []
        for p_idx, p in enumerate(row.get("paragraphs", [])):
            is_gold = bool(p.get("is_supporting", False))
            para = {
                "paragraph_id": f"para_{p_idx:03d}",
                "text": p.get("paragraph_text", "") or "",
                "title": p.get("title", "") or "",
                "is_gold": is_gold,
                "hop_index": para_idx_to_hop.get(p_idx) if is_gold else None,
            }
            (supporting if is_gold else distractor).append(para)
        item["supporting_paragraphs"] = supporting
        item["distractor_paragraphs"] = distractor

        # Decomposition -> sub_questions + task_shared_items
        sub_qs: List[str] = []
        task_shared: List[Dict[str, Any]] = []
        for k, d in enumerate(decomp, start=1):
            sub_q = d.get("question") or row.get("question", "")
            sub_qs.append(sub_q)
            para_idx = d.get("paragraph_support_idx")
            if isinstance(para_idx, int) and 0 <= para_idx < len(row.get("paragraphs", [])):
                para = row["paragraphs"][para_idx]
                text = para.get("paragraph_text", "") or ""
                first_sentence = text.split(".")[0].strip()
                if first_sentence:
                    task_shared.append(
                        {
                            "hop_index": k,
                            "text": first_sentence,
                            "source_paragraph_id": f"para_{para_idx:03d}",
                        }
                    )
        if not sub_qs:
            sub_qs = [item["question"]]
        item["sub_questions"] = sub_qs
        item["task_shared_items"] = task_shared
        item["hop_count"] = max(len(decomp), 2)

        # supporting_facts: reuse decomposition indices (sentence-level not available in MuSiQue)
        item["supporting_facts"] = [
            {"paragraph_id": f"para_{d['paragraph_support_idx']:03d}", "sentence_idx": 0}
            for d in decomp
            if isinstance(d.get("paragraph_support_idx"), int)
        ]
        return item
