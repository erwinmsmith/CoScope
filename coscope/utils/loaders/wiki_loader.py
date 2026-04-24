"""2WikiMultiHopQA loader."""

from __future__ import annotations

from typing import Any, Dict, List

from coscope.utils.loaders.base_loader import BaseLoader


class WikiLoader(BaseLoader):
    """Loader for the 2WikiMultiHopQA dataset (JSON format)."""

    dataset_name = "2wikimhqa"
    dataset_type = "qa"

    DEFAULT_TRAIN = "train.json"
    DEFAULT_DEV = "dev.json"
    # 2WikiMultiHopQA's official `test.json` has no gold answers /
    # supporting_facts (hidden test set), so it cannot drive rho or subset
    # labels. Fall back to dev.json for "test" split unless the caller
    # explicitly overrides `test_file` via dataset_config.
    DEFAULT_TEST = None

    def load(self) -> List[Dict[str, Any]]:
        path = self._resolve_split_file(self.DEFAULT_TRAIN, self.DEFAULT_DEV, self.DEFAULT_TEST)
        raw = self._load_any(path)
        return [self._normalize(row, idx) for idx, row in enumerate(raw)]

    # ------------------------------------------------------------------

    def _normalize(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        item = self._empty_raw_item()
        item["original_id"] = str(row.get("_id") or f"idx_{idx:06d}")
        item["question"] = row.get("question", "") or ""
        item["answer"] = str(row.get("answer", ""))
        qa_type = row.get("type") or "bridge"
        item["qa_type"] = qa_type

        context = row.get("context", []) or []
        supporting_facts_raw = row.get("supporting_facts", []) or []
        gold_titles = {fact[0] for fact in supporting_facts_raw if isinstance(fact, (list, tuple)) and fact}

        # v2.1: hop_index assigned by order of first appearance in supporting_facts.
        # For comparison type the two paragraphs are logically parallel; we still
        # assign hop_1 / hop_2 by order to give each its own workspace scope.
        title_to_hop: Dict[str, int] = {}
        for fact in supporting_facts_raw:
            if not isinstance(fact, (list, tuple)) or not fact:
                continue
            t = fact[0]
            if t not in title_to_hop:
                title_to_hop[t] = len(title_to_hop) + 1

        supporting: List[Dict[str, Any]] = []
        distractor: List[Dict[str, Any]] = []
        title_map: Dict[str, Any] = {}
        for p_idx, ctx in enumerate(context):
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                title, sentences = ctx[0], ctx[1]
            elif isinstance(ctx, dict):
                title, sentences = ctx.get("title", ""), ctx.get("sentences", [])
            else:
                continue
            text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            is_gold = title in gold_titles
            para = {
                "paragraph_id": f"para_{p_idx:03d}",
                "text": text,
                "title": title,
                "is_gold": is_gold,
                "hop_index": title_to_hop.get(title) if is_gold else None,
            }
            (supporting if is_gold else distractor).append(para)
            title_map.setdefault(title, (p_idx, sentences))

        item["supporting_paragraphs"] = supporting
        item["distractor_paragraphs"] = distractor

        # hop_count: 2Wiki is mostly 2-hop; use the supporting_facts length as hint
        hop_count = max(2, min(len(supporting_facts_raw) or 2, 4))
        item["hop_count"] = hop_count

        # supporting_facts normalized
        facts: List[Dict[str, Any]] = []
        for fact in supporting_facts_raw:
            if not isinstance(fact, (list, tuple)) or not fact:
                continue
            title = fact[0]
            sent_idx = fact[1] if len(fact) > 1 else 0
            mapped = title_map.get(title)
            if mapped is None:
                continue
            facts.append(
                {
                    "paragraph_id": f"para_{mapped[0]:03d}",
                    "sentence_idx": int(sent_idx),
                }
            )
        item["supporting_facts"] = facts

        # sub_questions: 2Wiki has no explicit sub-question, use placeholders keyed on qa_type
        item["sub_questions"] = [item["question"] for _ in range(hop_count)]

        # task_shared_items: one per hop (skip for pure comparison, since hops are independent)
        task_shared: List[Dict[str, Any]] = []
        if qa_type != "comparison" and facts:
            for k in range(1, hop_count + 1):
                if k - 1 >= len(supporting_facts_raw):
                    break
                fact = supporting_facts_raw[k - 1]
                if not isinstance(fact, (list, tuple)) or not fact:
                    continue
                title = fact[0]
                sent_idx = fact[1] if len(fact) > 1 else 0
                mapped = title_map.get(title)
                if mapped is None:
                    continue
                ctx_idx, sentences = mapped
                sentence = ""
                if isinstance(sentences, list) and sentences:
                    sentence = str(sentences[sent_idx] if sent_idx < len(sentences) else sentences[0])
                task_shared.append(
                    {
                        "hop_index": k,
                        "text": sentence.strip(),
                        "source_paragraph_id": f"para_{ctx_idx:03d}",
                    }
                )
        item["task_shared_items"] = task_shared
        return item
