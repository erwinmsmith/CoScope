"""HotpotQA loader (distractor setting, 2-hop fixed)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from coscope.data.loaders.base_loader import BaseLoader


class HotpotLoader(BaseLoader):
    """Loader for HotpotQA (distractor). Supports both parquet and JSON layouts."""

    dataset_name = "hotpotqa"
    dataset_type = "qa"

    # Parquet shards are the distribution format present in data/raw/hotpotqa/.
    # Legacy JSON dumps (hotpot_train_v1.1.json / hotpot_dev_distractor_v1.json)
    # can be enabled via dataset_config overrides.
    DEFAULT_TRAIN = "distractor_train_00000.parquet"
    DEFAULT_DEV = "distractor_validation.parquet"

    def load(self) -> List[Dict[str, Any]]:
        path = self._resolve_split_file(self.DEFAULT_TRAIN, self.DEFAULT_DEV)
        raw = self._load_any(path)
        return [self._normalize(row, idx) for idx, row in enumerate(raw)]

    # ------------------------------------------------------------------

    def _normalize(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        item = self._empty_raw_item()
        item["original_id"] = str(row.get("_id") or f"idx_{idx:06d}")
        item["question"] = row.get("question", "") or ""
        item["answer"] = str(row.get("answer", ""))
        item["qa_type"] = row.get("type") or "bridge"
        item["hop_count"] = 2

        context = row.get("context", [])
        supporting_facts_raw = row.get("supporting_facts", [])

        titles, sentences_list, sf_pairs = self._parse_context_and_facts(context, supporting_facts_raw)

        gold_titles = {t for t, _ in sf_pairs}
        # v2.1: assign hop_index by order of first appearance of each unique title
        # in supporting_facts. HotpotQA is fixed 2-hop, so we cap at hop_2.
        title_to_hop: Dict[str, int] = {}
        for title, _ in sf_pairs:
            if title not in title_to_hop:
                title_to_hop[title] = len(title_to_hop) + 1  # 1-based

        title_map: Dict[str, Tuple[int, List[str]]] = {}
        supporting: List[Dict[str, Any]] = []
        distractor: List[Dict[str, Any]] = []
        for p_idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
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
            title_map.setdefault(title, (p_idx, list(sentences) if isinstance(sentences, list) else [str(sentences)]))

        item["supporting_paragraphs"] = supporting
        item["distractor_paragraphs"] = distractor

        # supporting_facts normalized
        facts: List[Dict[str, Any]] = []
        for title, sent_idx in sf_pairs:
            mapped = title_map.get(title)
            if mapped is None:
                continue
            facts.append(
                {"paragraph_id": f"para_{mapped[0]:03d}", "sentence_idx": int(sent_idx)}
            )
        item["supporting_facts"] = facts

        # sub_questions: use paragraph titles as hints for Solver queries
        sub_qs: List[str] = []
        for k in range(2):
            if k < len(sf_pairs):
                title = sf_pairs[k][0]
                sub_qs.append(f"Find information about {title}.")
            else:
                sub_qs.append(item["question"])
        item["sub_questions"] = sub_qs

        # task_shared_items (bridge: per-hop fact sentence; comparison: empty)
        if item["qa_type"] == "comparison":
            item["task_shared_items"] = []
        else:
            task_shared: List[Dict[str, Any]] = []
            for k in range(1, 3):
                if k - 1 >= len(sf_pairs):
                    break
                title, sent_idx = sf_pairs[k - 1]
                mapped = title_map.get(title)
                if mapped is None:
                    continue
                ctx_idx, sentences = mapped
                sentence = sentences[sent_idx] if 0 <= sent_idx < len(sentences) else (sentences[0] if sentences else "")
                task_shared.append(
                    {
                        "hop_index": k,
                        "text": str(sentence).strip(),
                        "source_paragraph_id": f"para_{ctx_idx:03d}",
                    }
                )
            item["task_shared_items"] = task_shared

        return item

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_context_and_facts(
        context: Any, supporting_facts_raw: Any
    ) -> Tuple[List[str], List[List[str]], List[Tuple[str, int]]]:
        """Flatten Hotpot's context + supporting_facts across parquet dict and JSON list formats."""
        titles: List[str] = []
        sentences_list: List[List[str]] = []
        if isinstance(context, dict):
            titles = list(context.get("title", []))
            sentences_list = [list(s) if isinstance(s, list) else [str(s)] for s in context.get("sentences", [])]
        elif isinstance(context, list):
            for entry in context:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    titles.append(str(entry[0]))
                    sents = entry[1]
                    sentences_list.append(list(sents) if isinstance(sents, list) else [str(sents)])

        sf_pairs: List[Tuple[str, int]] = []
        if isinstance(supporting_facts_raw, dict):
            sf_titles = list(supporting_facts_raw.get("title", []))
            sf_sents = list(supporting_facts_raw.get("sent_id", []))
            for t, s in zip(sf_titles, sf_sents):
                sf_pairs.append((str(t), int(s)))
        elif isinstance(supporting_facts_raw, list):
            for fact in supporting_facts_raw:
                if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                    sf_pairs.append((str(fact[0]), int(fact[1])))
                elif isinstance(fact, (list, tuple)) and fact:
                    sf_pairs.append((str(fact[0]), 0))

        return titles, sentences_list, sf_pairs
