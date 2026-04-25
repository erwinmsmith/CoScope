"""
Base loader for graph.

Each concrete loader is responsible for normalizing a raw dataset into the
unified `raw_item` schema defined in `docs/GoT_Dataset_Requirements_v2.md`
section 5.2. No episode construction happens here - loaders only do format
conversion.

Unified raw_item schema:
    {
        "original_id": str,
        "question": str,
        "answer": str,
        "hop_count": int,
        "sub_questions": List[str],           # QA only, Math -> []
        "solution_steps": List[str],          # Math only, QA -> []
        "supporting_paragraphs": List[dict],  # [{"paragraph_id", "text", "is_gold", "title"?}]
        "distractor_paragraphs": List[dict],  # same shape as supporting_paragraphs
        "supporting_facts": List[dict],       # [{"paragraph_id", "sentence_idx"}]
        "dataset_type": str,                  # "qa" | "math"
        "math_category": Optional[str],
        "task_shared_items": List[dict],      # [{"hop_index", "text", "source_paragraph_id"}]
        "qa_type": Optional[str],             # "bridge" | "comparison" | "bridge-comparison"
    }
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseLoader(ABC):
    """Abstract loader. Each concrete subclass must implement `load()`."""

    dataset_name: str = ""          # concrete subclass should set this
    dataset_type: str = "qa"         # "qa" | "math"

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str,
        dataset_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            data_dir: path to the dataset's raw root (e.g. "data/raw/musique/").
            split: "train" | "dev" | "test".
            dataset_config: optional per-dataset config dict with keys such as
                `train_file`, `dev_file`, `test_file`. If omitted, subclass
                defaults are used.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.dataset_config: Dict[str, Any] = dataset_config or {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Return a list of unified raw_item dicts for the configured split.
        See module docstring for the schema.
        """

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _resolve_split_file(self, default_train: str, default_dev: str, default_test: Optional[str] = None) -> Path:
        """
        Pick the raw file for the current split, honoring dataset_config overrides.
        Falls back to dev_file when split="test" and no test_file is configured.
        """
        cfg = self.dataset_config
        if self.split == "train":
            fname = cfg.get("train_file", default_train)
        elif self.split == "dev":
            fname = cfg.get("dev_file", default_dev)
        else:
            fname = cfg.get("test_file") or default_test or cfg.get("dev_file", default_dev)
        if fname is None:
            raise FileNotFoundError(
                f"{self.dataset_name}: no file configured for split={self.split}"
            )
        return self.data_dir / fname

    @staticmethod
    def _load_json(file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", [data])
        return list(data)

    @staticmethod
    def _load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    @staticmethod
    def _load_parquet(file_path: Path) -> List[Dict[str, Any]]:
        import pandas as pd  # local import to avoid hard dep at import time

        df = pd.read_parquet(file_path)
        return df.to_dict(orient="records")

    def _load_any(self, file_path: Path) -> List[Dict[str, Any]]:
        """Dispatch by file extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl(file_path)
        if suffix == ".parquet":
            return self._load_parquet(file_path)
        if suffix == ".json":
            return self._load_json(file_path)
        raise ValueError(f"Unsupported file type for loader: {file_path}")

    # ------------------------------------------------------------------
    # Shared raw_item skeleton
    # ------------------------------------------------------------------

    def _empty_raw_item(self) -> Dict[str, Any]:
        """Produce a raw_item dict with all keys set to schema defaults."""
        return {
            "original_id": "",
            "question": "",
            "answer": "",
            "hop_count": 0,
            "sub_questions": [],
            "solution_steps": [],
            "supporting_paragraphs": [],
            "distractor_paragraphs": [],
            "supporting_facts": [],
            "dataset_type": self.dataset_type,
            "math_category": None,
            "task_shared_items": [],
            "qa_type": None,
        }
