"""
Batch dataset pipeline.

Iterates a loader's raw_items, builds episodes for each requested
`target_graph_type`, and shards the output into per-subset JSONL files.

S4 write protection: episodes with `s4_eligible=True` are only written to the
`test` split; attempts to write them elsewhere are rejected and logged.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from coscope.core.types import Episode, GraphType, ReasoningPathType, SubsetLabel
from coscope.utils.loaders import get_loader
from coscope.utils.output.serializer import Serializer
from coscope.utils.output.stats_reporter import StatsReporter, SubsetCoverageError
from coscope.construction.episode_builder import EpisodeBuilder


logger = logging.getLogger(__name__)


class DatasetPipeline:
    """Batch construction across raw_items x target_graph_types."""

    def __init__(
        self,
        *,
        episode_builder: Optional[EpisodeBuilder] = None,
        serializer: Optional[Serializer] = None,
        stats_reporter: Optional[StatsReporter] = None,
        processed_dir: Union[str, Path] = "coscope/data/processed",
        data_dir: Union[str, Path] = "coscope/data/raw",
        reasoning_path_type: str = "got",
    ):
        self.reasoning_path_type = reasoning_path_type.lower()
        self.reasoning_path_enum = self._normalize_reasoning_path_type(
            reasoning_path_type
        )
        self.episode_builder = episode_builder or EpisodeBuilder(
            reasoning_path_type=self.reasoning_path_enum
        )
        self.serializer = serializer or Serializer()
        self.stats_reporter = stats_reporter or StatsReporter()
        self.processed_dir = Path(processed_dir)
        self.data_dir = Path(data_dir)

    @property
    def reasoning_root(self) -> Path:
        """Return {processed_dir}/{reasoning_path_type}/ — root for all shards."""
        return self.processed_dir / self.reasoning_path_type

    # ------------------------------------------------------------------

    def run(
        self,
        dataset: str,
        split: str,
        target_graph_types: Sequence[Union[str, GraphType]],
        *,
        max_workers: int = 1,
        seed: int = 42,
        dataset_config: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        enforce_coverage: bool = True,
        require_s4: Optional[bool] = None,
        enforce_quality: bool = True,
    ) -> Dict[str, int]:
        """
        Run the pipeline end-to-end for one (dataset, split) slice.

        Returns a mapping of shard file path -> number of episodes written.

        If `enforce_coverage=True` (default), the run asserts that every
        required rho_subset label has at least one episode; failure raises
        `SubsetCoverageError`. `require_s4` defaults to `(split == "test")`.
        """
        loader = get_loader(
            dataset,
            data_dir=self.data_dir / self._raw_subdir(dataset, dataset_config),
            split=split,
            dataset_config=dataset_config,
        )
        raw_items = loader.load()
        if limit is not None:
            raw_items = raw_items[: int(limit)]
        logger.info("Loaded %d raw_items from %s/%s", len(raw_items), dataset, split)

        episodes: List[Episode] = []
        if max_workers and max_workers > 1:
            episodes = self._build_parallel(
                raw_items, dataset, split, target_graph_types, seed, max_workers
            )
        else:
            episodes = self._build_serial(
                raw_items, dataset, split, target_graph_types, seed
            )

        # Episodes that will actually land in this split's shards
        # (S4 episodes are dropped outside `test`).
        kept = [ep for ep in episodes if not (ep.s4_eligible and split != "test")]

        if enforce_coverage:
            need_s4 = require_s4 if require_s4 is not None else (split == "test")
            self.stats_reporter.check_coverage(
                kept,
                split=split,
                require_s4=need_s4,
                dataset=dataset,
            )

        if enforce_quality:
            # §16.2 hard quality gate: raises DatasetQualityError on failure.
            qr = self.stats_reporter.check_quality(
                kept, dataset=dataset, split=split, strict=True
            )
            for w in qr.get("warnings", []):
                logger.warning("[quality] %s", w)

        return self._write_shards(episodes, dataset=dataset, split=split)

    # ------------------------------------------------------------------

    def _build_serial(
        self,
        raw_items: Iterable[Dict[str, Any]],
        dataset: str,
        split: str,
        target_graph_types: Sequence[Union[str, GraphType]],
        seed: int,
    ) -> List[Episode]:
        out: List[Episode] = []
        for raw in raw_items:
            for gt in target_graph_types:
                ep = self._safe_build(raw, dataset, split, gt, seed)
                if ep is not None:
                    out.append(ep)
        return out

    def _build_parallel(
        self,
        raw_items: List[Dict[str, Any]],
        dataset: str,
        split: str,
        target_graph_types: Sequence[Union[str, GraphType]],
        seed: int,
        max_workers: int,
    ) -> List[Episode]:
        tasks = [
            (raw, gt) for raw in raw_items for gt in target_graph_types
        ]
        out: List[Episode] = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _build_one,
                    raw,
                    dataset,
                    split,
                    gt,
                    seed,
                    self.reasoning_path_type,
                )
                for raw, gt in tasks
            ]
            for fut in as_completed(futures):
                ep_dict = fut.result()
                if ep_dict is None:
                    continue
                ep = self.serializer.episode_from_dict(ep_dict)
                out.append(ep)
        return out

    def _safe_build(
        self,
        raw: Dict[str, Any],
        dataset: str,
        split: str,
        target_graph_type: Union[str, GraphType],
        seed: int,
    ) -> Optional[Episode]:
        try:
            return self.episode_builder.build_episode(
                raw_item=raw,
                dataset=dataset,
                split=split,
                target_graph_type=target_graph_type,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("episode build failed (%s/%s/%s): %s", dataset, split, target_graph_type, exc)
            return None

    # ------------------------------------------------------------------

    def _write_shards(
        self, episodes: List[Episode], *, dataset: str, split: str
    ) -> Dict[str, int]:
        shard_episodes: Dict[str, List[Episode]] = defaultdict(list)
        rejected = 0
        for ep in episodes:
            if ep.s4_eligible and split != "test":
                rejected += 1
                logger.error(
                    "Dropping s4_eligible episode %s placed in split=%s (must be test)",
                    ep.episode_id,
                    split,
                )
                continue
            subset_key = "s4" if ep.s4_eligible else ep.rho_subset.value.lower()
            graph_key = self._graph_shard_key(ep.graph_type)
            shard_name = f"{subset_key}_{graph_key}.jsonl"
            shard_episodes[shard_name].append(ep)

        written: Dict[str, int] = {}
        split_dir = self.reasoning_root / dataset / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for shard_name, eps in shard_episodes.items():
            path = split_dir / shard_name
            written[str(path)] = self.serializer.write_jsonl(path, eps)

        stats_path = self.reasoning_root / dataset / "stats" / f"{split}_stats.json"
        self.stats_reporter.write(episodes, path=stats_path, dataset=dataset, split=split)
        written[str(stats_path)] = 1

        if rejected:
            logger.warning("Total S4 episodes rejected from %s/%s: %d", dataset, split, rejected)
        return written

    # ------------------------------------------------------------------

    @staticmethod
    def _raw_subdir(dataset: str, dataset_config: Optional[Dict[str, Any]]) -> str:
        if dataset_config and "raw_path" in dataset_config:
            return str(dataset_config["raw_path"])
        return dataset

    @staticmethod
    def _graph_shard_key(graph_type) -> str:
        value = graph_type.value if hasattr(graph_type, "value") else str(graph_type)
        return value.lower()

    @staticmethod
    def _normalize_reasoning_path_type(value: str) -> ReasoningPathType:
        raw = str(value or "").strip().lower()
        mapping = {
            "got": ReasoningPathType.GOT,
            "cot": ReasoningPathType.COT,
            "tot": ReasoningPathType.TOT,
        }
        return mapping.get(raw, ReasoningPathType.GOT)


# ============================================================
# Worker helper (top-level for pickling)
# ============================================================


def _build_one(
    raw: Dict[str, Any],
    dataset: str,
    split: str,
    target_graph_type: Union[str, GraphType],
    seed: int,
    reasoning_path_type: str,
) -> Optional[Dict[str, Any]]:
    """Worker entry point used in the ProcessPoolExecutor path."""
    from coscope.utils.output.serializer import Serializer as _Ser
    from coscope.construction.episode_builder import EpisodeBuilder as _Builder

    builder = _Builder(reasoning_path_type=reasoning_path_type)
    ep = builder.build_episode(
        raw_item=raw,
        dataset=dataset,
        split=split,
        target_graph_type=target_graph_type,
        seed=seed,
    )
    if ep is None:
        return None
    return _Ser().episode_to_dict(ep)
