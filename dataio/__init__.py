"""
CoScope IO layer.

Holds the data-access and serialization boundary:

- :mod:`coscope.io.loaders` -- raw-dataset loaders (MuSiQue, HotpotQA,
  2WikiMultihopQA, MATH, GSM8K, ...). They convert third-party file formats
  into CoScope's internal raw-item dict.
- :mod:`coscope.io.serializer` -- :class:`Serializer` for the JSONL episode
  format (round-trips :class:`~coscope.core.types.Episode`).
- :mod:`coscope.io.stats_reporter` -- subset/coverage reporting over
  serialized episode shards.
- :mod:`coscope.io.validator` -- batch-level JSONL validation.

The IO layer is the correct place for future storage adapters (S3, cloud
blob, database-backed episode stores); the retrieval/evaluation pipelines
consume it through these modules and never reach below.
"""

from dataio.serializer import Serializer
from dataio.stats_reporter import StatsReporter
from dataio.validator import BatchValidator, BatchValidationResult

__all__ = [
    "Serializer",
    "StatsReporter",
    "BatchValidator",
    "BatchValidationResult",
]
