"""
Disk-cached embedder wrapper.

Wraps any ``Embedder`` (DashScope, sentence-transformers, ...) and caches
embeddings on disk keyed by SHA1(model_name | dim | text). Designed for the
Mode-B evaluation pipeline where the same text (queries / memory contents)
is embedded many times across variants. Cache is content-addressed so it is
safe across runs and across variant orderings.

Storage backend: SQLite. Single file, atomic writes, std-lib only.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.interfaces import Embedder, EmbeddingResult


def _key(model: str, dim: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(f"{model}\x00{dim}\x00".encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class DiskCachedEmbedder(Embedder):
    """Persistent SHA1-keyed cache around an inner Embedder."""

    def __init__(self, inner: Embedder, cache_path: Path):
        self.inner = inner
        self.dim = int(inner.dim)
        self.name = f"{inner.name}+disk_cache"
        self.model = getattr(inner, "model", inner.name)

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.commit()

        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Embedder Protocol
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text]).vectors[0]

    def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        n = len(texts)
        if n == 0:
            return EmbeddingResult(
                vectors=np.zeros((0, self.dim), dtype=np.float32),
                dim=self.dim,
                model=self.model,
            )

        keys = [_key(self.model, self.dim, t) for t in texts]
        cached = self._fetch_many(keys)

        out = np.empty((n, self.dim), dtype=np.float32)
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        miss_keys: List[str] = []
        for i, k in enumerate(keys):
            v = cached.get(k)
            if v is None:
                miss_indices.append(i)
                miss_texts.append(texts[i])
                miss_keys.append(k)
            else:
                out[i] = v

        if miss_texts:
            res = self.inner.embed_batch(miss_texts)
            new_vecs = res.vectors.astype(np.float32, copy=False)
            for j, idx in enumerate(miss_indices):
                out[idx] = new_vecs[j]
            self._store_many(miss_keys, new_vecs)
            self._misses += len(miss_indices)
        self._hits += n - len(miss_indices)

        return EmbeddingResult(
            vectors=out,
            dim=self.dim,
            model=self.model,
            meta={"cache_hits": self._hits, "cache_misses": self._misses},
        )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _fetch_many(self, keys: List[str]) -> dict:
        if not keys:
            return {}
        out: dict = {}
        with self._lock:
            cur = self._db.cursor()
            CHUNK = 500
            for i in range(0, len(keys), CHUNK):
                ks = keys[i : i + CHUNK]
                placeholders = ",".join("?" * len(ks))
                cur.execute(
                    f"SELECT key, vec FROM cache WHERE key IN ({placeholders})",
                    ks,
                )
                for row in cur.fetchall():
                    out[row[0]] = np.frombuffer(row[1], dtype=np.float32)
        return out

    def _store_many(self, keys: List[str], vecs: np.ndarray) -> None:
        if len(keys) == 0:
            return
        rows = [(k, vecs[i].tobytes()) for i, k in enumerate(keys)]
        with self._lock:
            self._db.executemany(
                "INSERT OR IGNORE INTO cache(key, vec) VALUES (?, ?)", rows
            )
            self._db.commit()

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # engine.EmbeddingProvider shape (used by retrieval pipeline)
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(query)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        result = self.embed_batch(texts)
        return [row for row in result.vectors]

    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            cur = self._db.cursor()
            cur.execute("SELECT COUNT(*) FROM cache")
            count = cur.fetchone()[0]
        return {
            "cache_path": str(self._db_path),
            "entries": int(count),
            "hits": self._hits,
            "misses": self._misses,
            "inner": self.inner.name,
            "dim": self.dim,
        }

    def close(self) -> None:
        with self._lock:
            self._db.close()
