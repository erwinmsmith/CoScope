"""Durable, idempotent checkpoint storage for long-running experiments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentTask:
    task_key: str
    benchmark: str
    example_id: str
    reasoning_mode: str
    sharing_policy: str
    batch_mode: str

    @classmethod
    def create(
        cls,
        benchmark: str,
        example_id: str,
        reasoning_mode: str,
        sharing_policy: str,
        batch_mode: str = "batched",
    ) -> ExperimentTask:
        readable = "\x1f".join(
            (
                benchmark,
                example_id,
                reasoning_mode,
                sharing_policy,
                batch_mode,
            )
        )
        digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()
        return cls(
            task_key=digest,
            benchmark=benchmark,
            example_id=example_id,
            reasoning_mode=reasoning_mode,
            sharing_policy=sharing_policy,
            batch_mode=batch_mode,
        )


@dataclass(frozen=True)
class ClaimedTask(ExperimentTask):
    attempt: int


@dataclass(frozen=True)
class ClaimedEvalJob:
    job_key: str
    reasoning_mode: str
    sharing_policy: str
    batch_mode: str
    attempt: int


class ActiveCoordinatorError(RuntimeError):
    """Raised when another non-stale coordinator owns the experiment."""


class CheckpointMismatchError(RuntimeError):
    """Raised when resume options do not match the stored manifest."""


class ExperimentCheckpoint:
    """SQLite/WAL-backed state store with one coordinator and many workers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_key TEXT PRIMARY KEY,
                    benchmark TEXT NOT NULL,
                    example_id TEXT NOT NULL,
                    reasoning_mode TEXT NOT NULL,
                    sharing_policy TEXT NOT NULL,
                    batch_mode TEXT NOT NULL DEFAULT 'batched',
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'running', 'succeeded', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    started_at REAL,
                    finished_at REAL,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    result_zlib BLOB,
                    metrics_zlib BLOB,
                    answer_f1 REAL,
                    success INTEGER,
                    llm_tokens INTEGER,
                    provider_tokens INTEGER,
                    embedding_tokens INTEGER,
                    retrieval_requests INTEGER,
                    store_queries INTEGER,
                    shared_store_queries INTEGER,
                    fallback_triggers INTEGER,
                    pollution_rate REAL,
                    unauthorized_exposure INTEGER,
                    end_to_end_seconds REAL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_claim
                    ON tasks(status, next_attempt_at, benchmark, example_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_condition
                    ON tasks(benchmark, reasoning_mode, sharing_policy, status);
                CREATE TABLE IF NOT EXISTS eval_jobs (
                    job_key TEXT PRIMARY KEY,
                    reasoning_mode TEXT NOT NULL,
                    sharing_policy TEXT NOT NULL,
                    batch_mode TEXT NOT NULL DEFAULT 'batched',
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'running', 'succeeded', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    started_at REAL,
                    finished_at REAL,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    error TEXT
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if "metrics_zlib" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN metrics_zlib BLOB"
                )
            if "batch_mode" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN batch_mode TEXT "
                    "NOT NULL DEFAULT 'batched'"
                )
            eval_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(eval_jobs)")
            }
            if "batch_mode" not in eval_columns:
                connection.execute(
                    "ALTER TABLE eval_jobs ADD COLUMN batch_mode TEXT "
                    "NOT NULL DEFAULT 'batched'"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_batch_condition
                ON tasks(
                    benchmark, reasoning_mode, sharing_policy,
                    batch_mode, status
                )
                """
            )

    def initialize_manifest(
        self,
        config: Mapping[str, Any],
        tasks: Iterable[ExperimentTask],
        *,
        include_mbpp_eval: bool,
    ) -> str:
        task_list = list(tasks)
        manifest = {
            "config": config,
            "task_keys": sorted(task.task_key for task in task_list),
        }
        serialized = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key = 'fingerprint'"
            ).fetchone()
            if existing is not None and existing["value"] != fingerprint:
                raise CheckpointMismatchError(
                    "checkpoint manifest differs from this launch; use a new "
                    "state directory or resume with identical options"
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('fingerprint', ?)",
                (fingerprint,),
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('config', ?)",
                (
                    json.dumps(
                        dict(config),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('created_at', ?)",
                (str(now),),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO tasks(
                    task_key, benchmark, example_id, reasoning_mode,
                    sharing_policy, batch_mode
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        task.task_key,
                        task.benchmark,
                        task.example_id,
                        task.reasoning_mode,
                        task.sharing_policy,
                        task.batch_mode,
                    )
                    for task in task_list
                ],
            )
            if include_mbpp_eval:
                modes = sorted({task.reasoning_mode for task in task_list})
                policies = sorted({task.sharing_policy for task in task_list})
                batch_modes = sorted({task.batch_mode for task in task_list})
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO eval_jobs(
                        job_key, reasoning_mode, sharing_policy, batch_mode
                    ) VALUES(?, ?, ?, ?)
                    """,
                    [
                        (
                            f"mbpp_plus:{mode}:{policy}:{batch_mode}",
                            mode,
                            policy,
                            batch_mode,
                        )
                        for mode in modes
                        for policy in policies
                        for batch_mode in batch_modes
                    ],
                )
        return fingerprint

    def acquire_coordinator(
        self,
        owner_id: str,
        *,
        stale_after_seconds: float,
    ) -> int:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = self._metadata(connection, "coordinator_owner")
            heartbeat_raw = self._metadata(
                connection, "coordinator_heartbeat"
            )
            heartbeat = float(heartbeat_raw) if heartbeat_raw else 0.0
            if (
                owner
                and owner != owner_id
                and now - heartbeat < stale_after_seconds
            ):
                raise ActiveCoordinatorError(
                    f"experiment is already owned by active coordinator {owner}"
                )
            self._set_metadata(connection, "coordinator_owner", owner_id)
            self._set_metadata(
                connection, "coordinator_heartbeat", str(now)
            )
            recovered = connection.execute(
                """
                UPDATE tasks
                SET status = 'pending', worker_id = NULL, started_at = NULL
                WHERE status = 'running'
                """
            ).rowcount
            connection.execute(
                """
                UPDATE eval_jobs
                SET status = 'pending', worker_id = NULL, started_at = NULL
                WHERE status = 'running'
                """
            )
        return recovered

    def heartbeat(self, owner_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_owner(connection, owner_id)
            self._set_metadata(
                connection, "coordinator_heartbeat", str(time.time())
            )

    def release_coordinator(self, owner_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = self._metadata(connection, "coordinator_owner")
            if owner == owner_id:
                self._set_metadata(connection, "coordinator_owner", "")
                self._set_metadata(connection, "coordinator_heartbeat", "0")

    def retry_final_failures(self) -> int:
        with self._connect() as connection:
            task_count = connection.execute(
                """
                UPDATE tasks
                SET status = 'pending', attempts = 0,
                    next_attempt_at = 0, error = NULL
                WHERE status = 'failed'
                """
            ).rowcount
            eval_count = connection.execute(
                """
                UPDATE eval_jobs
                SET status = 'pending', attempts = 0,
                    next_attempt_at = 0, error = NULL
                WHERE status = 'failed'
                """
            ).rowcount
        return task_count + eval_count

    def claim_tasks(
        self,
        owner_id: str,
        *,
        limit: int,
        max_attempts: int,
    ) -> list[ClaimedTask]:
        if limit <= 0:
            return []
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_owner(connection, owner_id)
            rows = connection.execute(
                """
                SELECT task_key, benchmark, example_id, reasoning_mode,
                       sharing_policy, batch_mode, attempts
                FROM tasks
                WHERE status = 'pending'
                  AND attempts < ?
                  AND next_attempt_at <= ?
                ORDER BY benchmark, example_id, reasoning_mode,
                         sharing_policy, batch_mode
                LIMIT ?
                """,
                (max_attempts, now, limit),
            ).fetchall()
            if not rows:
                return []
            keys = [row["task_key"] for row in rows]
            connection.executemany(
                """
                UPDATE tasks
                SET status = 'running', attempts = attempts + 1,
                    worker_id = ?, started_at = ?, error = NULL
                WHERE task_key = ? AND status = 'pending'
                """,
                [(owner_id, now, key) for key in keys],
            )
            return [
                ClaimedTask(
                    task_key=row["task_key"],
                    benchmark=row["benchmark"],
                    example_id=row["example_id"],
                    reasoning_mode=row["reasoning_mode"],
                    sharing_policy=row["sharing_policy"],
                    batch_mode=row["batch_mode"],
                    attempt=int(row["attempts"]) + 1,
                )
                for row in rows
            ]

    def complete_task(
        self,
        owner_id: str,
        task_key: str,
        result: Mapping[str, Any],
        *,
        retain_details: bool = True,
    ) -> None:
        serialized = json.dumps(
            dict(result),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        compressed = (
            zlib.compress(serialized, level=6) if retain_details else None
        )
        compact = zlib.compress(
            json.dumps(
                compact_task_metrics(result),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
            level=6,
        )
        metrics = _result_metrics(result)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_owner(connection, owner_id)
            changed = connection.execute(
                """
                UPDATE tasks
                SET status = 'succeeded', finished_at = ?, worker_id = NULL,
                    error = NULL, result_zlib = ?, metrics_zlib = ?,
                    answer_f1 = ?, success = ?, llm_tokens = ?,
                    provider_tokens = ?, embedding_tokens = ?,
                    retrieval_requests = ?, store_queries = ?,
                    shared_store_queries = ?, fallback_triggers = ?,
                    pollution_rate = ?, unauthorized_exposure = ?,
                    end_to_end_seconds = ?
                WHERE task_key = ? AND status = 'running' AND worker_id = ?
                """,
                (
                    time.time(),
                    compressed,
                    compact,
                    *metrics,
                    task_key,
                    owner_id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(
                    f"task {task_key} is not owned by coordinator {owner_id}"
                )

    def fail_task(
        self,
        owner_id: str,
        task_key: str,
        error: str,
        *,
        max_attempts: int,
        retry_backoff_seconds: float,
    ) -> str:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_owner(connection, owner_id)
            row = connection.execute(
                "SELECT attempts FROM tasks WHERE task_key = ?",
                (task_key,),
            ).fetchone()
            if row is None:
                raise KeyError(task_key)
            attempts = int(row["attempts"])
            status = "failed" if attempts >= max_attempts else "pending"
            next_attempt_at = (
                0.0
                if status == "failed"
                else now + retry_backoff_seconds * (2 ** (attempts - 1))
            )
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, worker_id = NULL, finished_at = ?,
                    next_attempt_at = ?, error = ?
                WHERE task_key = ? AND status = 'running' AND worker_id = ?
                """,
                (
                    status,
                    now,
                    next_attempt_at,
                    error,
                    task_key,
                    owner_id,
                ),
            )
        return status

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        counts = dict.fromkeys(
            ("pending", "running", "succeeded", "failed"),
            0,
        )
        counts.update({row["status"]: int(row["count"]) for row in rows})
        counts["total"] = sum(counts.values())
        return counts

    def claim_eval_jobs(
        self,
        owner_id: str,
        *,
        limit: int,
        max_attempts: int,
    ) -> list[ClaimedEvalJob]:
        if limit <= 0:
            return []
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_owner(connection, owner_id)
            rows = connection.execute(
                """
                SELECT job_key, reasoning_mode, sharing_policy,
                       batch_mode, attempts
                FROM eval_jobs
                WHERE status = 'pending'
                  AND attempts < ?
                  AND next_attempt_at <= ?
                ORDER BY reasoning_mode, sharing_policy, batch_mode
                LIMIT ?
                """,
                (max_attempts, now, limit),
            ).fetchall()
            connection.executemany(
                """
                UPDATE eval_jobs
                SET status = 'running', attempts = attempts + 1,
                    worker_id = ?, started_at = ?, error = NULL
                WHERE job_key = ? AND status = 'pending'
                """,
                [(owner_id, now, row["job_key"]) for row in rows],
            )
            return [
                ClaimedEvalJob(
                    job_key=row["job_key"],
                    reasoning_mode=row["reasoning_mode"],
                    sharing_policy=row["sharing_policy"],
                    batch_mode=row["batch_mode"],
                    attempt=int(row["attempts"]) + 1,
                )
                for row in rows
            ]

    def complete_eval_job(
        self,
        owner_id: str,
        job: ClaimedEvalJob,
        scores: Mapping[str, Any],
    ) -> int:
        """Atomically apply official MBPP scores and finish an EvalPlus job."""
        updated = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_owner(connection, owner_id)
            rows = connection.execute(
                """
                SELECT task_key, example_id, result_zlib
                FROM tasks
                WHERE benchmark = 'mbpp_plus'
                  AND reasoning_mode = ?
                  AND sharing_policy = ?
                  AND batch_mode = ?
                  AND status = 'succeeded'
                """,
                (
                    job.reasoning_mode,
                    job.sharing_policy,
                    job.batch_mode,
                ),
            ).fetchall()
            if len(rows) != len(scores):
                raise RuntimeError(
                    f"EvalPlus returned {len(scores)} scores for {len(rows)} "
                    f"checkpointed predictions in {job.job_key}"
                )
            for row in rows:
                score = scores.get(row["example_id"])
                if score is None:
                    raise RuntimeError(
                        f"EvalPlus omitted {row['example_id']} in {job.job_key}"
                    )
                result = json.loads(zlib.decompress(row["result_zlib"]))
                metrics = score.metrics()
                result["scores"] = metrics
                result["task_success"] = bool(metrics["success"])
                serialized = json.dumps(
                    result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                connection.execute(
                    """
                    UPDATE tasks
                    SET result_zlib = ?, metrics_zlib = ?,
                        answer_f1 = ?, success = ?
                    WHERE task_key = ?
                    """,
                    (
                        zlib.compress(serialized, level=6),
                        zlib.compress(
                            json.dumps(
                                compact_task_metrics(result),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8"),
                            level=6,
                        ),
                        float(metrics["f1"]),
                        int(bool(metrics["success"])),
                        row["task_key"],
                    ),
                )
                updated += 1
            changed = connection.execute(
                """
                UPDATE eval_jobs
                SET status = 'succeeded', worker_id = NULL, finished_at = ?,
                    error = NULL
                WHERE job_key = ? AND status = 'running' AND worker_id = ?
                """,
                (time.time(), job.job_key, owner_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"EvalPlus job {job.job_key} is not owned")
        return updated

    def fail_eval_job(
        self,
        owner_id: str,
        job: ClaimedEvalJob,
        error: str,
        *,
        max_attempts: int,
        retry_backoff_seconds: float,
    ) -> str:
        now = time.time()
        status = "failed" if job.attempt >= max_attempts else "pending"
        next_attempt_at = (
            0.0
            if status == "failed"
            else now + retry_backoff_seconds * (2 ** (job.attempt - 1))
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_owner(connection, owner_id)
            connection.execute(
                """
                UPDATE eval_jobs
                SET status = ?, worker_id = NULL, finished_at = ?,
                    next_attempt_at = ?, error = ?
                WHERE job_key = ? AND status = 'running' AND worker_id = ?
                """,
                (
                    status,
                    now,
                    next_attempt_at,
                    error,
                    job.job_key,
                    owner_id,
                ),
            )
        return status

    def eval_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM eval_jobs GROUP BY status"
            ).fetchall()
        counts = dict.fromkeys(
            ("pending", "running", "succeeded", "failed"),
            0,
        )
        counts.update({row["status"]: int(row["count"]) for row in rows})
        counts["total"] = sum(counts.values())
        return counts

    def status_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._connect() as connection:
            created_at = float(
                self._metadata(connection, "created_at") or now
            )
            counts = self.counts()
            conditions = connection.execute(
                """
                SELECT benchmark, reasoning_mode, sharing_policy,
                       batch_mode, status,
                       COUNT(*) AS count,
                       AVG(answer_f1) AS answer_f1,
                       AVG(success) AS success_rate,
                       SUM(llm_tokens) AS llm_tokens,
                       SUM(provider_tokens) AS provider_tokens,
                       AVG(end_to_end_seconds) AS mean_seconds
                FROM tasks
                GROUP BY benchmark, reasoning_mode, sharing_policy,
                         batch_mode, status
                ORDER BY benchmark, reasoning_mode, sharing_policy,
                         batch_mode, status
                """
            ).fetchall()
            eval_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM eval_jobs GROUP BY status"
            ).fetchall()
        elapsed = max(now - created_at, 0.0)
        rate = counts["succeeded"] / elapsed if elapsed else 0.0
        remaining = counts["pending"] + counts["running"]
        eta = remaining / rate if rate else None
        return {
            "updated_at": now,
            "elapsed_seconds": elapsed,
            "tasks": counts,
            "completion_fraction": (
                counts["succeeded"] / counts["total"]
                if counts["total"]
                else 0.0
            ),
            "completed_tasks_per_second": rate,
            "eta_seconds": eta,
            "conditions": [dict(row) for row in conditions],
            "evalplus": {
                row["status"]: int(row["count"]) for row in eval_rows
            },
        }

    def iter_results(
        self,
        *,
        benchmark: str | None = None,
        reasoning_mode: str | None = None,
        sharing_policy: str | None = None,
        batch_mode: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        clauses = ["status = 'succeeded'", "result_zlib IS NOT NULL"]
        parameters: list[str] = []
        for column, value in (
            ("benchmark", benchmark),
            ("reasoning_mode", reasoning_mode),
            ("sharing_policy", sharing_policy),
            ("batch_mode", batch_mode),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        query = (
            "SELECT result_zlib FROM tasks WHERE "
            + " AND ".join(clauses)
            + " ORDER BY benchmark, example_id, reasoning_mode, "
            "sharing_policy, batch_mode"
        )
        with self._connect() as connection:
            cursor = connection.execute(query, parameters)
            for row in cursor:
                yield json.loads(zlib.decompress(row["result_zlib"]))

    def iter_metric_records(
        self,
        *,
        benchmark: str | None = None,
        reasoning_mode: str | None = None,
        sharing_policy: str | None = None,
        batch_mode: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        clauses = ["status = 'succeeded'", "metrics_zlib IS NOT NULL"]
        parameters: list[str] = []
        for column, value in (
            ("benchmark", benchmark),
            ("reasoning_mode", reasoning_mode),
            ("sharing_policy", sharing_policy),
            ("batch_mode", batch_mode),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        query = (
            "SELECT metrics_zlib FROM tasks WHERE "
            + " AND ".join(clauses)
            + " ORDER BY benchmark, example_id, reasoning_mode, "
            "sharing_policy, batch_mode"
        )
        with self._connect() as connection:
            cursor = connection.execute(query, parameters)
            for row in cursor:
                yield json.loads(zlib.decompress(row["metrics_zlib"]))

    def export_results_jsonl(self, output: str | Path) -> int:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        count = 0
        with temporary.open("w", encoding="utf-8") as stream:
            for result in self.iter_results():
                stream.write(
                    json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                count += 1
            stream.flush()
        temporary.replace(destination)
        return count

    def next_ready_delay(self, *, default: float = 1.0) -> float:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT MIN(next_attempt_at) AS ready_at
                FROM tasks WHERE status = 'pending'
                """
            ).fetchone()
        ready_at = row["ready_at"] if row else None
        if ready_at is None:
            return default
        return max(0.05, min(default, float(ready_at) - time.time()))

    @staticmethod
    def _metadata(connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    @staticmethod
    def _set_metadata(
        connection: sqlite3.Connection,
        key: str,
        value: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _require_owner(
        self,
        connection: sqlite3.Connection,
        owner_id: str,
    ) -> None:
        if self._metadata(connection, "coordinator_owner") != owner_id:
            raise ActiveCoordinatorError(
                f"coordinator {owner_id} no longer owns this experiment"
            )


def _result_metrics(result: Mapping[str, Any]) -> tuple[Any, ...]:
    scores = result["scores"]
    provider = result["provider_usage"]
    categories = provider.get("by_category", {})
    embedding = categories.get("embedding", {})
    retrieval = result["retrieval"]
    context = result["context"]
    safety = result["safety"]
    latency = result["latency"]
    return (
        float(scores["f1"]),
        int(bool(result["task_success"])),
        int(categories.get("llm", {}).get("total_tokens", 0)),
        int(provider.get("all", {}).get("total_tokens", 0)),
        int(embedding.get("total_tokens", 0)),
        int(retrieval["requests"]),
        int(retrieval["store_queries"]),
        int(retrieval["shared_store_queries"]),
        int(retrieval["fallback_triggers"]),
        float(context["pollution_rate"]),
        int(safety["unauthorized_context_exposure"]),
        float(latency["end_to_end_seconds"]),
    )


def compact_task_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Retain all aggregate inputs while dropping prompts, outputs, and evidence."""
    return {
        "benchmark": result["benchmark"],
        "example_id": result["example_id"],
        "reasoning_mode": result["reasoning_mode"],
        "arm": result["arm"],
        "batch_mode": result.get("batch_mode", "batched"),
        "scores": dict(result["scores"]),
        "task_success": bool(result["task_success"]),
        "provider_usage": {
            "all": dict(result["provider_usage"]["all"]),
            "by_category": {
                name: dict(values)
                for name, values in result["provider_usage"][
                    "by_category"
                ].items()
            },
        },
        "retrieval": {
            key: result["retrieval"][key]
            for key in (
                "requests",
                "groups",
                "shared_store_queries",
                "private_store_queries",
                "store_queries",
                "shared_recall_savings",
                "query_reduction_rate",
                "fallback_triggers",
                "memory_store_backend",
                "vector_store_query_seconds",
                "recall_at_10",
                "mrr_at_10",
                "latency_seconds",
                "wall_clock_latency_seconds",
                "embedding_seconds",
            )
            if key in result["retrieval"]
        },
        "context": {
            key: result["context"][key]
            for key in (
                "selected_items",
                "mean_duplicate_evidence_rate",
                "polluted_items",
                "pollution_rate",
                "private_thinking_exposure",
                "cross_role_knowledge_exposure",
            )
        },
        "safety": dict(result["safety"]),
        "data_flow": {
            agent_id: {
                "private_memory_id": flow.get("private_memory_id"),
                "published_memory_id": flow.get("published_memory_id"),
            }
            for agent_id, flow in result["data_flow"].items()
        },
        "generation_budget": dict(result["generation_budget"]),
        "latency": dict(result["latency"]),
        "reasoning": {
            agent_id: {
                "node_count": values["node_count"],
                "generated_thoughts": values["generated_thoughts"],
                "llm_calls": values["llm_calls"],
            }
            for agent_id, values in result["reasoning"].items()
        },
    }
