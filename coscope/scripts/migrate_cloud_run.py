"""Create, verify, and restore a portable cloud experiment checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from coscope.evaluation.benchmark_registry import benchmark_path

SNAPSHOT_SCHEMA_VERSION = 1
STATE_FILES = (
    "manifest.json",
    "status.json",
    "final_metrics.json",
    "COMPLETED.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer a CoScope checkpoint without provider secrets. Stop the "
            "source service before export and start the target only after restore."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--state-dir", required=True)
    export_parser.add_argument("--data-root", required=True)
    export_parser.add_argument("--bundle-dir", required=True)
    export_parser.add_argument("--repo-root", default=".")
    export_parser.add_argument("--env-file", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle-dir", required=True)
    verify_parser.add_argument("--data-root", required=True)
    verify_parser.add_argument("--repo-root", default=".")
    verify_parser.add_argument("--env-file", required=True)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--bundle-dir", required=True)
    restore_parser.add_argument("--data-root", required=True)
    restore_parser.add_argument("--state-dir", required=True)
    restore_parser.add_argument("--repo-root", default=".")
    restore_parser.add_argument("--env-file", required=True)

    args = parser.parse_args()
    try:
        if args.command == "export":
            report = export_bundle(
                state_dir=Path(args.state_dir),
                data_root=Path(args.data_root),
                bundle_dir=Path(args.bundle_dir),
                repo_root=Path(args.repo_root),
                env_file=Path(args.env_file),
            )
        elif args.command == "verify":
            report = verify_bundle(
                bundle_dir=Path(args.bundle_dir),
                data_root=Path(args.data_root),
                repo_root=Path(args.repo_root),
                env_file=Path(args.env_file),
            )
        else:
            report = restore_bundle(
                bundle_dir=Path(args.bundle_dir),
                data_root=Path(args.data_root),
                state_dir=Path(args.state_dir),
                repo_root=Path(args.repo_root),
                env_file=Path(args.env_file),
            )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def export_bundle(
    *,
    state_dir: Path,
    data_root: Path,
    bundle_dir: Path,
    repo_root: Path,
    env_file: Path,
) -> dict[str, Any]:
    """Export an inactive run as a self-verifying directory."""
    state_dir = state_dir.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    env_file = env_file.expanduser().resolve()
    if bundle_dir.exists():
        raise ValueError(f"bundle directory already exists: {bundle_dir}")

    config = _load_run_config(state_dir)
    expected_revision = str(config["code_revision"])
    actual_revision = _git_revision(repo_root)
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"source checkout is {actual_revision}, but the run requires "
            f"{expected_revision}"
        )

    checkpoint_path = state_dir / "checkpoint.sqlite3"
    completed = (state_dir / "COMPLETED.json").exists()
    if not checkpoint_path.exists() and not completed:
        raise FileNotFoundError(
            f"neither checkpoint nor completion marker found in {state_dir}"
        )

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_dir.name}.",
            dir=bundle_dir.parent,
        )
    )
    try:
        state_output = staging / "state"
        state_output.mkdir(mode=0o700)
        if checkpoint_path.exists():
            _assert_checkpoint_quiescent(checkpoint_path)
            _backup_checkpoint(
                checkpoint_path,
                state_output / "checkpoint.sqlite3",
            )
        for name in STATE_FILES:
            source = state_dir / name
            if source.exists():
                shutil.copy2(source, state_output / name)
                os.chmod(state_output / name, 0o600)

        benchmarks = [str(item) for item in config["benchmarks"]]
        data_files = _dataset_inventory(benchmarks, data_root)
        state_files = _file_inventory(state_output, state_output.rglob("*"))
        environment = _safe_environment(env_file)
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": time.time(),
            "source_state_dir": str(state_dir),
            "code_revision": expected_revision,
            "run_fingerprint": _checkpoint_fingerprint(
                state_output / "checkpoint.sqlite3",
                state_output / "COMPLETED.json",
            ),
            "benchmarks": benchmarks,
            "required_secret_names": _required_secret_names(config),
            "runtime_environment": environment,
            "state_files": state_files,
            "data_files": data_files,
        }
        _write_json(staging / "migration_manifest.json", manifest)
        os.chmod(staging / "migration_manifest.json", 0o600)
        os.replace(staging, bundle_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "exported",
        "bundle_dir": str(bundle_dir),
        "code_revision": expected_revision,
        "state_files": len(state_files),
        "data_files": len(data_files),
        "contains_secrets": False,
    }


def verify_bundle(
    *,
    bundle_dir: Path,
    data_root: Path,
    repo_root: Path,
    env_file: Path,
) -> dict[str, Any]:
    """Verify a bundle, target checkout, and separately transferred datasets."""
    bundle_dir = bundle_dir.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    env_file = env_file.expanduser().resolve()
    manifest = _read_json(bundle_dir / "migration_manifest.json")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported migration snapshot schema")

    expected_revision = str(manifest["code_revision"])
    actual_revision = _git_revision(repo_root)
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"target checkout is {actual_revision}, but the run requires "
            f"{expected_revision}"
        )
    actual_environment = _safe_environment(env_file)
    if actual_environment != manifest["runtime_environment"]:
        raise RuntimeError(
            "target non-secret COSCOPE environment differs from the source"
        )
    _assert_required_secrets(
        env_file,
        [str(item) for item in manifest["required_secret_names"]],
    )
    _verify_inventory(bundle_dir / "state", manifest["state_files"])
    _verify_inventory(data_root, manifest["data_files"])

    checkpoint = bundle_dir / "state" / "checkpoint.sqlite3"
    if checkpoint.exists():
        _assert_checkpoint_quiescent(checkpoint)
        _assert_checkpoint_integrity(checkpoint)

    return {
        "status": "verified",
        "bundle_dir": str(bundle_dir),
        "code_revision": expected_revision,
        "run_fingerprint": manifest["run_fingerprint"],
        "state_files": len(manifest["state_files"]),
        "data_files": len(manifest["data_files"]),
    }


def restore_bundle(
    *,
    bundle_dir: Path,
    data_root: Path,
    state_dir: Path,
    repo_root: Path,
    env_file: Path,
) -> dict[str, Any]:
    """Restore a verified bundle into a new, empty state directory."""
    bundle_dir = bundle_dir.expanduser().resolve()
    state_dir = state_dir.expanduser().resolve()
    verification = verify_bundle(
        bundle_dir=bundle_dir,
        data_root=data_root,
        repo_root=repo_root,
        env_file=env_file,
    )
    if state_dir.exists() and any(state_dir.iterdir()):
        raise ValueError(f"target state directory is not empty: {state_dir}")

    state_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{state_dir.name}.",
            dir=state_dir.parent,
        )
    )
    try:
        for source in (bundle_dir / "state").iterdir():
            if source.is_file():
                shutil.copy2(source, staging / source.name)
        if state_dir.exists():
            state_dir.rmdir()
        os.replace(staging, state_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        **verification,
        "status": "restored",
        "state_dir": str(state_dir),
    }


def _load_run_config(state_dir: Path) -> dict[str, Any]:
    manifest_path = state_dir / "manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path)
    completed = _read_json(state_dir / "COMPLETED.json")
    config = completed.get("config")
    if not isinstance(config, dict):
        raise ValueError("completion marker does not contain a run config")
    return config


def _assert_checkpoint_quiescent(path: Path) -> None:
    with _connect_read_only(path) as connection:
        metadata = dict(
            connection.execute(
                """
                SELECT key, value FROM metadata
                WHERE key IN ('coordinator_owner', 'coordinator_heartbeat')
                """
            )
        )
        owner = metadata.get("coordinator_owner", "")
        running_tasks = int(
            connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )
        running_eval = int(
            connection.execute(
                "SELECT COUNT(*) FROM eval_jobs WHERE status = 'running'"
            ).fetchone()[0]
        )
    if owner or running_tasks or running_eval:
        raise RuntimeError(
            "checkpoint is still active; stop the source service cleanly before "
            f"export (owner={owner!r}, running_tasks={running_tasks}, "
            f"running_eval_jobs={running_eval})"
        )


def _backup_checkpoint(source: Path, destination: Path) -> None:
    with _connect_read_only(source) as source_connection:
        with sqlite3.connect(destination) as destination_connection:
            source_connection.backup(destination_connection)
    os.chmod(destination, 0o600)
    _assert_checkpoint_integrity(destination)


def _assert_checkpoint_integrity(path: Path) -> None:
    with _connect_read_only(path) as connection:
        result = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if result != "ok":
        raise RuntimeError(f"checkpoint integrity check failed: {result}")


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _checkpoint_fingerprint(
    checkpoint: Path,
    completion_marker: Path,
) -> str:
    if checkpoint.exists():
        with _connect_read_only(checkpoint) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'fingerprint'"
            ).fetchone()
        if row is None:
            raise ValueError("checkpoint has no experiment fingerprint")
        return str(row[0])
    completed = _read_json(completion_marker)
    return str(completed["fingerprint"])


def _dataset_inventory(
    benchmarks: list[str],
    data_root: Path,
) -> list[dict[str, Any]]:
    files: set[Path] = set()
    for benchmark in benchmarks:
        path = benchmark_path(benchmark, data_root=data_root).resolve()
        if not path.exists():
            raise FileNotFoundError(f"missing dataset for {benchmark}: {path}")
        if not path.is_relative_to(data_root):
            raise ValueError(f"dataset escapes data root: {path}")
        if path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
        else:
            files.add(path)
    return _file_inventory(data_root, files)


def _file_inventory(
    root: Path,
    files: Any,
) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(
        (Path(item) for item in files if Path(item).is_file()),
        key=lambda item: item.as_posix(),
    ):
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def _verify_inventory(
    root: Path,
    inventory: list[dict[str, Any]],
) -> None:
    for expected in inventory:
        path = (root / str(expected["path"])).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"manifest path escapes its root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"missing migration file: {path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"migration file size differs: {path}")
        if _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"migration file hash differs: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_secret_names(config: dict[str, Any]) -> list[str]:
    names = ["DEEPSEEK_API_KEY"]
    if config.get("embedding_provider") == "zhipu":
        names.append("ZHIPU_API_KEY")
    elif config.get("embedding_provider") == "dashscope":
        names.append("DASHSCOPE_API_KEY")
    return names


def _safe_environment(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        raise FileNotFoundError(f"provider environment file not found: {env_file}")
    values = dotenv_values(env_file)
    return {
        key: str(value)
        for key, value in sorted(values.items())
        if key.startswith("COSCOPE_")
        and value is not None
        and not _is_secret_name(key)
    }


def _assert_required_secrets(env_file: Path, names: list[str]) -> None:
    values = dotenv_values(env_file)
    missing = [name for name in names if not values.get(name)]
    if missing:
        raise ValueError(
            "target provider environment is missing required secrets: "
            + ", ".join(missing)
        )


def _is_secret_name(name: str) -> bool:
    return any(
        marker in name
        for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    )


def _git_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot read Git revision in {repo_root}")
    return completed.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
