from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from coscope.evaluation.checkpoint_store import (
    ExperimentCheckpoint,
    ExperimentTask,
)
from coscope.scripts.migrate_cloud_run import (
    export_bundle,
    restore_bundle,
    verify_bundle,
)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    dataset = data_root / "gsm8k" / "test.parquet"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"portable dataset")

    state_dir = tmp_path / "run"
    state_dir.mkdir()
    config = {
        "code_revision": _git_revision(),
        "benchmarks": ["gsm8k"],
        "embedding_provider": "zhipu",
    }
    (state_dir / "manifest.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (state_dir / "status.json").write_text("{}", encoding="utf-8")
    checkpoint = ExperimentCheckpoint(state_dir / "checkpoint.sqlite3")
    checkpoint.initialize_manifest(
        config,
        [ExperimentTask.create("gsm8k", "one", "cot", "coscope")],
        include_mbpp_eval=False,
    )
    env_file = tmp_path / "coscope.env"
    env_file.write_text(
        "\n".join(
            (
                "COSCOPE_RUNTIME_MODE=live",
                "COSCOPE_LLM_MODEL=deepseek-v4-flash",
                "COSCOPE_EMBEDDING_PROVIDER=zhipu",
                "DEEPSEEK_API_KEY=not-exported",
                "ZHIPU_API_KEY=also-not-exported",
            )
        ),
        encoding="utf-8",
    )
    return state_dir, data_root, env_file


def test_export_verify_and_restore_checkpoint(tmp_path: Path) -> None:
    state_dir, data_root, env_file = _make_run(tmp_path)
    bundle = tmp_path / "bundle"
    report = export_bundle(
        state_dir=state_dir,
        data_root=data_root,
        bundle_dir=bundle,
        repo_root=Path.cwd(),
        env_file=env_file,
    )

    assert report["contains_secrets"] is False
    migration_manifest = json.loads(
        (bundle / "migration_manifest.json").read_text(encoding="utf-8")
    )
    assert migration_manifest["required_secret_names"] == [
        "DEEPSEEK_API_KEY",
        "ZHIPU_API_KEY",
    ]
    serialized_manifest = (bundle / "migration_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "not-exported" not in serialized_manifest
    assert "also-not-exported" not in serialized_manifest
    assert verify_bundle(
        bundle_dir=bundle,
        data_root=data_root,
        repo_root=Path.cwd(),
        env_file=env_file,
    )["status"] == "verified"

    restored = tmp_path / "restored"
    assert restore_bundle(
        bundle_dir=bundle,
        data_root=data_root,
        state_dir=restored,
        repo_root=Path.cwd(),
        env_file=env_file,
    )["status"] == "restored"
    assert (restored / "checkpoint.sqlite3").is_file()
    restored_store = ExperimentCheckpoint(restored / "checkpoint.sqlite3")
    assert restored_store.counts()["total"] == 1


def test_export_rejects_active_checkpoint(tmp_path: Path) -> None:
    state_dir, data_root, env_file = _make_run(tmp_path)
    checkpoint = ExperimentCheckpoint(state_dir / "checkpoint.sqlite3")
    checkpoint.acquire_coordinator("active", stale_after_seconds=60)

    with pytest.raises(RuntimeError, match="still active"):
        export_bundle(
            state_dir=state_dir,
            data_root=data_root,
            bundle_dir=tmp_path / "bundle",
            repo_root=Path.cwd(),
            env_file=env_file,
        )


def test_verify_detects_dataset_drift(tmp_path: Path) -> None:
    state_dir, data_root, env_file = _make_run(tmp_path)
    bundle = tmp_path / "bundle"
    export_bundle(
        state_dir=state_dir,
        data_root=data_root,
        bundle_dir=bundle,
        repo_root=Path.cwd(),
        env_file=env_file,
    )
    (data_root / "gsm8k" / "test.parquet").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="size differs"):
        verify_bundle(
            bundle_dir=bundle,
            data_root=data_root,
            repo_root=Path.cwd(),
            env_file=env_file,
        )


def test_verify_detects_nonsecret_environment_drift(tmp_path: Path) -> None:
    state_dir, data_root, env_file = _make_run(tmp_path)
    bundle = tmp_path / "bundle"
    export_bundle(
        state_dir=state_dir,
        data_root=data_root,
        bundle_dir=bundle,
        repo_root=Path.cwd(),
        env_file=env_file,
    )
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            "deepseek-v4-flash",
            "different-model",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="environment differs"):
        verify_bundle(
            bundle_dir=bundle,
            data_root=data_root,
            repo_root=Path.cwd(),
            env_file=env_file,
        )
