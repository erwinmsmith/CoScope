"""Download pinned public benchmark files and verify their digests."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetFile:
    dataset: str
    relative_path: str
    url: str
    sha256: str


FILES = (
    DatasetFile(
        "aime2024",
        "aime/aime_2024_I.parquet",
        (
            "https://huggingface.co/datasets/MathArena/aime_2024_I/resolve/"
            "ea5b061c3e8039dc9858defaafc407d04b995e9f/"
            "data/train-00000-of-00001.parquet"
        ),
        "e4033c704609cc7cdfe712ed410357b190733dec75aa5b54a39adc55add49393",
    ),
    DatasetFile(
        "aime2024",
        "aime/aime_2024_II.parquet",
        (
            "https://huggingface.co/datasets/MathArena/aime_2024_II/resolve/"
            "29d5d31e9b46e215fc24d9b2a3047506823dd101/"
            "data/train-00000-of-00001.parquet"
        ),
        "eab2b6a77c048ec4efb7d5f91d1f7548b7fc8a08566b9fcddd1b014699f30dbf",
    ),
    DatasetFile(
        "aime2025",
        "aime/aime_2025.parquet",
        (
            "https://huggingface.co/datasets/MathArena/aime_2025/resolve/"
            "c94da77eb22bbd6439e62a323bec18493a421302/"
            "data/train-00000-of-00001.parquet"
        ),
        "9f9066ff48ad2e31f9bf1b1ac6d5e80693195f987985f2859f89dd25ffa51c2d",
    ),
    DatasetFile(
        "mbpp_plus",
        "mbpp_plus/MbppPlus.jsonl.gz",
        (
            "https://github.com/evalplus/mbppplus_release/releases/download/"
            "v0.2.0/MbppPlus.jsonl.gz"
        ),
        "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="aime2024,aime2025,mbpp_plus",
        help="comma-separated selection",
    )
    parser.add_argument("--root", default="raw")
    args = parser.parse_args()
    requested = {
        item.strip() for item in args.datasets.split(",") if item.strip()
    }
    known = {item.dataset for item in FILES}
    unknown = sorted(requested - known)
    if unknown:
        parser.error(
            f"unknown datasets: {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(known))}"
        )
    root = Path(args.root)
    for item in FILES:
        if item.dataset not in requested:
            continue
        target = root / item.relative_path
        if target.exists() and _sha256(target) == item.sha256:
            print(f"verified {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        _download(item.url, temporary_path)
        try:
            actual = _sha256(temporary_path)
            if actual != item.sha256:
                raise ValueError(
                    f"digest mismatch for {item.dataset}: "
                    f"expected {item.sha256}, got {actual}"
                )
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"downloaded {target}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CoScope benchmark downloader/0.2"},
    )
    last_error: OSError | None = None
    for attempt in range(4):
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                target.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            return
        except OSError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


if __name__ == "__main__":
    raise SystemExit(main())
