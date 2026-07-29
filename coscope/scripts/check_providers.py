"""Make minimal live calls to verify configured LLM and embedding providers."""

from __future__ import annotations

import argparse
import os

from coscope import CoScopeRuntime
from coscope.config import CoScopeSettings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        default=os.environ.get("COSCOPE_ENV_FILE", ".env"),
    )
    args = parser.parse_args()
    settings = CoScopeSettings.from_env(args.env_file)
    if settings.runtime_mode != "live":
        parser.error("provider checks require COSCOPE_RUNTIME_MODE=live")
    settings.llm.validate()
    settings.embedding.validate()
    runtime = CoScopeRuntime.from_settings(settings)
    vector = runtime.embedder.embed("CoScope provider check")
    if runtime.llm is None:
        raise RuntimeError("live LLM adapter was not configured")
    output = runtime.llm.invoke(
        [{"role": "user", "content": "Reply with exactly: OK"}]
    )
    print(
        f"Embedding: {runtime.embedder.model_version} ({len(vector)} dimensions)"
    )
    print(f"LLM: {runtime.llm.model_version} ({output.text.strip()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
