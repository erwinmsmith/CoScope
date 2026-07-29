"""Make minimal live calls to verify configured LLM and embedding providers."""

from __future__ import annotations

from coscope import CoScopeRuntime


def main() -> int:
    runtime = CoScopeRuntime.from_env()
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
