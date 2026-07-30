# Repository Guidelines

## Project Structure & Module Organization

The active package is `coscope/`, following the public
`erwinmsmith/CoScope` skeleton. `core/` contains agents, topology, artifacts,
memories, and events. Runtime orchestration lives in `runtime/`; CoT, ToT, and
GoT state machines live in `reasoning/` and must remain independent from
`AgentTopology`. Security and information-flow logic belong in `scope/`,
`retrieval/`, `context/`, and `memory/`. Provider boundaries are in
`adapters/`; replay compatibility belongs in `replay/`.

Model durable role differences with `AgentClass` and
`AgentContextPolicy`; keep run-specific state in `AgentInstance`.

Tests live in `tests/` as `test_*.py`. Treat `CoScope_unified/`, `raw/`,
`eval/`, `logs/`, and `new_results/` as ignored legacy or generated assets.
Do not import them from the active package or delete them during cleanup.

## Build, Test, and Development Commands

```bash
python -m pip install -e ".[dev]"  # editable install and development tools
python -m coscope.scripts.smoke_test
pytest
ruff check .
mypy coscope
```

Run the smoke test after changing runtime wiring. Run focused tests first, then
the full suite, for policy, grouping, lifecycle, reasoning, or context changes.

## Architecture Invariants

Resolve `EffectiveView` before search. Shared retrieval may use only explicit
`public_intent` and the intersection of public views. Scope signature and
query-similarity checks are both required. Reuse must never expand access;
each agent independently authorizes, reranks, falls back, and assembles
context. Branch-local or provisional artifacts require explicit, policy-checked
promotion before entering shared context. Traces must not contain raw private
or restricted content. Raw working text stays unembedded and private; only
class-authorized summaries may enter downstream contexts.

## Coding Style & Testing

Target Python 3.10+, use four spaces, type public interfaces, and keep lines
near 100 characters. Use `snake_case` for functions/modules, `PascalCase` for
classes, and `UPPER_SNAKE_CASE` for constants. Prefer dataclasses and small
policy components over broad mutable dictionaries.

Every security invariant needs a negative test. Include cases for tenant,
scope, branch, cache/signature, lifecycle, and context exposure boundaries.

## Commits and Security

Use concise imperative subjects such as `scope: enforce shared-view
intersection`. Keep refactors separate from experimental data changes. Never
commit secrets, raw datasets, evaluation dumps, model caches, logs, or local
papers. `pyproject.toml` is the dependency source of truth.
