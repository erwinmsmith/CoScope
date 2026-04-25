# CoScope Architecture Plan

Engineering-oriented reference for the current code layout, the gaps between
it and the README, and a concrete target structure that is friendly to a
later database integration.

**2026-04-25 flat-layout adoption.** The repository no longer wraps source
under a top-level `coscope/` package. All subsystems (`engine/`, `memory/`,
`llm/`, `dataio/`, `prompts/`, `retrieval/`, ...) sit directly at the repo
root, siblings of `CLAUDE.md` / `README.md` / `pyproject.toml`. The data
IO layer was renamed `io/` -> `dataio/` to avoid shadowing Python's stdlib
`io` module. The distribution name on PyPI remains `coscope`; only the
import paths changed.

This document is **the single source of truth for code organization**. When
the target layout below is fully applied, `README.md` should be updated to
match it verbatim.

---

## 1. Current Layout (Apr 2026)

```
/
├── agents/            # Planner / Solver / Verifier builders
├── auth/              # DEPRECATED shim -> memory/crud/auth/
├── config/            # YAML + env configuration
├── construction/      # Episode and dataset pipelines
├── core/              # Types, interfaces, scope IDs
├── data/              # Raw / processed / interim data (gitignored)
├── docs/              # Design docs (Schema.md, ...)
├── embedding/         # ST + DashScope embedders
├── engine.py          # CoScope facade (single file, odd)
├── evaluation/        # Metrics, runners, synthetic
├── examples/          # Usage snippets
├── graph/             # GoT / CoT / ToT builders + template strings
├── main.py            # CLI entry
├── memory/
│   ├── store.py       # In-memory backend (policy + scope filter)
│   ├── *_builder.py   # Workspace / task-shared / private / restricted
│   └── crud/
│       ├── manager.py
│       ├── types.py
│       └── auth/      # Access control, policy validator, S4 checker
├── prompts/
│   ├── manager.py     # Template manager (file / JSON / YAML)
│   ├── templates.py
│   └── registry.py    # Central DB-pluggable prompt registry
├── retrieval/         # Modular pipeline (see §3)
├── rollout/           # Rollout engine, prompt assembly, LLM clients
├── scripts/           # CLI entry points
└── utils/
    ├── loaders/       # Dataset loaders (MuSiQue, HotpotQA, ...)
    ├── output/        # serializer.py, stats_reporter, validator
    └── split/         # subset_assigner, split_manager
```

### Pain points

1. `utils/` has become a dumping ground:
   - `utils/loaders/` is a **data-access layer**, not utilities.
   - `utils/output/serializer.py` is **core IO** (Episode <-> JSONL), not a
     utility.
   - `utils/split/` is **evaluation logic** (subset S1-S4 assignment).
2. `rollout/` owns the LLM clients (`dashscope_client.py`,
   `template_llm_client.py`, `llm_client.py`), but the clients are also
   consumed by `scripts/generate_query_intent.py`. LLM access should be a
   first-class, rollout-agnostic layer.
3. `engine.py` is a lone module while every other subsystem is a package.
4. Prompt templates are split across three locations:
   - `prompts/templates.py` (role prompts)
   - `graph/{got,cot,tot}/prompt_templates.py` (graph-specific)
   - `rollout/prompt_assembly.py` (step 2 / scratch headers)
   The `prompts/registry.py` bootstrap partially solves this by mirroring
   everything into a single keyed store, but the source strings still live
   in three modules.

### What is already in good shape

- `retrieval/` is well-decomposed into
  `encoder / router / matrix / projection / retriever / reranker / fallback / fusion`
  and `pipeline.py`. This is the **designated performance-tuning surface**
  and should not be restructured casually.
- `memory/store.py` exposes `create_memory_store(...)` so the backend is
  already pluggable (in-memory default, faiss-compatible adapter possible).
- `memory/crud/auth/` holds all access control together with CRUD.
- `prompts/registry.py` exposes `set_backend(...)` so prompts can be served
  from a database without touching call sites.
- `embedding/` already follows the adapter pattern.

---

## 2. Target Layout

Goal: clean subsystem boundaries, each with a stable integration surface for
the future database product.

```
/
├── agents/            # unchanged
├── config/            # unchanged
├── construction/      # unchanged
├── core/              # unchanged
├── engine/            # <- was engine.py; facade package
│   └── __init__.py    #    exports CoScope
├── evaluation/
│   ├── metrics.py
│   ├── runner.py
│   ├── jsonl_runner.py
│   ├── synthetic.py
│   └── split/         # <- from utils/split/
├── graph/             # unchanged (templates stay, registry mirrors them)
├── dataio/                # <- new top-level "data access" layer
│   ├── serializer.py  # <- from utils/output/serializer.py
│   ├── stats_reporter.py
│   ├── validator.py
│   └── loaders/       # <- from utils/loaders/
│       ├── musique_loader.py
│       ├── hotpot_loader.py
│       └── ...
├── llm/               # <- new; extracted from rollout/
│   ├── __init__.py    #    exports LLMClient, LLMResponse
│   ├── base.py        # <- llm_client.py
│   ├── dashscope.py
│   └── template.py
├── memory/
│   ├── store.py       # unchanged API; backend pluggable
│   ├── crud/
│   │   ├── manager.py
│   │   ├── types.py
│   │   └── auth/      # unchanged
│   └── builders/      # <- from memory/*_builder.py, grouped
├── prompts/
│   ├── registry.py    # unchanged; central DB-pluggable store
│   ├── manager.py
│   └── templates.py
├── retrieval/         # unchanged (perf-tuning surface)
├── rollout/
│   ├── rollout_engine.py
│   ├── prompt_assembly.py
│   ├── artifact_types.py
│   ├── artifact_writer.py
│   ├── trace_validator.py
│   ├── rho_v3.py
│   └── fallback_synth.py
├── scripts/           # unchanged
(utils/ removed entirely)
```

### Deltas from current layout

| Move | From | To | Risk |
| ---- | ---- | -- | ---- |
| Auth | `/auth/` | `/memory/crud/auth/` | **DONE** (shim kept) |
| Prompts central registry | - | `/prompts/registry.py` | **DONE** |
| LLM clients | `rollout/{dashscope,template,llm}_client.py` | `/llm/{dashscope,template,base}.py` | **DONE** (shims kept in rollout/) |
| Dataset loaders | `utils/loaders/` | `dataio/loaders/` | **DONE** |
| JSONL serializer | `utils/output/` | `dataio/` | **DONE** |
| Subset split | `utils/split/` | `evaluation/split/` | **DONE** |
| Engine facade | `engine.py` | `engine/__init__.py` | **DONE** |
| Memory builders | `memory/*_builder.py` | `memory/builders/` | low (cosmetic; deferred) |

Each move keeps a backward-compatibility shim at the old path for one release
cycle, then the shim is removed.

---

## 3. DB Integration Extension Points

When the external database product is wired in, these are the only pluggable
surfaces that need to be touched. Everything else reuses them.

| Concern | Extension point | How to plug |
| ------- | --------------- | ----------- |
| Memory storage | `memory.store.create_memory_store` + `InMemoryMemoryStore` protocol | Implement `MemoryStore` protocol; return it from a factory. |
| Access control | `memory.crud.auth.AccessController` | Subclass or replace with a DB-backed controller that reads permissions from the product. |
| Prompts | `prompts.registry.set_backend(...)` | Implement `PromptBackend` (get / set / keys / contains) against the DB. |
| Embeddings | `embedding.*` | Any class with an `embed_query` / `embed_documents` interface. |
| LLM | `llm.LLMClient` (target) | Implement `generate` / `generate_batch`; point `ArtifactRolloutEngine` at it. |
| Retrieval candidates | `retrieval.retriever/*` | Plug a vector DB retriever behind the existing protocol. |

The `retrieval/` pipeline is explicitly left out: it is the perf-tuning
surface, not an integration surface, and it already calls into the pluggable
layers above.

---

## 4. Design Principles (added after review)

Two additional constraints emerging from user feedback:

**construction/ vs io/ boundary**
- `io/loaders/` is responsible for reading raw third-party formats and
  converting them to CoScope's internal raw-item dict (purely I/O).
- `construction/` builds `Episode` objects from those raw-item dicts,
  creating agents, memory entries, retrieval requests and subset tags.
  It is domain logic, not I/O. The layering is: **raw file → io/loaders
  → dict → construction/ → Episode**.

**Scripts must be thin**
- `scripts/` contains only argument parsing and top-level orchestration
  calls. No business logic. If a script needs an LLM call, it imports
  from `llm`, not from `rollout/`. If it needs serialisation, it
  imports from `io`. This was the root cause of the original
  cross-layer import (`scripts/generate_query_intent.py` importing from
  `rollout.dashscope_client` directly) — now fixed.

## 5. Next Actions (ordered)

1. ~~LLM extraction~~ **DONE**
2. ~~IO layer~~ **DONE**
3. ~~Evaluation split move~~ **DONE**
4. ~~Engine package~~ **DONE**
5. ~~README rewrite~~ **DONE**
6. **Memory builders grouping** — `memory/*_builder.py` → `memory/builders/`
   (cosmetic; deferred until a natural refactor moment).
7. **Prompt source consolidation** — move raw template strings from
   `graph/*/prompt_templates.py` into `prompts/` as the single source;
   keep the graph modules as thin importers. The registry already mirrors
   them; this closes the source-of-truth gap.
8. **utils/ removal** — once `utils/` shell is confirmed unused, delete it.
