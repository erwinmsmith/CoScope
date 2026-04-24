# System-Layer Partitioning & LLM Rollout — Design v1

Status: DRAFT for review (not yet implemented)
Supersedes: `workspace_semantic_hop` knowledge partitioning in the GoT pipeline.
Authors: CoScope team
Scope: `coscope/data/**`

---

## 1. Motivation

The current pipeline partitions **knowledge** (corpus) by hop-level workspace
scopes (`workspace/{dataset}/semantic/hop_k`). Because the underlying corpus
is homogeneous (all Wikipedia passages, all math steps), this partitioning is
synthetic and injects artificial barriers that hurt recall without producing
meaningful scope overlap.

Per CoScope's original design (`introduction.md` §5.1), the scope hierarchy is
defined over **runtime system data produced by agents during MAS execution**,
not over the underlying corpus. The partitioning that should drive `rho` is:

- `agent_private` — private drafts, scratchpads produced by a single agent
- `task_shared`  — plans, conclusions, blackboard items shared across agents
- `session`      — current-session messages, tool calls, state updates
- `workspace`    — long-lived knowledge (the corpus itself)
- `restricted`   — audit / quarantine / policy-isolated zones

This document specifies how to **(a)** flatten the `workspace` layer into a
standard RAG source, and **(b)** populate the other four layers with real
artifacts produced by an offline LLM-driven MAS rollout.

## 2. Non-Goals

- **Graph topology is NOT LLM-generated.** Graphs remain produced by
  `got/graph_templates.py` as deterministic fixtures; the LLM only *executes*
  those graphs (role-plays planner/solver/verifier), it does not *design*
  them.
- **No new core types.** `coscope.core.types.MemoryEntry` is rich enough
  (memory_type, scope_id, visibility, provenance.parent_event, timestamp,
  metadata). We add conventions and builders, not new classes.
- **No breaking change to Episode / shard schema beyond memory_entries
  content.** `got_graph`, `agents`, `retrieval_requests`, `ground_truth`
  remain bit-compatible.

## 3. Scope Layer Mapping (Before vs After)

| Layer (logical)  | Current impl.                         | After                                          | Content source             |
|------------------|----------------------------------------|------------------------------------------------|----------------------------|
| `workspace`      | `workspace_semantic_global` + `workspace_semantic_hop_k` | **Flat** `workspace_semantic(dataset)` only   | Corpus (static)            |
| `task_shared`    | oracle gold first-sentence             | LLM-generated **conclusion** artifacts         | MAS rollout                |
| `agent_private`  | placeholder with confidence=0          | LLM-generated **scratch** artifacts            | MAS rollout                |
| `session`        | (missing)                              | Message stream + tool-call log                 | MAS rollout                |
| `restricted`     | pre-generated restricted docs          | + LLM-generated **audit** artifacts (S4)       | MAS rollout + offline gen  |

`workspace_semantic_hop_k` is **removed** (including its scope_id helper).

## 4. Artifact Conventions on `MemoryEntry`

We do not subclass `MemoryEntry`. Instead we define the conventions below.

### 4.1 Artifact slots

Each GoT node emits one or more artifacts. The slot name determines layer,
memory_type, and visibility:

| Slot            | Producer role          | Layer (scope_id)              | memory_type | visibility        |
|-----------------|------------------------|-------------------------------|-------------|-------------------|
| `plan`          | Planner                | `task_shared`                 | ARTIFACT    | [TEAM]            |
| `scratch`       | Planner/Solver/Verifier| `agent_private(ep, node)`     | EPISODIC    | [OWNER]           |
| `conclusion`    | Solver                 | `task_shared`                 | ARTIFACT    | [TEAM]            |
| `message`       | any                    | `session(ep)`                 | EPISODIC    | [SESSION]         |
| `tool_call`     | any                    | `session(ep)`                 | WORKING     | [SESSION]         |
| `audit_report`  | Verifier               | `task_restricted(ep)`         | ARTIFACT    | [RESTRICTED]      |

`SESSION` visibility uses the existing `VisibilityLevel.SESSION` enum value.

### 4.2 Mandatory metadata fields

Every artifact MemoryEntry must carry these in `metadata`:

```
{
  "scope_layer":       "agent_private" | "task_shared_artifact"
                     | "task_shared_plan" | "session_message"
                     | "session_tool" | "restricted_audit",
  "slot":              "plan" | "scratch" | "conclusion" | "message"
                     | "tool_call" | "audit_report",
  "source_node_id":    "planner" | "solver_1" | ...       # producer node
  "hop_index":         int | None,
  "topo_index":        int,                                # stream order (0..N-1)
  "parent_artifact_ids": [str, ...],                       # upstream artifacts consumed
  "is_gold_evidence":  bool,                               # kept for ground-truth join
  "required_clearance": int,
  "dataset":           str,
  "episode_id":        str
}
```

- `provenance.agent_id` = `"{episode_id}_{node_id}"` (producer)
- `provenance.parent_event` = first parent artifact id (primary upstream)
- `provenance.timestamp` = deterministic wall time seeded from `(episode_id, topo_index)`
- `provenance.source` = `"llm:{model_name}"` for LLM-produced, `"template:v1"` for synthesized fallback

### 4.3 Content synthesis policy

Content is produced by `ArtifactRolloutEngine` in priority order:

1. **LLM rollout** (default): real model generates the text via §18 prompts.
2. **Template fallback**: if LLM unavailable, synthesize from gold evidence
   (current `task_shared_builder` logic moved into this fallback). Tagged
   `provenance.source="template:v1"` so downstream experiments can filter.

## 5. Artifact Trace DAG

The set of artifacts for one episode forms a DAG:

```
planner.plan ──► solver_1.scratch ──► solver_1.conclusion ──┐
                                                             ├─► solver_2.scratch ──► solver_2.conclusion ──► verifier.scratch ──► verifier.audit_report
planner.plan ────────────────────────────────────────────────┘
```

Invariants (validated by `trace_validator`):

- DAG (no cycles, topological order matches `GoTGraph` ancestor order).
- Every `conclusion` at node N has `parent_artifact_ids` ⊇ {conclusion of p | p ∈ parents(N)}.
- Every `scratch` at node N has `parent_artifact_ids` ⊇ {conclusion of p | p ∈ parents(N)} ∪ {planner.plan}.
- No artifact in `task_shared` layer may have `visibility=[OWNER]`.
- `audit_report` exists iff `graph_type == POLICY_ISOLATED`.

## 6. `rho_v3` Redefinition

`rho` is recomputed over the **artifact trace**, not the corpus.

### 6.1 Accessibility

For each pair of solver nodes `(i, j)`:

```
A_i = { artifact.memory_id | artifact visible to solver_i under its ScopeSpec
                              AND policy(artifact) compatible with policy(solver_i) }
```

"Visible to solver_i" means at least one of:

- `artifact.scope_id == task_shared(ep)` and `VisibilityLevel.TEAM` ∈ request.policy.visibility
- `artifact.scope_id == agent_private(ep, node_i)` (only own private)
- `artifact.scope_id == session(ep)` and `VisibilityLevel.SESSION` ∈ request.policy.visibility
- `artifact.scope_id == task_restricted(ep)` and request carries `RESTRICTED` clearance

Workspace (RAG corpus) is **excluded from rho** — all agents share it equally,
so including it would dominate the union and make `rho ≈ 1` trivially.

### 6.2 Pairwise IoU then aggregate

```
rho(episode) = mean over unordered solver pairs (i, j) of |A_i ∩ A_j| / |A_i ∪ A_j|
```

Degenerate cases:

- single solver: `rho = 1.0` if `|A_1| > 0` else `0.0` (unchanged from current)
- no solvers: `rho = 0.0`

### 6.3 Expected distribution by graph_type

| graph_type       | Expected rho                | Why                                                        |
|------------------|-----------------------------|------------------------------------------------------------|
| LINEAR (2-hop)   | moderate (0.30–0.55)        | solver_2 reads planner.plan + solver_1.conclusion          |
| LINEAR (≥3-hop)  | higher (0.50–0.70)          | shared prefix grows                                        |
| FORK             | low (0.10–0.25)             | parallel branches share only planner.plan                  |
| FORK_MERGE       | high (0.55–0.75)            | merge node reads all branches                              |
| INDEPENDENT      | very low (~0.05)            | no cross-solver shared conclusions                         |
| POLICY_ISOLATED  | low + `policy_conflict=True`| restricted layer inaccessible to solvers                   |

The S1/S2/S3 thresholds remain unchanged; S1–S3 coverage is now produced by
**graph topology** rather than corpus partitioning.

## 7. Rollout Engine Interface

Location: `coscope/data/rollout/rollout_engine.py`.

```
class ArtifactRolloutEngine:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry,   # dispatches GoT/CoT/ToT per reasoning_path_type
        max_retries: int = 2,
        temperature: float = 0.3,
        seed: int = 42,
    ): ...

    def run(
        self,
        *,
        raw_item: Dict[str, Any],
        got_graph: GoTGraph,
        dataset: str,
        episode_id: str,
        reasoning_path_type: ReasoningPathType,
    ) -> ArtifactTrace: ...
```

Where `ArtifactTrace` is a lightweight typed aggregate (NOT a new runtime type):

```
@dataclass
class ArtifactTrace:
    episode_id: str
    entries: List[MemoryEntry]              # the actual artifacts (private + shared + session + restricted)
    producer_index: Dict[str, List[str]]    # node_id -> list of memory_ids it produced
    consumer_index: Dict[str, List[str]]    # node_id -> list of memory_ids it must read
    stream: List[Tuple[int, str]]           # [(topo_index, memory_id)] in stream order
```

Runtime behavior:

1. Topologically order `got_graph.nodes`.
2. For each node, call the appropriate prompt builder from `got/cot/tot/prompt_templates.py`.
3. Emit `scratch` first (private), then `conclusion` (shared).
4. Verifier additionally emits `audit_report` (restricted) iff POLICY_ISOLATED.
5. Register every produced `memory_id` into parent index; downstream calls
   fill `parent_artifact_ids` from this index.
6. On LLM failure after retries, fall back to template synthesis and mark
   `provenance.source="template:v1"`.

## 8. LLM Client Interface

Location: `coscope/data/rollout/llm_client.py`.

Protocol (no implementation bundled):

```
class LLMClient(Protocol):
    name: str
    max_tokens: int

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> LLMResponse: ...

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[LLMResponse]: ...
```

Two concrete adapters planned (both optional imports to keep deps thin):

- `VLLMClient`     → local vLLM server (recommended default for full-scale rollout)
- `OpenAIClient`   → OpenAI-compatible endpoint (for small-N validation)

All LLM parameters routed through `main.py` (per project rule #9).

## 9. Embedder Interface (RAG layer)

Location: `coscope/data/embedding/embedder.py`.

```
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: List[str]) -> np.ndarray: ...  # shape (N, dim)
```

Concrete adapters:

- `BGEEmbedder`     → bge-m3 / bge-large-zh via sentence-transformers
- `OpenAIEmbedder`  → text-embedding-3-small

Used by:

- `WorkspaceBuilder` (flat corpus layer) — populate `MemoryEntry.embedding`
- `ArtifactRolloutEngine` (optional) — embed artifact content for downstream
  similarity-based rho (future rho_v4)

## 10. Episode Schema Impact

`Episode` dataclass fields: **unchanged**.

`memory_entries` composition changes:

```
Before:
  [workspace_global ...] + [workspace_hop_k ...] + [task_shared_oracle ...]
  + [agent_private placeholders ...] + [restricted (S4 only) ...]

After:
  [workspace_semantic_flat ...]                        (from WorkspaceBuilder, now flat)
  + artifact_trace.entries                              (from ArtifactRolloutEngine)
  + [restricted audit_reports for S4]                   (from RestrictedBuilder + rollout)
```

`meta`:

```
meta["rollout"] = {
    "engine":    "ArtifactRolloutEngine/v1",
    "llm":       {"name": "...", "temperature": 0.3, "seed": 42},
    "prompts":   {"got": "v18.3", "cot": "v18.4", "tot": "v18.5"},
    "rho_version": "v3"
}
```

## 11. Directory Additions

```
coscope/data/
├── rollout/                              (NEW)
│   ├── __init__.py
│   ├── llm_client.py                     # LLMClient Protocol + VLLMClient + OpenAIClient
│   ├── prompt_assembly.py                # assembles prompt inputs from GoTGraph + raw_item
│   ├── artifact_writer.py                # helper: (slot, node, content) -> MemoryEntry
│   ├── rollout_engine.py                 # ArtifactRolloutEngine
│   ├── trace_validator.py                # trace DAG invariants
│   └── fallback_synth.py                 # template-based fallback content
│
├── embedding/                            (NEW)
│   ├── __init__.py
│   ├── embedder.py                       # Embedder Protocol
│   ├── bge_embedder.py                   # BGE adapter
│   ├── openai_embedder.py                # OpenAI adapter
│   └── corpus_indexer.py                 # builds flat workspace index
│
└── memory/
    └── artifact_trace_builder.py         (NEW) # wraps ArtifactRolloutEngine for pipeline
```

## 12. Modifications to Existing Files

| File                                                 | Change                                                                         |
|------------------------------------------------------|--------------------------------------------------------------------------------|
| `data/memory/workspace_builder.py`                   | Remove hop-level emission; keep global only                                    |
| `data/memory/scope_ids.py`                           | Deprecate `workspace_semantic_hop`; add `session(ep)`                           |
| `data/memory/scope_ids.py` `SCOPE_LAYERS`            | Drop `workspace_semantic_hop`; add `session_message`, `session_tool`, `task_shared_plan`, `task_shared_artifact`, `restricted_audit` |
| `data/memory/task_shared_builder.py`                 | Moved into `rollout/fallback_synth.py::synth_task_shared_fallback`             |
| `data/memory/private_builder.py`                     | Deleted (replaced by rollout artifacts); remove from pipeline imports          |
| `data/pipeline/episode_builder.py`                   | Replace `task_shared_builder` + `private_builder` calls with `artifact_trace_builder.build(...)` |
| `data/got/rho_calculator.py`                         | Rewrite `_accessible_memory_ids_for_solver` per §6.1; bump cache `version` to `v3` |
| `data/agents/common.py::_scope_spec`                 | Drop hop workspace scope from `workspace_scopes`                               |
| `data/output/serializer.py`                          | No change (MemoryEntry round-trip already supports new metadata keys)          |
| `data/output/stats_reporter.py`                      | Add `rho_version` and `artifact_count_per_slot` to per-split stats             |
| `data/scripts/build_all.py`                          | New flags: `--llm-backend`, `--llm-model`, `--embed-backend`, `--embed-model`, `--rollout-temperature`, `--rollout-seed`, `--use-template-fallback` |
| `main.py`                                            | Route all new flags here (project rule #9)                                     |

## 13. Migration of Existing 43 GB Processed Data

Two options:

- **Option A — Full regenerate.** Rollout must happen anyway; drop
  `data/processed/got/*`, rerun `build_all.py` with new pipeline.
  Expected cost: 3–7 days on a single A100 with vLLM + 7B model.
- **Option B — In-place backfill.** Write `scripts/backfill_artifacts.py` that
  loads each shard, deletes `workspace_hop_k` + old private/shared entries,
  runs rollout for those episodes, appends artifacts, recomputes `rho`, then
  rewrites the shard. Preserves episode_id mapping and existing splits.

Recommendation: **Option B** for `dev` and `test` (preserves S4 split), **Option
A** for `train` (simpler, larger). Either way, shard file layout is unchanged.

## 14. Test Strategy

- `tests/rollout/test_rollout_engine.py` — mock LLM, check DAG invariants.
- `tests/rollout/test_trace_validator.py` — inject broken traces.
- `tests/memory/test_rho_v3.py` — sanity: LINEAR > FORK > INDEPENDENT.
- End-to-end smoke: run 10 MuSiQue episodes with mock LLM, assert all §5 invariants.

## 15. Open Questions

1. **Embedding inclusion in shards.** Embeddings can inflate shard size 5–10×.
   Proposal: write embeddings to a sidecar `{shard}.vec.npy` keyed by memory_id.
2. **Multi-turn dialogue artifacts.** Do we allow planner to produce a
   second plan after seeing first solver conclusion? v1: no; planner runs once.
3. **CoT/ToT execution.** v1 implements GoT rollout only; CoT uses LINEAR
   rollout with CoT prompts; ToT requires branch-aware prompt assembly —
   defer to `v1.1`.
4. **Rollout caching.** Keyed by `(episode_id, graph_type, reasoning_path_type, llm.name, seed)`
   — skip re-rollout on re-runs.

---

End of v1. Sign off required before code changes land.
