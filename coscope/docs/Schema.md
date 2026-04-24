# CoScope Data Schema 1.0.0

**Version:** `1.0.0` · **Date:** 2026-04-23 · **Status:** Frozen

> Schema version follows semver. `meta.schema_version` in every serialized
> Episode equals this value. Validate with `coscope/scripts/audit_jsonl.py`.

> Single reference for every object produced by the CoScope data-construction
> pipeline. Any schema change must be reflected here and in
> `coscope/core/artifact_types.py` (the three canonical mapping tables).

---

## 1. System Overview

The pipeline converts each raw QA/Math item into one **Episode** per
`(graph_type, reasoning_path_type)` combination.

```
raw_item  ──►  GoTGraph (topology)
          ──►  Agents  (one per node)
          ──►  ArtifactTrace  (LLM rollout → MemoryEntry list)
          ──►  rho  (structural overlap metric)
          ──►  GroundTruthEvidence  (for offline retrieval eval)
```

**Datasets:** MuSiQue · 2WikiMultiHopQA · HotpotQA · GSM8K · MATH

**Reasoning paths:** GoT (v1) · CoT (v1.1) · ToT (v1.2)

---

## 2. Core Enumerations

Values below are the **serialized JSON strings** (case-sensitive), not the
Python enum member names. Casing is not uniform across enums.

| Enum | Serialized values | Casing | Defined in |
|---|---|---|---|
| `ReasoningPathType` | `"GoT"` · `"CoT"` · `"ToT"` | mixed | `coscope/core/types.py` |
| `GraphType` | `"LINEAR"` · `"FORK"` · `"MERGE"` · `"FORK_MERGE"` · `"INDEPENDENT"` · `"POLICY_ISOLATED"` | UPPER | same |
| `SubsetLabel` | `"S1"` · `"S2"` · `"S3"` · `"S4"` | UPPER | same |
| `NodeType` | `"PLANNER"` · `"SOLVER"` · `"VERIFIER"` | UPPER | same |
| `MemoryType` | `"episodic"` · `"semantic"` · `"artifact"` · `"shared"` · `"working"` | lower | same |
| `AgentRole` | `"planner"` · `"solver"` · `"verifier"` · `"custom"` | lower | same |
| `VisibilityLevel` | `"owner"` · `"team"` · `"session"` · `"public"` · `"restricted"` | lower | same |
| `EdgeType` | `"DEPENDS_ON"` · `"PARALLEL"` · `"MERGE_INTO"` | UPPER | same |

> Note: `GoTNode.node_type` is serialized UPPER (`"PLANNER"`) while
> `GoTNode.agent_role` and `Agent.role` are serialized lower (`"planner"`).
> Both refer to the same role; the casing difference is historical and preserved for stability.

---

## 3. Artifact Slots

Every piece of agent-produced content is assigned to exactly one of **5 slots**.
This is the authoritative mapping (source: `artifact_types.py`):

| Slot | Producer | Step | scope\_layer | Scope | Memory type | Visibility |
|---|---|---|---|---|---|---|
| `PLAN` | Planner | decomposition | `task_shared_plan` | `task/{ep}/shared` | ARTIFACT | TEAM |
| `QUERY_INTENT` | any | Step 2 pre-retrieval reasoning | `agent_private_intent` | `task/{ep}/{node}/private` | EPISODIC | OWNER |
| `SCRATCH` | any | Step 5 post-retrieval draft | `agent_private` | `task/{ep}/{node}/private` | EPISODIC | OWNER |
| `CONCLUSION` | Solver | outward conclusion | `task_shared_artifact` | `task/{ep}/shared` | ARTIFACT | TEAM |
| `AUDIT_REPORT` | Verifier | audit (S4 only) | `restricted_audit` | `task/{ep}/restricted` | ARTIFACT | RESTRICTED |

**Emission rules:**

- Planner emits: `QUERY_INTENT + SCRATCH + PLAN` (3 artifacts)
- Each Solver emits: `QUERY_INTENT + SCRATCH + CONCLUSION` (3 artifacts)
- Verifier emits: `QUERY_INTENT + SCRATCH + AUDIT_REPORT` (3 artifacts, S4 only)
- Session slots (`MESSAGE`, `TOOL_CALL`) are reserved, not emitted in v1

**Artifact count per episode:**

| Graph type | Solvers | Artifacts |
|---|---|---|
| LINEAR 2-hop | 2 | 9 |
| FORK | 2 | 9 |
| FORK\_MERGE | 3 | 12 |
| INDEPENDENT | 4 | 15 |
| POLICY\_ISOLATED | 2 | 12 |

---

## 4. Memory Scope Layout

Four physical scope regions per episode (helpers in `coscope/core/scope_ids.py`):

| Region | scope\_id pattern | Who reads | Content |
|---|---|---|---|
| Workspace | `workspace/{dataset}/semantic` | All agents (flat) | Corpus paragraphs / formulas |
| Task-shared | `task/{ep}/shared` | All agents | PLAN + CONCLUSIONs |
| Agent-private | `task/{ep}/{node}/private` | Owner only | QUERY\_INTENT + SCRATCH |
| Restricted | `task/{ep}/restricted` | Verifier (clearance=3) | AUDIT\_REPORT |

> The workspace is **flat** (v1): a single corpus scope per dataset, accessible
> equally by all roles. Per-hop partitioning has been removed.

---

## 5. Episode Object

Defined in `coscope/core/types.py`.

```
Episode
├── episode_id          str
├── dataset             "musique" | "2wikimhqa" | "hotpotqa" | "gsm8k" | "math"
├── split               "train" | "dev" | "test"
├── original_id         str
├── hop_count           int
├── graph_type          GraphType
├── reasoning_path_type ReasoningPathType
│
├── rho                 float       structural overlap (0–1)
├── rho_subset          S1 | S2 | S3
├── policy_conflict     bool
├── s4_eligible         bool
│
├── question            str
├── answer              str
│
├── got_graph           GoTGraph    nodes + edges + graph_type
├── agents              List[Agent]
├── memory_entries      List[MemoryEntry]   workspace entries + artifact trace
├── retrieval_requests  List[RetrievalRequest]
├── ground_truth        List[GroundTruthEvidence]
└── meta                Dict        schema_version, seed, rho_version, rollout info
```

**`meta` required keys:**

| Key | Value |
|---|---|
| `schema_version` | `"1.0.0"` (semantic version; bump major on breaking changes) |
| `rho_version` | `"v3"` |
| `seed` | int |
| `created_at` | ISO 8601 with timezone |
| `rollout` | `{engine, llm: {name}, seed, reasoning_path_type, n_entries}` |
| `reasoning_path_type` | `"GoT"` (mirrors top-level field; for quick filtering) |

---

## 6. MemoryEntry Metadata Convention

All artifact entries (non-workspace) carry these mandatory metadata keys:

| Key | Type | Description |
|---|---|---|
| `slot` | str | one of `ArtifactSlot.value` |
| `scope_layer` | str | from the canonical mapping table (§3) |
| `source_node_id` | str | producing graph node |
| `hop_index` | int \| null | null for PLAN and AUDIT\_REPORT |
| `topo_index` | int | emission order within episode |
| `parent_artifact_ids` | list[str] | upstream artifacts this entry depends on |
| `is_gold_evidence` | bool | True when content derived from gold oracle |
| `required_clearance` | int | 0–3 (3 for AUDIT\_REPORT) |
| `dataset` | str | |
| `episode_id` | str | |

**Provenance:**

| Field | Value |
|---|---|
| `agent_id` | `"{episode_id}_{node_id}"` |
| `timestamp` | `2025-01-01T00:00:00Z + topo_index seconds` (deterministic, reproducible) |
| `source` | `"llm:{model}"` or `"template:v1"` (fallback) |
| `version` | `1` |

---

## 7. GoTGraph Structure

```
GoTGraph
├── graph_type   GraphType
├── nodes        List[GoTNode]
│   └── GoTNode:
│         node_id, node_type, agent_role, hop_index,
│         parent_node_ids, child_node_ids,
│         produced_artifact_ids   (List[str])
│         produced_by_slot        (Dict[slot -> memory_id])
│         produced_artifacts      (List[MemoryEntry])
└── edges        List[GoTEdge]
    └── GoTEdge: from_node, to_node, edge_type
```

**Per-node artifact linkage.** Each serialized `GoTNode` carries, in addition
to its topology, three convenience fields populated at serialization time:

| Field | Type | Description |
|---|---|---|
| `produced_artifact_ids` | `List[str]` | All `memory_id`s whose `metadata.source_node_id == this.node_id` AND `metadata.slot` is set (i.e. the artifacts this node produced). |
| `produced_by_slot` | `Dict[str, str]` | `{slot -> memory_id}` mapping, e.g. `{"plan": "mem_...", "query_intent": "mem_...", "scratch": "mem_..."}`. One entry per slot this node emitted. |
| `produced_artifacts` | `List[MemoryEntry]` | Full memory entries for all artifacts this node produced. Includes complete content, metadata, and provenance. |

Consumers can therefore fetch a node's full output directly without separate lookups:

```python
node = ep["got_graph"]["nodes"][i]
# Direct access to full artifact content
for artifact in node["produced_artifacts"]:
    print(f"Slot: {artifact['metadata']['slot']}")
    print(f"Content: {artifact['content']}")
```

Expected slot sets:

| `node_type` | Slots in `produced_by_slot` |
|---|---|
| `PLANNER` | `{query_intent, scratch, plan}` |
| `SOLVER` | `{query_intent, scratch, conclusion}` |
| `VERIFIER` | `{query_intent, scratch, audit_report}` (only in POLICY_ISOLATED) |

---

## 8. Retrieval Request and Agent Config

Each agent node generates exactly one `RetrievalRequest`. The scope it sees:

```
ScopeSpec(
    private_scopes   = [task/{ep}/{node}/private],
    shared_scopes    = [task/{ep}/shared],
    workspace_scopes = [workspace/{dataset}/semantic],
    governed_scopes  = [task/{ep}/restricted]   # verifier only
)
```

`PolicyConstraints` controls visibility level, clearance ceiling, and
merge eligibility for CoScope's shared first-stage retrieval.

---

## 9. Ground Truth Evidence

Used by offline evaluation to compute Recall@k and coverage per agent.

```
GroundTruthEvidence
├── memory_id              str
├── scope_layer            str        see §3
├── hop_index              int | null
├── required_by_agent_ids  List[str]
└── evidence_type          "shared_required" | "private_required"
```

---

## 10. ρ (rho) — Structural Overlap Metric

**Formula** (implemented in `coscope/rollout/rho_v3.py`):

```
For each Solver i:
    A_i = { plan.memory_id }
        ∪ { conclusion.memory_id | producer ∈ ancestors_or_self(i) }

ρ = mean_{i < j}  |A_i ∩ A_j| / |A_i ∪ A_j|
```

Workspace entries and private artifacts (QUERY\_INTENT, SCRATCH) are excluded —
they do not differentiate structural information sharing.

**Empirical distribution** (MuSiQue dev, 3 items × 5 graph types, DashScope qwen-plus):

| Graph type | ρ | Subset |
|---|---|---|
| FORK | 0.333 | S3 |
| INDEPENDENT | 0.394 | S3 |
| FORK\_MERGE | 0.444 | S2 |
| LINEAR (2-hop) | 0.667 | S1 |
| POLICY\_ISOLATED | 0.667 | S1 |

**Subset thresholds** (`coscope/config/yaml/subset_thresholds.yaml`):

| Subset | Condition | Captures |
|---|---|---|
| S1 | ρ > 0.6 | HIGH overlap — LINEAR, POLICY\_ISOLATED |
| S2 | 0.4 < ρ ≤ 0.6 | MEDIUM overlap — FORK\_MERGE |
| S3 | ρ ≤ 0.4 | LOW overlap — FORK, INDEPENDENT |
| S4 | policy\_conflict flag | Orthogonal to rho; policy-isolated episodes |

---

## 11. Trace Validation Invariants

Two layers of validation:

**11.1 Build-time (`coscope/rollout/trace_validator.validate_trace`)** — 8 hard rules enforced during construction. Episodes failing any rule are dropped.

| # | Rule | Level |
|---|---|---|
| 1 | All `memory_id` values are globally unique within the trace | error |
| 2 | Every entry's `metadata.slot` is a valid `ArtifactSlot` | error |
| 3 | `stream` is monotonically non-decreasing by `topo_index` | error |
| 4 | Every `memory_id` in `stream` exists in `entries` | error |
| 5 | No `task_shared_*` entry carries `owner` visibility | **error** |
| 6 | All ids in `parent_artifact_ids` resolve to real entries | error |
| 7 | Each Solver's CONCLUSION references graph-parent CONCLUSIONs in `parent_artifact_ids` | warning |
| 8 | AUDIT\_REPORT present ↔ `graph_type == POLICY_ISOLATED` | error |

**11.2 Post-serialization audit (`coscope/scripts/audit_jsonl.py`)** — 20 schema-wide invariants for end-to-end contract checking of finalized JSONL shards. Run before handing off any dataset:

```bash
python -m coscope.scripts.audit_jsonl coscope/data/processed/got
```

The auditor covers, in addition to the 8 build-time rules above:

- enum casing (top-level + nested)
- `hop_count` vs. max solver `hop_index` consistency
- `policy_conflict` ↔ `POLICY_ISOLATED`
- memory_id uniqueness across shards (not just within episode)
- required metadata keys on every artifact entry (10 keys)
- `produced_by_slot` ↔ `produced_artifact_ids` consistency
- `topo_index` is int; `required_clearance` in `{0,1,2,3}`
- `ground_truth.memory_id` resolvability
- `created_at` is ISO 8601
- `meta.rollout.llm` present
- cross-shard `(original_id, question, answer)` consistency

Exit code: `0` if clean, `2` if any issue.

---

## 12. Serialized JSONL Layout

One JSON line per Episode. Shards at:

```
coscope/data/processed/{reasoning_path}/{dataset}/{split}/{subset}_{graph_type}.jsonl

e.g.  coscope/data/processed/got/musique/train/s1_linear.jsonl
      coscope/data/processed/got/hotpotqa/test/s4_policy_isolated.jsonl
```

Top-level keys:

```json
{
  "episode_id": "...",
  "dataset": "musique",
  "split": "train",
  "original_id": "...",
  "hop_count": 2,
  "graph_type": "LINEAR",
  "reasoning_path_type": "GoT",
  "rho": 0.667,
  "rho_subset": "S1",
  "policy_conflict": false,
  "s4_eligible": false,
  "question": "...",
  "answer": "...",
  "got_graph": {
    "graph_type": "LINEAR",
    "nodes": [
      {
        "node_id": "solver_1", "node_type": "SOLVER", "agent_role": "solver",
        "hop_index": 1,
        "parent_node_ids": ["planner"], "child_node_ids": ["solver_2"],
        "produced_artifact_ids": ["mem_...", "mem_...", "mem_..."],
        "produced_by_slot": {
          "query_intent": "mem_...",
          "scratch":      "mem_...",
          "conclusion":   "mem_..."
        }
      }
    ],
    "edges": [
      { "from_node": "planner", "to_node": "solver_1", "edge_type": "DEPENDS_ON" }
    ]
  },
  "agents": [...],
  "memory_entries": [...],
  "retrieval_requests": [...],
  "ground_truth": [...],
  "meta": {
    "schema_version": "1.0.0",
    "rho_version": "v3",
    "seed": 42,
    "created_at": "2026-04-23T14:49:39.937018+00:00",
    "reasoning_path_type": "GoT",
    "rollout": {
      "engine": "ArtifactRolloutEngine/v1",
      "llm": { "name": "dashscope:qwen-plus" },
      "seed": 42,
      "reasoning_path_type": "GoT",
      "n_entries": 9
    }
  }
}
```

---

## 13. LLM and Embedder Interfaces

### LLMClient (`coscope/core/interfaces.py`)

Protocol with `generate(prompt, *, temperature, top_p, seed, max_new_tokens) -> LLMResponse`
and `generate_batch`. Returns `LLMResponse(text, prompt_tokens, completion_tokens,
finish_reason, model, latency_ms)`.

- **Available:** `TemplateLLMClient` — deterministic template fallback (`coscope/rollout/template_llm_client.py`)
- **Available:** `DashScopeClient` — Qwen via OpenAI-compatible API, default `qwen-plus` (`coscope/rollout/dashscope_client.py`)

### Embedder (`coscope/core/interfaces.py`)

Protocol with `embed(text) -> np.ndarray` and `embed_batch(texts) -> EmbeddingResult`
(shape `(N, dim)`).

- **Available:** `DashScopeEmbedder` — `text-embedding-v3`, default `dim=1024` (`coscope/embedding/dashscope_embedder.py`)

---

## 14. Schema Versioning

| Field | Meaning |
|---|---|
| `meta.schema_version` | Episode-level schema semantic version (`"1.0.0"` = this document). MAJOR bumps on breaking changes. |
| `meta.rho_version` | ρ formula version (`"v3"` = this document) |

All readers must check both fields. Schema `1.x.x` guarantees backward-compatible additions only; any breaking change (renamed/removed field, changed enum value) bumps to `2.0.0`.

**1.0.0 vs pre-v1 changes:**

- **Added:** `QUERY_INTENT` slot; `metadata.slot`, `parent_artifact_ids`, `topo_index`; scope layers `task_shared_plan`, `task_shared_artifact`, `agent_private_intent`, `restricted_audit`
- **Removed:** `workspace_semantic_hop` scope layer; agent\_private placeholder entries; oracle-filled `task_shared_episodic` entries
- **Kept:** All Episode top-level fields; GoTGraph / Agent / RetrievalRequest / GroundTruthEvidence structure; `task_restricted` layer
