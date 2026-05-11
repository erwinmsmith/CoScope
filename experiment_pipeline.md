# CoScope Experiment Pipeline

> End-to-end description of how raw QA data becomes the numbers in
> `Experiment Result Tables — Scope-Overlap Collaborative Retrieval.md`.
> Companion doc: `coscope流程.md` (runtime per-round agent flow) describes
> what happens *inside* an episode; this doc describes how episodes are
> *selected, built and evaluated* at the experiment level.

---

## 0. Pipeline at a glance

```
              raw QA file                       config/yaml/dataset_config.yaml
                   │                                       │
                   ▼                                       │
          dataio.loaders.<ds>_loader                       │
                   │   (1 raw item = {original_id,         │
                   │    question, answer,                  │
                   │    supporting_paragraphs,             │
                   │    distractor_paragraphs, ...})       │
                   ▼                                       │
         scripts.build_all                                 │
           per (dataset × rpt × graph_type)                │
           ├─ memory.builders.workspace_builder            │
           ├─ memory.builders.task_shared_builder          │
           ├─ memory.builders.agent_private_builder        │
           ├─ memory.builders.restricted_builder ◄─ data/interim/restricted/
           ├─ graph.{got,cot,tot}.<rpt>_builder            │   {ds}_restricted.jsonl
           ├─ rho_calculator                  ─► ρ ∈ [0,1] │   (offline, scripts.generate_restricted)
           └─ subset_assignment  (config/yaml/             │
                                  subset_thresholds.yaml)  │
                   │                                       │
                   ▼                                       │
         shards on disk:                                   │
         data/processed/{rpt}/{ds}/test/                   │
           s1_*.jsonl  s2_*.jsonl  s3_*.jsonl              │
           s4_policy_isolated.jsonl  (if --include-s4)     │
                   │                                       │
                   ▼                                       ▼
         scripts.eval_jsonl ───────── DashScope text-embedding-v3
           per variant ∈ {a1..a8}     (data/cache/embeddings/)
                   │
                   ▼
         data/processed/{rpt}/{ds}/test/
           eval_v9_qwen_full_k{5,10,20}.json
         data/processed/{rpt}/{ds}/test_qi/
           eval_v9_qwen_a8_k{5,10,20}.json   (Stage B, after
                                              scripts.generate_query_intent)
```

---

## 1. Dataset selection

Five datasets, configured in `config/yaml/dataset_config.yaml`. For
each we use the **dev split** (loader maps `test → dev` because public
test labels are unavailable):

| Dataset | Raw file | task_shared_mode | Episodes/test |
| ------- | -------- | ---------------- | ------------: |
| MuSiQue | `musique_ans_v1.0_dev.jsonl` | realtime  | 12 085 (GoT), 4 834 (ToT) |
| 2WikiMultiHopQA | `dev.json` | realtime  | 62 880 (GoT), 25 152 (ToT) |
| HotpotQA | `distractor_validation.parquet` | realtime | 37 025 (GoT), 14 810 (ToT) |
| GSM8K | `test.parquet` | oracle | (not yet run end-to-end) |
| MATH  | `test.parquet` | oracle | (not yet run end-to-end) |

* **realtime** vs **oracle** controls how `task_shared` memories are
  populated: realtime fills them from intermediate agent outputs;
  oracle pre-fills from ground-truth steps (used for math).
* Episode counts differ between GoT and ToT because each `(rpt, raw_item)`
  pair instantiates a different topology and a different number of agents
  (see §3).

---

## 2. Per-item processing (one raw item → up to 2 episodes)

`scripts.build_all --datasets <ds> --splits test --reasoning-path-type {got|cot|tot}`
drives this loop. For each raw item it produces **one core episode** under
the chosen `graph_type` and **optionally one S4 episode** (`--include-s4`).

### 2.1 Memory population

Four scope buckets are populated per episode:

| Bucket | Builder | What goes in |
| ------ | ------- | ------------ |
| `workspace` (`mem_ws_*`) | `WorkspaceBuilder` | All supporting + distractor paragraphs (visible to every agent). IDs are **rpt-stable** (cross-RPT MD5 hash over normalized text), so embeddings cached during GoT runs are reused unchanged for ToT/CoT runs. |
| `task_shared` (`mem_ts_*`) | `TaskSharedBuilder` | Per-hop intermediate facts. In `realtime` mode these are seeded with conclusion placeholders that the rollout later overwrites; in `oracle` mode they are populated directly from gold solution steps. |
| `agent_private` (`mem_pr_*`) | `AgentPrivateBuilder` | Per-agent scratch (notes, draft answers). Always episode-scoped. |
| `restricted` (`mem_rs_*`) | `RestrictedBuilder` | **Privacy-sensitive verifier-only memory**. Loaded from `data/interim/restricted/{ds}_restricted.jsonl` if present; otherwise the bucket is empty (this is why 2Wiki / Hotpot S4 currently has zero `mem_rs_*` entries — see §1f.2 / §1g.2 / Table 6 caveat in the results doc). |

Restricted entries are generated **offline**, before any build, by:

```bash
python -m scripts.generate_restricted \
    --dataset {ds} --split test \
    --data-dir data/raw/{ds}
```

Each line of `{ds}_restricted.jsonl` is one record per `original_id`:

```json
{"original_id": "2hop__123", "entries": [
    {"content": "para_005 credibility: 1.0 (gold evidence)", "confidence": 1.0,
     "source": "para_005", "metadata": {"source_paragraph_id": "para_005", "kind": "gold_fact"}},
    {"content": "para_010 credibility: 0.0 (distractor)", "confidence": 0.0,
     "source": "para_010", "metadata": {"source_paragraph_id": "para_010", "kind": "distractor"}},
    ...
]}
```

### 2.2 Graph topology

`reasoning_path_type` picks one of three builders:

* **GoT** (`graph.got.graph_templates`): 5 graph types in a fixed mixture
  (LINEAR, FORK, FORK_MERGE, INDEPENDENT, POLICY_ISOLATED). Each raw
  item becomes one core episode under one graph_type drawn by seed.
* **ToT** (`graph.tot.tree_builder`): one FORK tree per raw item
  (one Planner → `hop_count` parallel Solvers, no merge node). 4-hop
  items produce 4 fan-out branches.
* **CoT** (`graph.cot.chain_builder`): a single linear chain
  (Planner → Solver_1 → ... → Solver_n). One episode per raw item.

For all three rpts, **if `--include-s4` is set**, an additional
`POLICY_ISOLATED` episode is produced with the same agents plus a
Verifier that holds the restricted bucket.

### 2.3 ρ and subset assignment

`graph/got/rho_calculator.py` computes ρ = pairwise mean IoU over the
scope-id sets of every (solver, solver) pair. Then
`config/yaml/subset_thresholds.yaml` partitions:

```
s1_threshold = 0.3   →  S1 if ρ > 0.3       (LINEAR, POLICY_ISOLATED branches)
s3_upper     = 0.1   →  S3 if ρ ≤ 0.1       (FORK)
                         S2 otherwise        (FORK_MERGE, INDEPENDENT)
S4 is orthogonal:        s4_eligible=True    (only for POLICY_ISOLATED)
```

> **Caveat (also noted in §X.2):** under ToT FORK, ρ is a deterministic
> function of `(graph_type, hop_count)` with std = 0 inside each hop
> bucket. So ρ in ToT is effectively a hop-count proxy, not a
> content-overlap measurement. This is a topology proxy and the paper
> narrative should call it that.

### 2.4 Shard layout

Build output is **one JSONL shard per subset** under the canonical path:

```
data/processed/{rpt}/{ds}/test/
    s1_linear.jsonl                    (GoT only)
    s1_fork.jsonl                      (ToT, S1-FORK = high-ρ FORK)
    s1_policy_isolated.jsonl           (GoT/ToT non-S4 sliced from S4 shard)
    s2_fork_merge.jsonl                (GoT only)
    s2_independent.jsonl               (GoT only)
    s3_fork.jsonl
    s4_policy_isolated.jsonl           (with restricted bucket)
```

Each line is a full `Episode` object (serialized via
`serialization.serializer.Serializer`): agents, memory entries,
retrieval requests, GoTGraph, ρ, rho_subset, original_id, etc.

---

## 3. Stage A — retrieval-only evaluation (no LLM)

```bash
DATASET=musique RPT=tot bash scripts/run_stage_a_rpt.sh
# or directly:
python -m scripts.eval_jsonl \
    --shards data/processed/tot/musique/test/s*.jsonl \
    --variants a1 a2 a3 a4 a4_nofb a4_norerank a5 a5_noproj a5_norerank a6 a7 \
    --k 10 \
    --embedder dashscope --dashscope-model text-embedding-v3 \
    --dashscope-dim 1024 \
    --cache-dir data/cache/embeddings \
    --output .../eval_v9_qwen_full_k10.json
```

* The eval pass reads each shard, **replays the retrieval pipeline
  per variant** (see §0 in the results doc for the 11 variant
  definitions), and writes a single JSON with per-(variant, subset)
  cells.
* **Embedding cache** at `data/cache/embeddings/` is a SQLite store
  keyed by `(embedder, model, dim, text_hash)`. Workspace memory
  embeddings are populated once on the first cold run; later
  (rpt, k) combinations reuse them. Only fresh **query strings** —
  the per-agent retrieval queries derived from the topology — trigger
  new embedding API calls.
* `PYTHONHASHSEED=0` is enforced inside the driver. Residual
  reproducibility noise on S4 (≤ 0.5 pt) comes from top-k tie-breaking
  among equally-scored memories near the cutoff.

### 3.1 Metrics emitted

Per (variant, subset) cell — full schema is in §0.5 of the results doc:

| Metric | Source |
| ------ | ------ |
| `recall_at_k`         | retriever output vs `gold_memory_ids` of the request |
| `mrr_at_k`            | reciprocal rank of first gold hit |
| `false_merge_rate` (**FMR**)            | fraction of S4 requests that touched any shared bucket |
| `content_false_merge_rate` (**cFMR**)   | fraction of S4 requests where a `mem_rs_*` actually appeared in top-k |
| `first_stage_savings` | 1 − (`#unique_first_stage_pulls` / `#per_agent_independent_pulls`) |

---

## 4. Stage B — LLM-rewritten query intent (A8)

Only the `a8` variant calls an LLM. It is split into a separate
**pre-pass** so that the LLM cost is paid once and then `a8` can be
re-evaluated as many times as we want (e.g. at different k).

```bash
DATASET=musique RPT=tot bash scripts/run_stage_b_dataset.sh
# under the hood, this runs:
python -m scripts.generate_query_intent \
    --shards data/processed/tot/musique/test/s*.jsonl \
    --dataset musique --split test \
    --llm dashscope --model qwen-plus \
    --output-dir data/processed/tot/musique/test_qi \
    --workers 32 --batch-size 32
# then eval_jsonl over the new shards with variants {a6, a7, a8}.
```

* For every `RetrievalRequest` in every episode, we ask qwen-plus to
  produce a `query_intent` string (~one short paragraph) given
  `(role, sub_question, ancestor_conclusions)`. This string overwrites
  `request.metadata["query_intent"]`.
* The downstream `eval_jsonl --variants a8 a6 a7` then reuses the
  exact same first-stage candidate set as A6 but reranks the top-k
  by `cosine(embed(query_intent), embed(memory))`.
* A8 cost is roughly **one qwen-plus call per retrieval request**
  (≈ 5 calls / episode). Empirically: MuSiQue ToT 4 834 ep takes
  ~ 20 min @ 32 workers, fallback rate < 0.1 %.

---

## 5. Reproducibility checklist

| Knob | Value used in the results doc |
| ---- | ------------------------------- |
| Random seed                 | `PYTHONHASHSEED=0`, build seed=42 |
| Embedder                    | DashScope `text-embedding-v3` (dim 1024) |
| Embedding cache             | `data/cache/embeddings/` (SQLite) |
| Query-intent LLM            | qwen-plus via DashScope, 32 workers |
| Subset thresholds           | `config/yaml/subset_thresholds.yaml` (s1=0.3, s3=0.1) |
| Restricted interim          | `data/interim/restricted/{ds}_restricted.jsonl` (MuSiQue only at the moment; see §X.2 of results doc) |
| k cutoffs                   | 5, 10, 20 (all three populated for GoT × 3 datasets and ToT × 3 datasets) |
| Build CLI                   | `scripts/build_all.py` |
| Eval CLI                    | `scripts/eval_jsonl.py` |
| Stage A driver              | `scripts/run_stage_a_rpt.sh` |
| Stage B driver              | `scripts/run_stage_b_dataset.sh` |
| k=5/k=20 sweep              | `scripts/eval_k5_k20_sweep.sh` (cache-warm, no LLM) |

---

## 6. What is NOT yet built

(Mirror of §X.2 in the results doc, restated as engineering tasks.)

* `data/interim/restricted/{2wikimhqa,hotpotqa}_restricted.jsonl`
  — needed before cross-dataset cFMR becomes a real privacy claim
  (currently mechanically 0 on those two datasets).
* End-to-end answer generation (`Tables 1–5` downstream EM/F1/Acc).
  All current results are retrieval-only; rolling out the actual
  agent loop with an LLM and computing EM/F1 is a separate run.
* External baselines (BM25 / DPR / ColBERT) — slots exist in
  Tables 2–5 but are unpopulated.
