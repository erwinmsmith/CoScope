# Experiment Result Tables — Scope-Overlap Collaborative Retrieval

This is the **single source of truth** for variant definitions, metric
definitions, and result placeholders. Every table below assumes the lock-down
in §0 and §0.5; do not introduce ad-hoc variant aliases elsewhere.

Authoritative implementation reference: `retrieval/pipeline.py::CoScopeRetrievalPipeline.retrieve`.

---

## §0 · Variant Definitions (LOCKED to code, 2026-04-29)

Each variant is fully specified by 6 axes. **Default reasoning structure = GoT.**
For the same variant on CoT / ToT, suffix as `Ax-CoT` / `Ax-ToT` (see §0.5
and Table 15).

| ID | Routing | Bucket aggregation | Rerank | Private fallback | Projection | Query text |
|---|---|---|---|---|---|---|
| **A1** | independent (per request) | — (no buckets) | yes | no | — | raw sub-question |
| **A2** | force-merge (1 global pool, no scope/policy filtering) | — | **no** | no | — | raw |
| **A3** | scope-only buckets | mean of bucket queries | **no** | **no** | — | raw |
| **A4** | scope-only buckets | mean | **yes** | **yes** | — | raw |
| A4_nofb | scope-only buckets | mean | yes | **no** | — | raw |
| A4_norerank | scope-only buckets | mean | **no** | yes | — | raw |
| **A5** | scope-only buckets | SVD-projected query matrix | yes | yes | **truncated SVD** | raw |
| A5_noproj | scope-only buckets | identity projection (≡ A4 in projection space) | yes | yes | identity | raw |
| A5_norerank | scope-only buckets | SVD-projected | **no** | yes | truncated SVD | raw |
| **A6** | **hierarchical block** (policy-aware) | SVD-projected | yes | yes | truncated SVD | raw |
| **A7** | hierarchical block | SVD-projected | yes | yes | truncated SVD | raw (alias of A6 today) |
| **A8** | hierarchical block | SVD-projected | yes | yes | truncated SVD | **LLM-rewritten `query_intent`** |

**Ablation chain (corrected to match code):**

```
A1  ──[scope bucketing]──▶  A3
A3  ──[+rerank +fallback]──▶  A4
A4  ──[+SVD warm-start]──▶  A5
A5  ──[+hierarchical/policy-aware routing]──▶  A6
A6  ──[Step-2 LLM query rewrite]──▶  A8        (A7 is a code-level alias of A6)
```

**Footnotes (be explicit about gaps):**

* **A2** is unsafe by construction; reported only as the upper-bound on
  S4 false-merge / content-leak rate. Never recommend A2 in deployment.
* **A6 vs A5** — A6 = A5 + **block routing** (per design `coscope流程.md`
  / `introduction.md §14.4.1`). Code: A6 uses `HierarchicalRouter.route`
  while A5 uses `_route_scope_only`. Both share the SVD projection.
  This matches the design exactly; there is **no "learned-W" component
  in the design**, so do not introduce one unless the spec changes.
* **A7** is **A6 with Step 2 disabled** (i.e. raw question as query,
  no LLM rewrite). In code today A7 silently aliases A6 because Step 2
  is gated by `request.metadata["query_intent"]`: when that field is
  absent (default), A6 already runs on the raw question. So as long as
  query_intent has *not* been generated for a shard, A7 ≡ A6
  numerically. A7 only becomes meaningfully different from A8 once
  query_intent is populated (Stage B).
* **A8** requires an offline Step-2 pass (`scripts/generate_query_intent.py`)
  to populate `request.metadata["query_intent"]`. Without that,
  A8 falls back to A6 silently.

---

## §0.5 · Metric Definitions (LOCKED)

**Retrieval-only metrics (computed by `scripts/eval_jsonl.py`, no LLM):**

| Metric | Definition | Where reported |
|---|---|---|
| Recall@k | mean over requests: `|topk ∩ gold| / |gold|`. **k schedule: report 5 / 10 / 20**; headline tables use k=10. k>20 is omitted (most requests have ≤5 gold memories). | Tables 1a, 9, 10 |
| MRR@k | mean reciprocal rank of first hit in top-k (0 if absent). Same k schedule. | Tables 1a, 10 |
| FMR (S4) | False Merge Rate: fraction of S4 verifier requests whose top-k contains *any* result drawn from a shared bucket | Tables 1, 6 |
| **cFMR (S4)** | content-FMR: top-k actually contains a `restricted` memory not authorized for the requesting agent. **This is the real privacy leak metric.** | Tables 1, 6 |
| First-stage savings | `1 − (actual first-stage retrievals) / (independent baseline first-stage retrievals)` | Table 9 |

**End-to-end metrics (require LLM solver pass; not yet wired in `eval_jsonl`):**

| Metric | Datasets | Definition |
|---|---|---|
| EM | MuSiQue, 2WikiMultiHopQA, HotpotQA | Exact match between generated answer and gold answer (after normalization) |
| F1 | MuSiQue, 2WikiMultiHopQA, HotpotQA | Token-level F1 between generated and gold answer |
| Acc | GSM8K, MATH | Final-numeric / final-expression match |

> **Status note (2026-04-29):**
>
> 1. **Embedder lock-down.** Paper main results MUST use
>    DashScope `text-embedding-v3` (dim=1024) per `coscope流程.md`.
>    Earlier `eval_v8_st_full.json` (sentence-transformers MiniLM,
>    dim=384) is kept ONLY as a fast offline development preview and
>    must NOT be cited as the paper number. A Qwen re-run is queued.
>
> 2. **Mode A (end-to-end EM/F1/Acc) policy.** Mode A will NOT be run
>    on the full 12085 dev set. Instead we will sample a **stratified
>    subset of 200–500 episodes** (balanced over S1/S2/S3/S4) and run
>    the full Step 1–6 loop on it. Numbers reported in Tables 1–5
>    EM/F1/Acc columns are explicitly understood to come from this
>    subset. The cell convention below applies:
>    - Empty cell → not yet run.
>    - `*` → planned to be run via Mode A subset only.
>    - Numbers with footnote `(N=…, Qwen)` → final paper figure.

---

## Table 1 · Main Results — MuSiQue (EM / F1)

> Metrics: EM = Exact Match, F1 = token-level F1. S4 reports cFMR (privacy
> leak). Cells marked `*` await end-to-end LLM pipeline. See **Table 1a**
> for retrieval-only numbers we already have.

| **Method**  | **S1 EM** | **S1 F1** | **S2 EM** | **S2 F1** | **S3 EM** | **S3 F1** | **S4 FMR** |
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------- |
| BM25        |           |           |           |           |           |           |            |
| DPR         |           |           |           |           |           |           |            |
| ColBERT     |           |           |           |           |           |           |            |
| Iter-RetGen |           |           |           |           |           |           |            |
| HippoRAG    |           |           |           |           |           |           |            |
| MemWalker   |           |           |           |           |           |           |            |
| MemMA       |           |           |           |           |           |           |            |
| **A1**      |           |           |           |           |           |           |            |
| **A2**      |           |           |           |           |           |           |            |
| **A3**      |           |           |           |           |           |           |            |
| **A4**      |           |           |           |           |           |           |            |
| **A5**      |           |           |           |           |           |           |            |
| **A6**      |           |           |           |           |           |           |            |
| **A7**      |           |           |           |           |           |           |            |
| **A8**      |           |           |           |           |           |           |            |

---

## Table 1a · Retrieval Quality — MuSiQue (12085 ep, GoT, k=10)  ⚠ MiniLM PREVIEW

> Source: `data/processed/got/musique/test/eval_v8_st_full.json`
> (sentence-transformers `all-MiniLM-L6-v2`, dim=384, PYTHONHASHSEED=0,
> 71 min wall on full 12085 ep). N per cell:
> S1=4744, S2=3672, S3=1252, S4=2417.
>
> **These numbers are an offline development preview. They will be
> superseded by Table 1b (Qwen text-embedding-v3) for the paper.**
> Conclusions on variant ranking (A4 ≫ A6, SVD = no-op, A4_norerank
> drops 30 pt) are expected to carry over directionally; absolute
> values will shift.

| **Variant** | **S1 R@10** | **S2 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR@10** | **S2 MRR@10** | **S3 MRR@10** | **S4 MRR@10** | **S4 cFMR** | **Savings (S2)** |
| ----------- | ----------: | ----------: | ----------: | ----------: | ------------: | ------------: | ------------: | ------------: | ----------: | ---------------: |
| **A1** | 0.919 | 0.906 | 0.963 | 0.731 | 0.917 | 0.880 | 0.886 | 0.724 | 0.000 | 0% |
| A2 | 0.650 | 0.684 | 0.757 | 0.668 | 0.717 | 0.726 | 0.762 | 0.719 | **0.341** | 80% |
| **A3** | 0.662 | 0.692 | 0.761 | 0.719 | 0.725 | 0.731 | 0.762 | 0.736 | 0.000 | 80% |
| **A4** | **0.972** | **0.956** | 0.964 | **0.918** | **0.928** | **0.895** | 0.875 | **0.814** | 0.000 | 80% |
| A4_nofb | 0.972 | 0.956 | 0.964 | 0.917 | 0.928 | 0.895 | 0.875 | 0.814 | 0.000 | 80% |
| A4_norerank | 0.662 | 0.692 | 0.761 | 0.719 | 0.725 | 0.731 | 0.762 | 0.736 | 0.000 | 80% |
| **A5** | 0.934 | 0.922 | 0.965 | 0.810 | 0.922 | 0.887 | **0.889** | 0.753 | 0.000 | 80% |
| A5_noproj | 0.926 | 0.913 | 0.963 | 0.827 | 0.920 | 0.884 | 0.889 | 0.759 | 0.000 | 80% |
| A5_norerank | 0.456 | 0.485 | 0.592 | 0.482 | 0.474 | 0.430 | 0.385 | 0.401 | 0.000 | 80% |
| **A6** | 0.937 | 0.927 | 0.963 | 0.739 | 0.922 | 0.887 | 0.888 | 0.726 | 0.000 | 61% |
| **A7** | 0.937 | 0.927 | 0.963 | 0.739 | 0.922 | 0.887 | 0.888 | 0.726 | 0.000 | 61% |
| **A8** | (pending) | | | | | | | | | |

---

## Table 1b · Retrieval Quality — MuSiQue (12085 ep, GoT, **Qwen text-embedding-v3**) — PAPER MAIN

> **Source:** `data/processed/got/musique/test/eval_v9_qwen_full_{k5,k10,k20}.json`
> + A8 from `data/processed/got/musique/test_qi/eval_v9_qwen_a8_k10.json`.
> Embedder: DashScope `text-embedding-v3` (dim=1024). Same 12085-ep test
> split as Table 1a. PYTHONHASHSEED=0.
>
> Run walls: k=10 = 9573 s (cold, ~85 k unique texts → SQLite cache),
> k=5 = ~3700 s (cache-warm), k=20 = 3881 s (cache-warm), Stage B
> A8 query_intent = 4729 s (qwen-plus, 32 workers, 12085 ep, 64 k LLM
> calls, fb-rate 0.05 %).

### 1b.1 Recall@10 / MRR@10 (HEADLINE)

| **Variant** | **S1 R@10** | **S2 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR@10** | **S2 MRR@10** | **S3 MRR@10** | **S4 MRR@10** | **S4 cFMR** | **Avg Savings** |
| ----------- | ----------: | ----------: | ----------: | ----------: | ------------: | ------------: | ------------: | ------------: | ----------: | --------------: |
| **A1** | 0.9853 | 0.9667 | 0.9981 | 0.7722 | 0.8946 | 0.8298 | 0.8032 | 0.7095 | 0.000 | 0% |
| A2 | 0.6806 | 0.7078 | 0.7831 | 0.5997 | 0.7197 | 0.6875 | 0.6980 | 0.6367 | **0.6835** | 75% |
| **A3** | 0.7275 | 0.7437 | 0.7914 | 0.7724 | 0.7509 | 0.7003 | 0.6981 | 0.7281 | 0.000 | 75% |
| **A4** | **0.9995** | **0.9817** | 0.9980 | **0.9426** | 0.8753 | 0.8001 | 0.7873 | **0.7578** | 0.000 | 75% |
| A4_nofb | 0.9995 | 0.9817 | 0.9980 | 0.9425 | 0.8753 | 0.8000 | 0.7873 | 0.7578 | 0.000 | 75% |
| A4_norerank | 0.7275 | 0.7437 | 0.7914 | 0.7724 | 0.7509 | 0.7003 | 0.6981 | 0.7281 | 0.000 | 75% |
| **A5** | 0.9890 | 0.9708 | **0.9985** | 0.8584 | **0.8954** | **0.8315** | 0.8050 | 0.7379 | 0.000 | 75% |
| A5_noproj | 0.9865 | 0.9677 | 0.9981 | 0.8688 | 0.8955 | 0.8316 | **0.8056** | 0.7416 | 0.000 | 75% |
| A5_norerank | 0.4608 | 0.4808 | 0.4899 | 0.4452 | 0.5231 | 0.4631 | 0.3473 | 0.4431 | 0.000 | 75% |
| **A6** | 0.9911 | 0.9734 | 0.9981 | 0.7745 | 0.8954 | 0.8315 | 0.8050 | 0.7093 | 0.000 | 45% |
| **A7** | 0.9911 | 0.9734 | 0.9981 | 0.7745 | 0.8954 | 0.8315 | 0.8050 | 0.7093 | 0.000 | 45% |
| **A8** | 0.9928 | 0.9760 | 0.9986 | 0.7754 | **0.9223** | **0.8796** | **0.8749** | 0.7283 | 0.000 | 45% |

> **A8 (query_intent rerank) — Stage B done (12085 ep, qwen-plus, 32 workers, wall 79 min).**
> Source: `data/processed/got/musique/test_qi/eval_v9_qwen_a8_k10.json`. A8 is
> identical to A7 on stage-1 candidates (same shared retriever, same router,
> so R@10 lifts are tiny: +0.1–0.3 pt), but the LLM-generated `query_intent`
> rerank produces a clear MRR boost: **+2.7 / +4.8 / +7.0 / +1.9 pt** on
> S1/S2/S3/S4 — i.e. correct memories are pushed into earlier ranks within
> the top-10. A4 still wins R@k on S2/S4; A8 is the **MRR / top-1 latency**
> winner.

> Bold = best on each column. *Avg Savings* averages the per-subset
> first-stage savings reported by the runner.

### 1b.2 Recall@5 / Recall@20 (alternative cutoffs)

> k=5 from `eval_v9_qwen_full_k5.json` ✅; k=20 from
> `eval_v9_qwen_full_k20.json` ✅ (wall 3881 s, 12085 ep, cache-warm).

| **Variant**     | **S1 R@5** | **S1 R@20** | **S2 R@5** | **S2 R@20** | **S4 R@5** | **S4 R@20** |
| --------------- | ---------: | ----------: | ---------: | ----------: | ---------: | ----------: |
| **A1**          | 0.7644     | **1.0000**  | 0.7595     | **1.0000**  | 0.6720     | 0.7797      |
| A2              | 0.4854     | 0.8847      | 0.5335     | 0.8836      | 0.4776     | 0.7065      |
| **A3**          | 0.5202     | 0.9346      | 0.5488     | 0.9342      | 0.5909     | 0.9561      |
| **A4**          | **0.7851** | **1.0000**  | **0.7806** | **1.0000**  | **0.8025** | 0.9709      |
| A4_nofb         | 0.7851     | 1.0000      | 0.7806     | 1.0000      | 0.8025     | 0.9713      |
| A4_norerank     | 0.5202     | 0.9346      | 0.5488     | 0.9342      | 0.5909     | 0.9561      |
| **A5**          | 0.7678     | 1.0000      | 0.7622     | 1.0000      | 0.7058     | 0.9582      |
| A5_noproj       | 0.7659     | 1.0000      | 0.7604     | 1.0000      | 0.7064     | **0.9853**  |
| A5_norerank     | 0.3112     | 0.7526      | 0.3261     | 0.7406      | 0.2983     | 0.7728      |
| **A6**          | 0.7705     | 1.0000      | 0.7656     | 1.0000      | 0.6760     | 0.7797      |
| **A7**          | 0.7705     | 1.0000      | 0.7656     | 1.0000      | 0.6760     | 0.7797      |

> **R@5 observation:** A4 still wins on every cell. Notably A4 keeps a
> larger margin over A1 at small k: S4 R@5 = 0.802 vs A1 R@5 = 0.672
> (+13.0 pt, vs +17.0 pt at k=10) — private fallback brings high-value
> evidence into the top-5, not just within the top-10. Block routing
> (A6) costs S4 R@5 by 12.6 pt vs A4 (0.676 vs 0.802), the same
> direction as at k=10.
>
> **R@20 observation:** the candidate pool is large enough that S1/S2 hit
> 1.000 for almost every variant — k=20 saturates on these subsets and is
> not informative. **S4** remains the discriminator: A5_noproj (0.985) >
> A4_nofb (0.971) ≈ A4 (0.971) > A5 (0.958) > A3 (0.956) ≫ A6/A7/A1
> (0.780). The block-router variants (A6/A7) are capped at A1's recall
> because they never fall back to the global private pool — confirming
> the design choice to keep A4-style fallback as the headline. A5's
> projection slightly *hurts* S4 R@20 (–2.7 pt vs A5_noproj), the only
> place where the projection is net-negative; we keep A5 in the chain
> for the projection ablation story but A4 / A4_nofb is the recommended
> production setting.

---

## Table 2 · Main Results — 2WikiMultiHopQA (EM / F1)

| **Method** | **S1 EM** | **S1 F1** | **S2 EM** | **S2 F1** | **S3 EM** | **S3 F1** |
| ---------- | --------- | --------- | --------- | --------- | --------- | --------- |
| BM25 | | | | | | |
| DPR | | | | | | |
| ColBERT | | | | | | |
| Iter-RetGen | | | | | | |
| HippoRAG | | | | | | |
| MemWalker | | | | | | |
| MemMA | | | | | | |
| **A1** | | | | | | |
| **A2** | | | | | | |
| **A3** | | | | | | |
| **A4** | | | | | | |
| **A5** | | | | | | |
| **A6** | | | | | | |
| **A7** | | | | | | |
| **A8** | | | | | | |

---

## Table 3 · Main Results — HotpotQA (EM / F1)

| **Method** | **S1 EM** | **S1 F1** | **S2 EM** | **S2 F1** | **S3 EM** | **S3 F1** |
| ---------- | --------- | --------- | --------- | --------- | --------- | --------- |
| BM25 | | | | | | |
| DPR | | | | | | |
| ColBERT | | | | | | |
| Iter-RetGen | | | | | | |
| HippoRAG | | | | | | |
| MemWalker | | | | | | |
| MemMA | | | | | | |
| **A1** | | | | | | |
| **A2** | | | | | | |
| **A3** | | | | | | |
| **A4** | | | | | | |
| **A5** | | | | | | |
| **A6** | | | | | | |
| **A7** | | | | | | |
| **A8** | | | | | | |

---

## Table 4 · Main Results — GSM8K (Accuracy)

| **Method** | **S1 Acc** | **S2 Acc** | **S3 Acc** |
| ---------- | ---------- | ---------- | ---------- |
| BM25 | | | |
| DPR | | | |
| ColBERT | | | |
| Iter-RetGen | | | |
| **A1** | | | |
| **A2** | | | |
| **A3** | | | |
| **A4** | | | |
| **A5** | | | |
| **A6** | | | |
| **A7** | | | |
| **A8** | | | |

---

## Table 5 · Main Results — MATH (Accuracy)

| **Method** | **S1 Acc** | **S2 Acc** | **S3 Acc** |
| ---------- | ---------- | ---------- | ---------- |
| BM25 | | | |
| DPR | | | |
| ColBERT | | | |
| Iter-RetGen | | | |
| **A1** | | | |
| **A2** | | | |
| **A3** | | | |
| **A4** | | | |
| **A5** | | | |
| **A6** | | | |
| **A7** | | | |
| **A8** | | | |

---

## Table 6 · S4 Privacy — Cross-dataset (FMR / cFMR)

> Two privacy metrics are reported. **FMR** = structural false-merge
> (1.0 means a shared bucket was used at all on S4 verifier requests).
> **cFMR** is the *real* privacy metric: a restricted memory actually
> appeared in the agent's top-k. Production deployments care about cFMR.

| **Method** | **MuSiQue FMR / cFMR** | **2Wiki FMR / cFMR** | **HotpotQA FMR / cFMR** | **GSM8K FMR / cFMR** | **MATH FMR / cFMR** |
| ---------- | ---------------------- | -------------------- | ----------------------- | -------------------- | ------------------- |
| **A1** | 0.000 / **0.000** | | | | |
| **A2** | 1.000 / **0.341** | | | | |
| **A3** | 1.000 / **0.000** | | | | |
| **A4** | 1.000 / **0.000** | | | | |
| **A5** | 1.000 / **0.000** | | | | |
| **A6** | 0.000 / **0.000** | | | | |
| **A7** | 0.000 / **0.000** | | | | |
| **A8** | (pending) | | | | |

---

## Table 7 · Cross-dataset Consistency — A1 vs A8

> Δ = A8 − A1. ρ_mean is the average overlap rate of the dataset's S1+S2+S3 episodes. Note: absolute ρ values are not comparable across QA and Math datasets (different ρ computation basis).

| **Dataset** | **Type** | **ρ_mean** | **A1 Score** | **A8 Score** | **Δ** |
| ----------- | -------- | ---------- | ------------ | ------------ | ----- |
| MuSiQue | QA (EM/F1) | | | | |
| 2WikiMultiHopQA | QA (EM/F1) | | | | |
| HotpotQA | QA (EM/F1) | | | | |
| GSM8K | Math (Acc) | | | | |
| MATH | Math (Acc) | | | | |

---

## Table 8 · Component Ablation — Marginal Contribution

> Each row shows the delta from removing exactly one component. Computed on MuSiQue dev, GoT structure, S2 subset (partial overlap).

| **Ablation Axis** | **Variant Pair** | **ΔRecall@10** | **ΔEM** | **ΔF1** | **ΔLatency (ms)** | **Note** |
| ----------------- | ---------------- | -------------- | ------- | ------- | ----------------- | -------- |
| Scope bucketing | A1 → A3 | | | | | Mainly efficiency: −Recall, +Savings |
| Rerank + private fallback | A3 → A4 | | | | | The biggest single recall jump |
| SVD warm-start | A4 → A5 | | | | | Tiny / no effect at small `n_q` (see §6 soft-downgrade) |
| Hierarchical (policy-aware) routing | A5 → A6 | | | | | Drops S4 recall vs A5 by ~7pt; learned-W not yet implemented |
| Step-2 LLM query rewrite | A6 → A8 | | | | | Pending (Stage B) |
| (A7) | A6 → A7 | 0 | 0 | 0 | 0 | A7 is a code-level alias of A6 today |

---

## Table 9 · Efficiency Metrics — A8 vs A1

> Shared-first-stage Savings = relative reduction in first-stage retrieval count. Conversion Rate = fraction of C_shared candidates actually used by at least one agent.

| **Subset** | **Dataset** | **Shared-1st-stage Savings** | **Shared-to-Useful Conv. Rate** | **Fallback Necessity Rate** | **Avg Latency A1 (ms)** | **Avg Latency A8 (ms)** | **Latency Δ%** |
| ---------- | ----------- | ---------------------------- | ------------------------------- | --------------------------- | ----------------------- | ----------------------- | -------------- |
| S1 | MuSiQue | | | | | | |
| S2 | MuSiQue | | | | | | |
| S3 | MuSiQue | | | | | | |
| S1 | HotpotQA | | | | | | |
| S2 | HotpotQA | | | | | | |
| S3 | HotpotQA | | | | | | |
| S1 | GSM8K | | | | | | |
| S2 | GSM8K | | | | | | |
| S3 | GSM8K | | | | | | |

---

## Table 10 · Retrieval Quality Metrics — A8 (MuSiQue dev)

> Per-role breakdown. Evidence Hit Rate = fraction of C_i_final containing at least one ground-truth evidence item.

| **Agent Role** | **Recall@5** | **Recall@10** | **MRR@10** | **Evidence Hit Rate** | **Answer Support Rate** |
| -------------- | ------------ | ------------- | ---------- | --------------------- | ----------------------- |
| Planner | | | | | |
| Solver (hop 1) | | | | | |
| Solver (hop 2) | | | | | |
| Solver (hop 3) | | | | | |
| Verifier | | | | | |
| **Overall** | | | | | |

---

## Table 11 · Agent Count n Ablation (A8, MuSiQue S2)

> Fixed episode set. n varied by sub-sampling agents per bucket. SVD rank r re-optimized per n.

| **n (agents/bucket)** | **Best r** | **Recall@10** | **EM** | **F1** | **Avg Latency (ms)** | **Fallback Rate** |
| --------------------- | ---------- | ------------- | ------ | ------ | -------------------- | ----------------- |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |

---

## Table 12 · SVD Rank r Sensitivity — A5 (MuSiQue dev)

> Truncated SVD, no learned W. Evaluated across two n regimes.

| **r** | **n < 3 Recall@10** | **n < 3 EM** | **n ≥ 3 Recall@10** | **n ≥ 3 EM** | **Note** |
| ----- | ------------------- | ------------ | ------------------- | ------------ | -------- |
| 8 | | | | | |
| 16 | | | | | |
| 32 | | | | | |
| 64 | | | | | |

---

## Table 13 · Learned W Hyperparameter Grid — A6 (MuSiQue dev, Recall@10)

> Grid search: r ∈ {16, 32, 64, 128} × τ ∈ {0.05, 0.1, 0.5, 1.0}. Cell = Recall@10.

| **r \ τ** | **τ = 0.05** | **τ = 0.10** | **τ = 0.50** | **τ = 1.00** |
| --------- | ------------ | ------------ | ------------ | ------------ |
| r = 16 | | | | |
| r = 32 | | | | |
| r = 64 | | | | |
| r = 128 | | | | |

---

## Table 14 · Learned W Cross-dataset Generalization (A5 vs A6)

> W trained on MuSiQue train only. Zero-shot transfer to other datasets. Δ = A6 − A5.

| **Dataset** | **A5 Recall@10** | **A6 Recall@10** | **Δ Recall@10** | **A5 Score** | **A6 Score** | **Δ Score** | **Verdict** |
| ----------- | ---------------- | ---------------- | --------------- | ------------ | ------------ | ----------- | ----------- |
| MuSiQue | | | | | | | in-domain |
| 2WikiMultiHopQA | | | | | | | zero-shot |
| HotpotQA | | | | | | | zero-shot |
| GSM8K | | | | | | | zero-shot |
| MATH | | | | | | | zero-shot |

---

## Table 15 · Reasoning Structure Comparison (MuSiQue, A4 retrieval fixed)

> Reasoning structure is **orthogonal** to the retrieval variant. Naming
> rule: variant ID `A4-GoT`, `A4-CoT`, `A4-ToT` for the same retrieval
> pipeline applied on top of each structure. Only A4 (the retrieval
> winner) is reported here; per-cell entries can be added for any other
> retrieval variant if needed.
>
> Each structure independently computes ρ and subset labels — the same raw
> item lands in different (S1/S2/S3) buckets depending on the chosen
> structure. ρ values are NOT comparable across rows.
>
> **CoT status note (2026-04-30):** current CoT numbers are a retrieval-only
> **partial preview** from the legal CoT graph family only:
> `LINEAR` (`S1`, n=516) and `POLICY_ISOLATED` (`S4`, n=310). End-to-end
> EM/F1 is still pending, so those cells remain `*`. CoT `S2/S3` are not yet
> populated in the current partial slice.

| **Structure** | **Subset** | **ρ_mean** | **EM** | **F1** | **Recall@10** | **MRR@10** | **Fallback rate** | **Latency (ms)** |
| ------------- | ---------- | ---------- | ------ | ------ | ------------- | ---------- | ----------------- | ---------------- |
| **GoT (A4)** | S1 | 0.50 | * | * | 0.972 | 0.928 | | |
| **GoT (A4)** | S2 | 0.20 | * | * | 0.956 | 0.895 | | |
| **GoT (A4)** | S3 | 0.00 | * | * | 0.964 | 0.875 | | |
| **GoT (A4)** | S4 | — | * | * | 0.918 | 0.814 | | |
| CoT (A4-CoT) | S1 | | * | * | 0.278 | 0.587 | | |
| CoT (A4-CoT) | S2 | | * | * | (pending) | (pending) | | |
| CoT (A4-CoT) | S3 | | * | * | (pending) | (pending) | | |
| CoT (A4-CoT) | S4 | — | * | * | 0.208 | 0.440 | | |
| ToT (A4-ToT) | S1 | | * | * | | | | |
| ToT (A4-ToT) | S2 | | * | * | | | | |
| ToT (A4-ToT) | S3 | | * | * | | | | |

---

## Table 15a · CoT Partial Preview (MuSiQue dev, retrieval-only)

> **Status:** interim CoT-only preview for meeting use. These numbers are
> retrieval-side metrics only and must **not** be treated as the final paper
> figures.
>
> Sources:
> - `cot_partial_516_linear/summary.md` → `LINEAR / S1`, n=516
> - `cot_s4_partial_310/summary.md` → `POLICY_ISOLATED / S4`, n=310
>
> Metrics:
> - `EvidenceRecall@10` = gold-evidence coverage
> - `Hit@10` = at least one gold evidence item appears in top-10
> - `MRR@10` = reciprocal rank of the first gold evidence item
> - `FMR` = routing-level false merge on S4

### 15a.1 CoT-LINEAR / S1 (n=516)

| **Variant** | **EvidenceRecall@10** | **Hit@10** | **MRR@10** | **Savings** | **FMR** |
| ----------- | --------------------: | ---------: | ---------: | ----------: | ------: |
| **A1** | 1.0000 | 1.0000 | 0.8080 | 0.0000 | 0.0000 |
| **A3** | 0.2778 | 0.6667 | 0.3688 | 0.6667 | 0.0000 |
| **A4** | 0.2778 | 0.6667 | 0.5867 | 0.6667 | 0.0000 |
| A4_nofb | 0.2778 | 0.6667 | 0.5867 | 0.6667 | 0.0000 |
| A4_norerank | 0.2778 | 0.6667 | 0.3688 | 0.6667 | 0.0000 |
| **A5** | 0.6111 | 1.0000 | 0.6505 | 0.3333 | 0.0000 |
| A5_noproj | 0.6111 | 1.0000 | 0.7174 | 0.3333 | 0.0000 |
| A5_norerank | 0.4267 | 0.9076 | 0.4388 | 0.3333 | 0.0000 |
| **A6** | 0.7222 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| **A7** | 0.7222 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| **A8** | 0.7222 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### 15a.2 CoT-POLICY_ISOLATED / S4 (n=310)

| **Variant** | **EvidenceRecall@10** | **Hit@10** | **MRR@10** | **Savings** | **FMR** |
| ----------- | --------------------: | ---------: | ---------: | ----------: | ------: |
| **A1** | 0.7500 | 0.7500 | 0.5957 | 0.0000 | 0.0000 |
| **A3** | 0.2083 | 0.5000 | 0.2661 | 0.7500 | 1.0000 |
| **A4** | 0.2083 | 0.5000 | 0.4399 | 0.7500 | 1.0000 |
| A4_nofb | 0.2083 | 0.5000 | 0.4399 | 0.7500 | 1.0000 |
| A4_norerank | 0.2083 | 0.5000 | 0.2661 | 0.7500 | 1.0000 |
| **A5** | 0.4583 | 0.7500 | 0.4754 | 0.2500 | 0.0000 |
| A5_noproj | 0.4583 | 0.7500 | 0.5223 | 0.2500 | 0.0000 |
| A5_norerank | 0.3216 | 0.6790 | 0.3212 | 0.2500 | 0.0000 |
| **A6** | 0.5417 | 0.7500 | 0.7500 | 0.0000 | 0.0000 |
| **A7** | 0.5417 | 0.7500 | 0.7500 | 0.0000 | 0.0000 |
| **A8** | 0.5417 | 0.7500 | 0.7500 | 0.0000 | 0.0000 |

### 15a.3 Interim Takeaways

- On the current CoT `LINEAR / S1` slice, shared retrieval already yields
  clear first-stage savings, but `A4` does not preserve evidence-level
  coverage; `A5` is a more stable compromise.
- On the current CoT `POLICY_ISOLATED / S4` slice, aggressive sharing
  (`A3/A4`) exposes routing-level false merges (`FMR=1.0`) and degrades
  retrieval quality sharply.
- `A5_noproj` outperforms or matches `A5` on both slices, so SVD projection
  still shows no stable benefit in the current CoT partial data.
- `A6/A7/A8` are currently identical on both slices, so query rewriting has
  not yet separated these variants under the present post-hoc CoT evaluation
  setup.

---

## Table 15b · CoT Partial Preview (HotpotQA dev, retrieval-only)

> **Status:** interim CoT-only preview for meeting use. These numbers are
> retrieval-side metrics only and must **not** be treated as the final paper
> figures.
>
> Source:
> - `cot_hotpot_partial_115_linear/summary.md` → `LINEAR / S1`, n=118
>
> Current scope:
> - only `LINEAR / S1` is available
> - `POLICY_ISOLATED / S4` is still pending for HotpotQA

### 15b.1 CoT-LINEAR / S1 (n=118)

| **Variant** | **EvidenceRecall@10** | **Hit@10** | **MRR@10** | **Savings** | **FMR** |
| ----------- | --------------------: | ---------: | ---------: | ----------: | ------: |
| **A1** | 1.0000 | 1.0000 | 0.8691 | 0.0000 | 0.0000 |
| **A3** | 0.2260 | 0.5424 | 0.3164 | 0.6667 | 0.0000 |
| **A4** | 0.2260 | 0.5424 | 0.4788 | 0.6667 | 0.0000 |
| A4_nofb | 0.2260 | 0.5424 | 0.4788 | 0.6667 | 0.0000 |
| A4_norerank | 0.2260 | 0.5424 | 0.3164 | 0.6667 | 0.0000 |
| **A5** | 0.5593 | 0.8757 | 0.6073 | 0.3333 | 0.0000 |
| A5_noproj | 0.5593 | 0.8757 | 0.6840 | 0.3333 | 0.0000 |
| A5_norerank | 0.5056 | 0.8672 | 0.4256 | 0.3333 | 0.0000 |
| **A6** | 0.7740 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| **A7** | 0.7740 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| **A8** | 0.7740 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### 15b.2 Interim Takeaways

- The HotpotQA CoT partial slice shows the same qualitative pattern as the
  MuSiQue CoT partial slice: aggressive shared retrieval (`A4`) buys large
  savings but sharply hurts evidence coverage.
- `A5` remains a middle-ground option: lower savings than `A4`, but clearly
  better `EvidenceRecall@10`, `Hit@10`, and `MRR@10`.
- `A5_noproj` again matches `A5` on coverage and improves ranking quality,
  so SVD projection still does not show a stable positive contribution.
- `A6/A7/A8` are again identical on the current post-hoc CoT slice, so query
  rewriting has not yet separated the private / hierarchical variants.

---

## Table 16 · Subset Distribution by Reasoning Structure (MuSiQue dev)

> Same episode set, different ρ computations due to varying agent-private scope size. GoT has largest private scope → lowest ρ → smallest S1 share.

| **Structure** | **S1 count** | **S1 ρ_mean** | **S2 count** | **S2 ρ_mean** | **S3 count** | **S3 ρ_mean** |
| ------------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- |
| GoT | | | | | | |
| ToT | | | | | | |
| CoT | | | | | | |

---

## Table 17 · Subset Distribution Summary (all 5 datasets, GoT, seed=42)

> ρ thresholds (from `config/yaml/subset_thresholds.yaml`):
> S1 if ρ > 0.3, S3 if ρ ≤ 0.1, else S2. S4 is orthogonal (`s4_eligible=True`).

| **Dataset**     | **Total episodes** | **S1 count** | **S1 ρ_mean** | **S2 count** | **S2 ρ_mean** | **S3 count** | **S3 ρ_mean** | **S4 count** |
| --------------- | -----------------: | -----------: | ------------: | -----------: | ------------: | -----------: | ------------: | -----------: |
| **MuSiQue**     | **12085**          | **4744**     | **0.466**     | **3672**     | **0.198**     | **1252**     | **0.000**     | **2417**     |
| 2WikiMultiHopQA |                    |              |               |              |               |              |               |              |
| HotpotQA        |                    |              |               |              |               |              |               |              |
| GSM8K           |                    |              |               |              |               |              |               |              |
| MATH            |                    |              |               |              |               |              |               |              |
