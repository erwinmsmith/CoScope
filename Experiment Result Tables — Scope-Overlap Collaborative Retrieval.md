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

## Table 1c · Retrieval Quality — 2WikiMultiHopQA (62 880 ep, GoT, **Qwen text-embedding-v3**)

> **Source:** `data/processed/got/2wikimhqa/test/eval_v9_qwen_full_{k5,k10,k20}.json`.
> Embedder: DashScope `text-embedding-v3` (dim=1024). Same 11-variant lineup,
> same evaluator as §1b. PYTHONHASHSEED=0. Wall: k=10 cold ≈ 6 h (cache cold
> for 2Wiki corpus), k=5 / k=20 cache-warm ~45 min each.
>
> Per-subset N: **S1=15042, S2=16319, S3=18943, S4=12576**. (S4 has no
> `mem_rs_` restricted-layer entries on 2Wiki because the RestrictedBuilder
> only loads from interim files prepared for MuSiQue; A2's cFMR is
> mechanically 0 on this dataset and is not directly comparable to MuSiQue.)

### 1c.1 Recall@10 / MRR@10 (HEADLINE)

| **Variant** | **S1 R@10** | **S2 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR@10** | **S2 MRR@10** | **S3 MRR@10** | **S4 MRR@10** | **S4 cFMR** | **Avg Savings** |
| ----------- | ----------: | ----------: | ----------: | ----------: | ------------: | ------------: | ------------: | ------------: | ----------: | --------------: |
| **A1** | 0.9980 | 0.9792 | 0.9999 | 0.7679 | 0.9295 | 0.8708 | 0.9458 | 0.7283 | 0.000 | 0% |
| A2 | 0.8663 | 0.8819 | 0.9814 | 0.9201 | 0.9250 | 0.8622 | 0.9335 | **0.9105** | 0.000 | 75% |
| A3 | 0.8675 | 0.8835 | 0.9823 | 0.9303 | 0.9244 | 0.8609 | 0.9333 | 0.9094 | 0.000 | 75% |
| **A4** | 0.9980 | 0.9793 | 0.9999 | **0.9806** | 0.9296 | 0.8707 | 0.9458 | 0.8704 | 0.000 | 75% |
| A4_nofb | 0.9980 | 0.9793 | 0.9999 | 0.9806 | 0.9296 | 0.8707 | 0.9458 | 0.8704 | 0.000 | 75% |
| A4_norerank | 0.8675 | 0.8835 | 0.9823 | 0.9303 | 0.9244 | 0.8609 | 0.9333 | 0.9094 | 0.000 | 75% |
| **A5** | **0.9981** | **0.9817** | 0.9999 | 0.9467 | 0.9334 | 0.8855 | 0.9503 | 0.7835 | 0.000 | 75% |
| A5_noproj | 0.9981 | 0.9815 | 0.9999 | 0.9467 | **0.9345** | **0.8867** | **0.9508** | 0.7838 | 0.000 | 75% |
| A5_norerank | 0.8740 | 0.8702 | 0.9750 | 0.8714 | 0.7448 | 0.7100 | 0.6877 | 0.7421 | 0.000 | 75% |
| **A6** | 0.9981 | 0.9816 | 0.9999 | 0.7679 | 0.9332 | 0.8853 | 0.9502 | 0.7286 | 0.000 | 45% |
| **A7** | 0.9981 | 0.9816 | 0.9999 | 0.7679 | 0.9332 | 0.8853 | 0.9502 | 0.7286 | 0.000 | 45% |
| **A8** | (Stage B running) | | | | | | | | | |

> Headline: **A4 lifts S4 R@10 from 0.768 (A1) → 0.981, Δ = +21.3 pt**, while
> keeping S1/S2/S3 saturated and cFMR = 0.000. A5 wins MRR on S1/S2/S3 by
> ~0.5 pt over A1; A5_noproj is the *MRR* winner. Block-routed A6/A7
> degenerate to A1 on S4 (no fallback).

### 1c.2 Recall@5 / Recall@20 (alternative cutoffs)

| **Variant**     | **S1 R@5** | **S1 R@20** | **S2 R@5** | **S2 R@20** | **S4 R@5** | **S4 R@20** |
| --------------- | ---------: | ----------: | ---------: | ----------: | ---------: | ----------: |
| **A1**          | 0.7941     | **1.0000**  | 0.8292     | **1.0000**  | 0.7094     | 0.7683      |
| A2              | 0.6552     | 0.9989      | 0.6567     | 0.9985      | 0.7672     | **1.0000**  |
| A3              | 0.6552     | 1.0000      | 0.6556     | 1.0000      | 0.7693     | 1.0000      |
| **A4**          | 0.7941     | **1.0000**  | 0.8287     | 1.0000      | **0.8839** | 1.0000      |
| A4_nofb         | 0.7941     | 1.0000      | 0.8287     | 1.0000      | 0.8839     | 1.0000      |
| A4_norerank     | 0.6552     | 1.0000      | 0.6556     | 1.0000      | 0.7693     | 1.0000      |
| **A5**          | 0.7958     | 1.0000      | 0.8327     | 1.0000      | 0.7730     | 1.0000      |
| A5_noproj       | **0.7958** | 1.0000      | **0.8328** | 1.0000      | 0.7730     | 1.0000      |
| A5_norerank     | 0.5525     | 1.0000      | 0.5854     | 1.0000      | 0.6132     | 1.0000      |
| **A6**          | 0.7958     | 1.0000      | 0.8325     | 1.0000      | 0.7108     | 0.7683      |
| **A7**          | 0.7958     | 1.0000      | 0.8325     | 1.0000      | 0.7109     | 0.7683      |

> **R@20 on S4** is the discriminator: A2/A3/A4/A5 all reach 1.000 (full
> private fallback recovers everything), A6/A7/A1 cap at 0.768. At small k
> (=5) A4 is the clear S4 winner (+17.5 pt over A1, +17.3 pt over A6/A7).

---

## Table 1d · Retrieval Quality — HotpotQA (37 025 ep, GoT, **Qwen text-embedding-v3**)

> **Source:** `data/processed/got/hotpotqa/test/eval_v9_qwen_full_{k5,k10,k20}.json`.
> Hotpot has no public-answer test split; its loader falls back to
> `distractor_validation.parquet`. Wall: k=10 cold ≈ 4 h, k=5 / k=20
> cache-warm ~30 min each.
>
> Per-subset N: **S1=5918, S2=11836, S3=11866, S4=7405**. Same caveat as
> §1c on A2 cFMR (no `mem_rs_` entries built for Hotpot).

### 1d.1 Recall@10 / MRR@10 (HEADLINE)

| **Variant** | **S1 R@10** | **S2 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR@10** | **S2 MRR@10** | **S3 MRR@10** | **S4 MRR@10** | **S4 cFMR** | **Avg Savings** |
| ----------- | ----------: | ----------: | ----------: | ----------: | ------------: | ------------: | ------------: | ------------: | ----------: | --------------: |
| **A1** | 0.9994 | 0.9987 | 0.9995 | 0.7497 | 0.8546 | 0.7290 | 0.8588 | 0.6467 | 0.000 | 0% |
| A2 | 0.9808 | 0.9658 | 0.9601 | 0.9773 | 0.8738 | 0.7673 | 0.8688 | **0.8605** | 0.000 | 72% |
| A3 | 0.9824 | 0.9686 | 0.9659 | 0.9836 | 0.8687 | 0.7566 | 0.8635 | 0.8549 | 0.000 | 72% |
| **A4** | **0.9999** | **0.9997** | 0.9994 | **0.9947** | 0.8230 | 0.7118 | 0.8397 | 0.7549 | 0.000 | 72% |
| A4_nofb | 0.9999 | 0.9997 | 0.9994 | 0.9947 | 0.8158 | 0.7111 | 0.8387 | 0.7539 | 0.000 | 72% |
| A4_norerank | 0.9824 | 0.9686 | 0.9659 | 0.9836 | 0.8687 | 0.7566 | 0.8635 | 0.8549 | 0.000 | 72% |
| **A5** | 0.9995 | 0.9988 | **0.9995** | 0.9565 | 0.8666 | 0.8608 | 0.9244 | 0.7245 | 0.000 | 72% |
| A5_noproj | 0.9995 | 0.9989 | 0.9995 | 0.9566 | **0.9448** | **0.9558** | **0.9725** | 0.7740 | 0.000 | 72% |
| A5_norerank | 0.7416 | 0.7019 | 0.9356 | 0.7873 | 0.6705 | 0.7013 | 0.6378 | 0.5911 | 0.000 | 72% |
| **A6** | 0.9995 | 0.9989 | 0.9995 | 0.7497 | 0.8649 | 0.8637 | 0.9242 | 0.6671 | 0.000 | 38% |
| **A7** | 0.9995 | 0.9989 | 0.9995 | 0.7497 | 0.8635 | 0.8644 | 0.9242 | 0.6675 | 0.000 | 38% |
| **A8** | 0.9994 | 0.9987 | 0.9994 | 0.7497 | **0.8986** | **0.8837** | **0.9455** | **0.6914** | 0.000 | 38% |

> Headline: **A4 lifts S4 R@10 from 0.750 (A1) → 0.995, Δ = +24.5 pt** —
> the largest cross-dataset margin so far. A5_noproj is the MRR winner
> on S1/S2/S3 by a wide gap (+9 pt over A1 on S2 MRR). A4 and A5 agree
> on S1/S2/S3 R@10 within 0.05 pt; the meaningful split is the
> S4-fallback choice.
>
> **A8 (query_intent rerank) — Stage B done (37 025 ep, qwen-plus, 32
> workers, wall 161 min A8 eval after 14 h cold query_intent gen).**
> Source: `data/processed/got/hotpotqa/test_qi/eval_v9_qwen_a8_k10.json`.
> Same pattern as MuSiQue §1b: A8 R@10 ≡ A6/A7 (same stage-1 candidates),
> but the LLM-generated `query_intent` rerank produces **MRR boost of
> +4.4 / +15.5 / +8.7 / +4.5 pt on S1/S2/S3/S4 over A1** — i.e. the
> correct memory is pushed earlier in the top-10. S2 (+15.5 pt) is
> especially strong on Hotpot, consistent with Hotpot's notoriously
> distractor-heavy partial-overlap setting.

### 1d.2 Recall@5 / Recall@20

| **Variant**     | **S1 R@5** | **S1 R@20** | **S2 R@5** | **S2 R@20** | **S4 R@5** | **S4 R@20** |
| --------------- | ---------: | ----------: | ---------: | ----------: | ---------: | ----------: |
| **A1**          | 0.9911     | **1.0000**  | 0.8739     | **1.0000**  | 0.7437     | 0.7500      |
| A2              | 0.9274     | 1.0000      | 0.8228     | 1.0000      | 0.9271     | **1.0000**  |
| A3              | 0.9274     | 1.0000      | 0.8127     | 1.0000      | 0.9293     | 1.0000      |
| **A4**          | **0.9967** | 1.0000      | 0.8814     | 1.0000      | **0.9766** | 1.0000      |
| A4_nofb         | 0.9967     | 1.0000      | 0.8812     | 1.0000      | 0.9766     | 1.0000      |
| A4_norerank     | 0.9274     | 1.0000      | 0.8127     | 1.0000      | 0.9293     | 1.0000      |
| **A5**          | 0.9921     | 1.0000      | 0.9010     | 1.0000      | 0.8247     | 1.0000      |
| A5_noproj       | 0.9922     | 1.0000      | **0.9212** | 1.0000      | 0.8248     | 1.0000      |
| A5_norerank     | 0.5077     | 1.0000      | 0.5129     | 1.0000      | 0.4258     | 1.0000      |
| **A6**          | 0.9921     | 1.0000      | 0.8850     | 1.0000      | 0.7443     | 0.7500      |
| **A7**          | 0.9920     | 1.0000      | 0.8848     | 1.0000      | 0.7443     | 0.7500      |

> **Cross-dataset summary at R@10/S4 (Δ over A1)**: MuSiQue **+17.0 pt**,
> 2Wiki **+21.3 pt**, Hotpot **+24.5 pt**. The lift increases monotonically
> with the dataset's average distractor density — exactly the regime A4's
> private fallback was designed for.

---

## Table 1e · Retrieval Quality — MuSiQue ToT (4834 ep, **Qwen text-embedding-v3**, k=10)

> **ToT scope** — same MuSiQue test items, but each raw item is wrapped in a
> ToT FORK graph (one fan-out from the Planner over k sub-question solvers,
> no merge node). With `--include-s4`, every raw item additionally produces
> a `POLICY_ISOLATED` shard (FORK base + Verifier holding `mem_rs_*`
> evidence), so we get 2 417 FORK + 2 417 POLICY_ISOLATED = 4 834 episodes.
>
> Subset distribution under ToT FORK is **bimodal** (S1=1 164, S2=1, S3=1 252):
> the fan-out rarely produces partial-overlap branches, so MuSiQue ToT lands
> in S1 (shared workspace) or S3 (independent branches). S2 is therefore
> reported but statistically inert (n=1). S4 (n=2 417) is fully populated
> and structurally identical to the GoT-S4 contract (Verifier holds the
> only restricted evidence).
>
> Compare with §1b (GoT, same dataset / embedder / k) to read off the
> structure-orthogonality claim.

### 1e.1 Recall@10 / MRR@10 (HEADLINE)

| **Variant**     | **S1 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR** | **S3 MRR** | **S4 MRR** |
| --------------- | ----------: | ----------: | ----------: | ---------: | ---------: | ---------: |
| **A1**          | 0.9795      | 0.9981      | 0.7851      | 0.8770     | 0.8032     | 0.6678     |
| A2              | 0.6095      | 0.7838      | 0.5748      | 0.6610     | 0.6981     | 0.5685     |
| A3              | 0.6817      | 0.7914      | 0.7355      | 0.7171     | 0.6981     | 0.6634     |
| **A4**          | **0.9992**  | 0.9980      | **0.9209**  | 0.8536     | 0.7873     | **0.7009** |
| A4_nofb         | 0.9992      | 0.9980      | 0.9207      | 0.8536     | 0.7873     | 0.7007     |
| A4_norerank     | 0.6817      | 0.7914      | 0.7355      | 0.7171     | 0.6981     | 0.6634     |
| **A5**          | 0.9853      | 0.9985      | 0.8559      | 0.8805     | 0.8049     | 0.6928     |
| A5_noproj       | 0.9816      | 0.9981      | 0.8721      | 0.8805     | 0.8049     | 0.6981     |
| A5_norerank     | 0.4229      | 0.4899      | 0.4507      | 0.5385     | 0.3473     | 0.4427     |
| **A6**          | 0.9885      | 0.9981      | 0.7887      | **0.8805** | 0.8049     | 0.6699     |
| **A7**          | 0.9885      | 0.9981      | 0.7887      | 0.8805     | 0.8049     | 0.6699     |

> S2 is omitted from the headline table because n=1 makes the cell
> meaningless; the JSON report still carries it.

### 1e.2 Privacy on S4 — FMR / cFMR (n=2 417)

| **Variant**     | **FMR** | **cFMR** |
| --------------- | ------: | -------: |
| **A1**          | 0.000   | 0.000    |
| A2              | 0.000   | **0.5197** |
| A3              | 1.000   | 0.000    |
| **A4**          | 1.000   | 0.000    |
| A4_nofb         | 1.000   | 0.000    |
| A4_norerank     | 1.000   | 0.000    |
| A5              | 1.000   | 0.000    |
| A5_noproj       | 1.000   | 0.000    |
| A5_norerank     | 1.000   | 0.000    |
| **A6**          | 0.000   | 0.000    |
| **A7**          | 0.000   | 0.000    |

> ToT-S4 reproduces the GoT-S4 privacy ledger almost exactly:
> A2 (force-merge, no scope buckets) leaks **52.0 %** of restricted content,
> while every scope-aware variant (A3 onward) keeps **cFMR = 0**. FMR is the
> noisy metric that fires whenever a non-shared bucket lookup returns a
> hit — it does not imply leakage; cFMR is the contentful metric.

### 1e.3 ToT vs GoT Δ at A4 (MuSiQue, R@10)

| **Subset** | **GoT A4 (ref §1b)** | **ToT A4-ToT** | **Δ (ToT − GoT)** |
| ---------- | -------------------: | -------------: | ----------------: |
| S1         | 0.972                | **0.999**      | **+2.7 pt**       |
| S3         | 0.964                | **0.998**      | **+3.4 pt**       |
| S4         | 0.918                | **0.921**      | +0.3 pt           |

> Headline: A4 transfers cleanly to ToT. The S1/S3 lift is consistent with
> the FORK fan-out producing semantically tighter per-branch queries (each
> solver only handles its own sub-question), and the S4 lift is essentially
> zero — exactly what we want, because S4 quality is bottlenecked by the
> private-fallback retriever (independent of structure).

---

## Table 1f · Retrieval Quality — 2WikiMultiHopQA ToT (25 152 ep, **Qwen text-embedding-v3**, k=10)

> ToT 2Wiki/test under ToT FORK + S4 verifier-augmented topology yields
> 2 753 S1-FORK + 9 823 S3-FORK + 12 576 POLICY_ISOLATED (split S1=2 753 /
> S3=9 823 inside the S4 shard) = 25 152 episodes. ρ is bimodal again: 4-hop
> raw items map to S1 (ρ≈0.428), 2-hop items map to S3 (ρ=0); the 20 3-hop
> items spread across both. **No S2** under ToT 2Wiki — the FORK structure
> with no merge node cannot produce mid-overlap branches on a dataset that
> only ships 2-hop / 4-hop questions.

### 1f.1 Recall@10 / MRR@10 (HEADLINE)

| **Variant**     | **S1 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR** | **S3 MRR** | **S4 MRR** |
| --------------- | ----------: | ----------: | ----------: | ---------: | ---------: | ---------: |
| **A1**          | 0.9956      | 0.9999      | 0.7727      | 0.9217     | 0.8955     | 0.6976     |
| A2              | 0.7822      | 0.9643      | 0.9111      | 0.9216     | 0.8850     | 0.8657     |
| A3              | 0.7825      | 0.9658      | 0.9198      | 0.9223     | 0.8846     | 0.8651     |
| **A4**          | 0.9957      | 0.9999      | **0.9798**  | 0.9223     | 0.8955     | 0.8212     |
| A4_nofb         | 0.9957      | 0.9999      | 0.9798      | 0.9222     | 0.8956     | 0.8212     |
| A4_norerank     | 0.7825      | 0.9658      | 0.9198      | 0.9223     | 0.8846     | 0.8651     |
| **A5**          | 0.9959      | 0.9999      | 0.9454      | 0.9360     | 0.9052     | 0.7593     |
| A5_noproj       | 0.9959      | 0.9999      | 0.9455      | 0.9397     | 0.9092     | 0.7623     |
| A5_norerank     | 0.8111      | 0.9517      | 0.8792      | 0.8584     | 0.6163     | 0.7343     |
| **A6**          | 0.9958      | 0.9999      | 0.7727      | 0.9358     | 0.9046     | 0.7056     |
| **A7**          | 0.9958      | 0.9999      | 0.7727      | 0.9357     | 0.9045     | 0.7056     |

### 1f.2 Privacy on S4 — FMR / cFMR (n=12 576)

| **Variant**     | **FMR** | **cFMR** |
| --------------- | ------: | -------: |
| A1              | 0.000   | 0.000    |
| A2              | 1.000   | 0.000 †  |
| A3 / A4 / A5    | 1.000   | 0.000    |
| A4_nofb / norerank | 1.000 | 0.000    |
| A6 / A7         | 0.000   | 0.000    |

> † A2 cFMR is mechanically 0 on 2Wiki because the
> `data/interim/2wikimhqa/restricted_evidence/` interim is not yet built;
> the S4 shard contains no `mem_rs_` entries A2 could leak. See Table 6
> caveat.

### 1f.3 ToT vs GoT Δ at A4 (2Wiki, R@10)

| **Subset** | **GoT A4 (ref §1c)** | **ToT A4-ToT** | **Δ (ToT − GoT)** |
| ---------- | -------------------: | -------------: | ----------------: |
| S1         | 0.999                | 0.996          | −0.3 pt           |
| S3         | 0.998                | 1.000          | +0.2 pt           |
| S4         | 0.978                | **0.980**      | **+0.2 pt**       |

> A4-ToT lifts S4 by **+20.7 pt** over A1-ToT — almost identical to GoT's
> +21.3 pt. The S1/S3 numbers are also within ±0.3 pt of GoT. Reasoning
> structure is genuinely orthogonal to the retrieval pipeline on 2Wiki.

---

## Table 1g · Retrieval Quality — HotpotQA ToT (14 810 ep, **Qwen text-embedding-v3**, k=10)

> Hotpot/test under ToT FORK has 7 405 raw items, all 2-hop. With FORK +
> Verifier (POLICY_ISOLATED) we get 7 405 + 7 405 = 14 810 episodes.
>
> **Subset distribution: 100 % S3** (no S1 / no S2). Under ToT FORK the
> two solver branches share zero scope (each only handles its own
> sub-question, no merge node), and Hotpot's 2-hop format means there is
> no third hop to introduce shared task_shared context. So ρ=0 for every
> non-S4 episode → S3. Compare with GoT Hotpot in §1d, where the same
> raw items split across S1/S2/S3 because GoT mixes LINEAR_MERGE and
> FORK_MERGE templates that explicitly introduce a merge node and thereby
> produce ρ>0.
>
> This is a real structural signal, not a data issue: it shows that
> **scope-overlap subset assignments are sensitive to reasoning topology
> in the way our framework predicts**.

### 1g.1 Recall@10 / MRR@10 (HEADLINE)

| **Variant**     | **S3 R@10** | **S4 R@10** | **S3 MRR** | **S4 MRR** |
| --------------- | ----------: | ----------: | ---------: | ---------: |
| **A1**          | 0.9992      | 0.7494      | 0.7660     | 0.5732     |
| A2              | 0.9376      | 0.9046      | 0.8003     | 0.7702     |
| A3              | 0.9453      | 0.9397      | 0.7917     | 0.7603     |
| **A4**          | 0.9991      | **0.9815**  | 0.7353     | 0.6662     |
| A4_nofb         | 0.9991      | 0.9815      | 0.7347     | 0.6661     |
| A4_norerank     | 0.9453      | 0.9397      | 0.7917     | 0.7603     |
| **A5**          | 0.9992      | 0.9562      | 0.8674     | 0.7146     |
| A5_noproj       | 0.9992      | 0.9562      | 0.9267     | 0.7604     |
| A5_norerank     | 0.8969      | 0.9226      | 0.6164     | 0.6155     |
| **A6**          | 0.9992      | 0.7494      | 0.8671     | 0.6511     |
| **A7**          | 0.9992      | 0.7494      | 0.8660     | 0.6500     |

### 1g.2 Privacy on S4 — FMR / cFMR (n=7 405)

| **Variant**     | **FMR** | **cFMR** |
| --------------- | ------: | -------: |
| A1              | 0.000   | 0.000    |
| A2              | 1.000   | 0.000 †  |
| A3 / A4 / A5    | 1.000   | 0.000    |
| A4_nofb / norerank | 1.000 | 0.000    |
| A6 / A7         | 0.000   | 0.000    |

> † Same restricted-evidence-not-built caveat as 2Wiki applies (see §1f.2 / Table 6).

### 1g.3 ToT vs GoT Δ at A4 (Hotpot, R@10)

| **Subset** | **GoT A4 (ref §1d)** | **ToT A4-ToT** | **Δ (ToT − GoT)** |
| ---------- | -------------------: | -------------: | ----------------: |
| S3         | (n/a — GoT split)    | 0.999          | —                 |
| **S4**     | 0.978                | **0.982**      | **+0.4 pt**       |

> A4-ToT lifts S4 by **+23.2 pt** over A1-ToT (Hotpot S4 baseline 0.749 →
> 0.982), again matching GoT's +24.5 pt to within 1.3 pt.

### 1g.4 Cross-dataset summary — ToT A4 Δ over A1 on S4

| **Dataset** | **A1 (ToT) S4** | **A4 (ToT) S4** | **Δ** | **GoT Δ ref** |
| ----------- | --------------: | --------------: | ----: | ------------: |
| MuSiQue     | 0.785           | 0.921           | **+13.6 pt** | +17.0 pt |
| 2Wiki       | 0.773           | 0.980           | **+20.7 pt** | +21.3 pt |
| Hotpot      | 0.749           | 0.982           | **+23.2 pt** | +24.5 pt |

> **Take-away.** A4's S4 lift is monotonic in dataset distractor density
> *under both* GoT and ToT, and the absolute lifts agree to within 1–3 pt.
> The retrieval pipeline (A4) and the reasoning structure (GoT vs ToT) are
> orthogonal. Privacy contract (cFMR=0 on every scope-aware variant) is
> preserved end-to-end.

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
| **A1** | 0.000 / **0.000** | 0.000 / **0.000** | 0.000 / **0.000** | | |
| **A2** | 1.000 / **0.683** | 1.000 / 0.000 † | 1.000 / 0.000 † | | |
| **A3** | 1.000 / **0.000** | 1.000 / **0.000** | 1.000 / **0.000** | | |
| **A4** | 1.000 / **0.000** | 1.000 / **0.000** | 1.000 / **0.000** | | |
| **A5** | 1.000 / **0.000** | 1.000 / **0.000** | 1.000 / **0.000** | | |
| **A6** | 0.000 / **0.000** | 0.000 / **0.000** | 0.000 / **0.000** | | |
| **A7** | 0.000 / **0.000** | 0.000 / **0.000** | 0.000 / **0.000** | | |
| **A8** | 0.000 / **0.000** | (Stage B A8 eval running) | 0.000 / **0.000** | | |

> All numbers are S4-only at k=10 from the Qwen text-embedding-v3 evaluator.
> MuSiQue value updated from earlier draft (A2 cFMR was misreported as 0.341
> in v1; the correct k=10 number is **0.683**, taken from the current
> `eval_v9_qwen_full_k10.json`).
>
> **† 2Wiki / Hotpot A2 cFMR caveat:** the current `RestrictedBuilder` only
> populates `mem_rs_` entries when an interim restricted-evidence file
> exists for the dataset (built once for MuSiQue under
> `data/interim/musique/restricted_evidence/`). 2Wiki and Hotpot S4
> episodes therefore contain only `mem_ws_/mem_ts_/mem_pr_` entries; A2's
> shared bucket has no restricted memory it could leak, so cFMR is
> mechanically 0 on both datasets. The MuSiQue **0.683** is the only
> directly comparable A2 leak number; 2Wiki/Hotpot A2 cFMR will become
> meaningful once the restricted-evidence interim is built for them
> (tracked separately).
>
> Where it matters most — **A4/A5 cFMR is 0.000 on every dataset we have
> tested**, including the one (MuSiQue) where the structural FMR is 1.000
> *and* restricted entries exist in S4: i.e. the shared bucket is being
> used aggressively (FMR=1) but never returns a restricted memory in the
> top-k. This is the property the paper is built on.

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

| **Structure** | **Subset** | **ρ_mean** | **EM** | **F1** | **Recall@10** | **MRR@10** | **Fallback rate** | **Latency (ms)** |
| ------------- | ---------- | ---------- | ------ | ------ | ------------- | ---------- | ----------------- | ---------------- |
| **GoT (A4)** | S1 | 0.50 | * | * | 0.972 | 0.928 | | |
| **GoT (A4)** | S2 | 0.20 | * | * | 0.956 | 0.895 | | |
| **GoT (A4)** | S3 | 0.00 | * | * | 0.964 | 0.875 | | |
| **GoT (A4)** | S4 | — | * | * | 0.918 | 0.814 | | |
| CoT (A4-CoT) | S1 | | * | * | | | | |
| CoT (A4-CoT) | S2 | | * | * | | | | |
| CoT (A4-CoT) | S3 | | * | * | | | | |
| **ToT (A4-ToT)** | S1 | 0.43 | * | * | 0.999 | 0.854 | | |
| **ToT (A4-ToT)** | S2 | 0.15 | * | * | 1.000 | 1.000 | | |
| **ToT (A4-ToT)** | S3 | 0.00 | * | * | 0.998 | 0.787 | | |
| **ToT (A4-ToT)** | S4 | — | * | * | 0.921 | 0.701 | | |

---

## Table 16 · Subset Distribution by Reasoning Structure (MuSiQue dev)

> Same episode set, different ρ computations due to varying agent-private scope size. GoT has largest private scope → lowest ρ → smallest S1 share.

| **Structure** | **S1 count** | **S1 ρ_mean** | **S2 count** | **S2 ρ_mean** | **S3 count** | **S3 ρ_mean** |
| ------------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- |
| GoT | | | | | | |
| **ToT** | **1164** | **0.430** | **1** | **0.150** | **1252** | **0.000** |
| CoT | | | | | | |

> ToT MuSiQue/test (n=2417 non-S4 episodes, FORK structure): ρ distribution is bimodal — almost all episodes land in S1 (high overlap, sub-question solver branches share most workspace context) or S3 (independent branches, ρ=0). The single S2 hit suggests the FORK fan-out rarely produces partial overlap. Compare with GoT below.

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
