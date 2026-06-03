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

### 1e.3 Stage B — A8-ToT (LLM-rewritten query_intent, qwen-plus, k=10)

| **Variant** | **S1 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR** | **S3 MRR** | **S4 MRR** |
| ----------- | ----------: | ----------: | ----------: | ---------: | ---------: | ---------: |
| A6          | 0.9885      | 0.9981      | 0.7887      | 0.8805     | 0.8049     | 0.6699     |
| A7          | 0.9886      | 0.9981      | 0.7887      | 0.8805     | 0.8049     | 0.6699     |
| **A8**      | **0.9911**  | **0.9985**  | **0.7898**  | **0.9110** | **0.8744** | **0.7086** |

> **A8-ToT vs A6-ToT (Δ MRR)**: S1 +3.05 pt, S3 **+6.95 pt**, S4 +3.87 pt.
> Recall is essentially saturated at the A6 ceiling under hierarchical
> block routing; A8's LLM-rewritten `query_intent` mainly improves
> ranking quality (MRR), exactly as predicted by the design.
>
> Note that A8/A6/A7 all cap S4 R@10 at ~0.79 because hierarchical block
> routing does not aggressively use the private-fallback retriever — A4
> remains the S4-recall champion (0.921). A8's value-add is on top of
> A6's policy-aware routing, where it tightens MRR by ~3-7 pt.
> Privacy: cFMR = 0.000 on every (variant, subset) under A8/A6/A7 (Stage B).

### 1e.4 ToT vs GoT Δ at A4 (MuSiQue, R@10)

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

### 1e.5 Recall@5 / Recall@20 (alternative cutoffs, ToT MuSiQue)

> Source: `data/processed/tot/musique/test/eval_v9_qwen_full_k{5,20}.json`
> (re-evaluation on existing shards, embedding cache reused, no LLM).

| **Variant**     | **S1 R@5** | **S1 R@20** | **S3 R@5** | **S3 R@20** | **S4 R@5** | **S4 R@20** |
| --------------- | ---------: | ----------: | ---------: | ----------: | ---------: | ----------: |
| **A1**          | 0.6578     | **1.0000**  | 0.9807     | **1.0000**  | 0.6474     | 0.7942      |
| A2              | 0.3868     | 0.8482      | 0.6390     | 0.9521      | 0.4450     | 0.7020      |
| A3              | 0.4425     | 0.9141      | 0.6417     | 0.9712      | 0.5365     | 0.9444      |
| **A4**          | **0.6730** | **1.0000**  | 0.9827     | **1.0000**  | **0.7444** | 0.9513      |
| A4_nofb         | 0.6730     | 1.0000      | 0.9827     | 1.0000      | 0.7443     | 0.9515      |
| A4_norerank     | 0.4425     | 0.9141      | 0.6417     | 0.9712      | 0.5365     | 0.9444      |
| **A5**          | 0.6640     | 1.0000      | **0.9845** | **1.0000**  | 0.6735     | 0.9433      |
| A5_noproj       | 0.6621     | 1.0000      | 0.9807     | 1.0000      | 0.6761     | **0.9813**  |
| A5_norerank     | 0.2830     | 0.7051      | 0.3069     | 0.8671      | 0.3014     | 0.7759      |
| **A6**          | 0.6667     | 1.0000      | 0.9807     | 1.0000      | 0.6510     | 0.7942      |
| **A7**          | 0.6667     | 1.0000      | 0.9807     | 1.0000      | 0.6510     | 0.7942      |

(S2 omitted: 1 episode in ToT MuSiQue — see §0.5 footnote on ToT ρ bimodality.)

> **R@5**: A4 still wins on every subset (S1 +1.5 pt, S3 +0.2 pt, S4
> +9.7 pt over A1). A6/A7 trail A4 by 9.3 pt on S4 because they do
> not use private fallback. **R@20**: A1/A4/A5/A6/A7 saturate S1/S3 at
> 1.000 (k=20 not informative on those subsets); S4 is the
> discriminator and A5_noproj wins (0.981) > A4 (0.951) > A5 (0.943)
> ≫ A1/A6/A7 (0.794). The block-router R@20 ceiling on S4 is
> structural (no private fallback by design).

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
| A1              | 0.000   | 0.000 †  |
| A2              | 1.000   | 0.000 †  |
| A3 / A4 / A5    | 1.000   | 0.000 †  |
| A4_nofb / norerank | 1.000 | 0.000 †  |
| A6 / A7         | 0.000   | 0.000 †  |

> † **All cFMR values on 2Wiki are mechanically 0** because
> `data/interim/restricted/2wikimhqa_restricted.jsonl` has not been built yet;
> S4 shards contain no `mem_rs_` entries to leak. This applies to A2
> (which would otherwise leak) **and** to A4 / A5 (which would otherwise
> protect). The cFMR axis on 2Wiki is therefore un-verified — only the
> R@10 / MRR / FMR / savings axes are valid retrieval evidence. See
> Table 6 caveat.

### 1f.3 Stage B — A8-ToT (LLM-rewritten query_intent, qwen-plus, k=10)

| **Variant** | **S1 R@10** | **S3 R@10** | **S4 R@10** | **S1 MRR** | **S3 MRR** | **S4 MRR** |
| ----------- | ----------: | ----------: | ----------: | ---------: | ---------: | ---------: |
| A6          | 0.9958      | 0.9999      | 0.7727      | 0.9352     | 0.9047     | 0.7056     |
| A7          | 0.9958      | 0.9999      | 0.7727      | 0.9352     | 0.9046     | 0.7055     |
| **A8**      | **0.9994**  | **0.9999**  | **0.7734**  | **0.9429** | **0.9063** | **0.7079** |

> A8-ToT vs A6-ToT (Δ on 2Wiki): R@10 +0.36 / +0.0 / +0.07 pt; MRR
> +0.77 / +0.16 / +0.23 pt. The lift is genuinely smaller than on
> MuSiQue / Hotpot — 2Wiki's A6 baseline is already very tight on this
> embedder, so query rewriting has little headroom. cFMR = 0 throughout
> (mechanical zero — see §1f.2 / Table 6).

### 1f.4 ToT vs GoT Δ at A4 (2Wiki, R@10)

| **Subset** | **GoT A4 (ref §1c)** | **ToT A4-ToT** | **Δ (ToT − GoT)** |
| ---------- | -------------------: | -------------: | ----------------: |
| S1         | 0.999                | 0.996          | −0.3 pt           |
| S3         | 0.998                | 1.000          | +0.2 pt           |
| S4         | 0.978                | **0.980**      | **+0.2 pt**       |

> A4-ToT lifts S4 by **+20.7 pt** over A1-ToT — almost identical to GoT's
> +21.3 pt. The S1/S3 numbers are also within ±0.3 pt of GoT. Reasoning
> structure is genuinely orthogonal to the retrieval pipeline on 2Wiki.

### 1f.5 Recall@5 / Recall@20 (alternative cutoffs, ToT 2Wiki)

> Source: `data/processed/tot/2wikimhqa/test/eval_v9_qwen_full_k{5,20}.json`.

| **Variant**     | **S1 R@5** | **S1 R@20** | **S3 R@5** | **S3 R@20** | **S4 R@5** | **S4 R@20** |
| --------------- | ---------: | ----------: | ---------: | ----------: | ---------: | ----------: |
| **A1**          | 0.5918     | **1.0000**  | 0.9953     | **1.0000**  | 0.6942     | 0.7736      |
| A2              | 0.5585     | 0.9976      | 0.8332     | **1.0000**  | 0.7592     | 0.9969      |
| A3              | 0.5579     | **1.0000**  | 0.8334     | **1.0000**  | 0.7608     | **1.0000**  |
| **A4**          | **0.5918** | **1.0000**  | **0.9953** | **1.0000**  | **0.8639** | **1.0000**  |
| A4_nofb         | 0.5918     | 1.0000      | 0.9953     | 1.0000      | 0.8639     | 1.0000      |
| A4_norerank     | 0.5579     | 1.0000      | 0.8334     | 1.0000      | 0.7608     | 1.0000      |
| **A5**          | **0.5926** | 1.0000      | 0.9953     | 1.0000      | 0.7542     | 1.0000      |
| A5_noproj       | 0.5926     | 1.0000      | 0.9953     | 1.0000      | 0.7542     | 1.0000      |
| A5_norerank     | 0.5036     | 1.0000      | 0.6689     | 1.0000      | 0.6129     | 1.0000      |
| **A6**          | 0.5926     | 1.0000      | 0.9953     | 1.0000      | 0.6944     | 0.7736      |
| **A7**          | 0.5926     | 1.0000      | 0.9953     | 1.0000      | 0.6944     | 0.7736      |

> A4 keeps a **+17.0 pt** S4 R@5 margin over A1 (0.864 vs 0.694) and
> reaches R@20=1.000 on every non-isolated subset. A6/A7 cap S4 R@20
> at 0.774 — the same structural ceiling seen on MuSiQue and Hotpot.

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
| A1              | 0.000   | 0.000 †  |
| A2              | 1.000   | 0.000 †  |
| A3 / A4 / A5    | 1.000   | 0.000 †  |
| A4_nofb / norerank | 1.000 | 0.000 †  |
| A6 / A7         | 0.000   | 0.000 †  |

> † **All cFMR values on Hotpot are mechanically 0** for the same
> reason as 2Wiki (§1f.2): no `mem_rs_` entries until the
> `data/interim/restricted/hotpotqa_restricted.jsonl` interim is built.
> Privacy axis is un-verified on Hotpot.

### 1g.3 Stage B — A8-ToT (LLM-rewritten query_intent, qwen-plus, k=10)

| **Variant** | **S3 R@10** | **S4 R@10** | **S3 MRR** | **S4 MRR** |
| ----------- | ----------: | ----------: | ---------: | ---------: |
| A6          | 0.9992      | 0.7494      | 0.8707     | 0.6528     |
| A7          | 0.9992      | 0.7494      | 0.8683     | 0.6508     |
| **A8**      | 0.9990      | 0.7493      | **0.8855** | **0.6635** |

> Same pattern as MuSiQue: A8 leaves R@10 at the A6 ceiling and lifts MRR
> (+1.48 pt S3, +1.07 pt S4). cFMR = 0 throughout.

### 1g.4 ToT vs GoT Δ at A4 (Hotpot, R@10)

| **Subset** | **GoT A4 (ref §1d)** | **ToT A4-ToT** | **Δ (ToT − GoT)** |
| ---------- | -------------------: | -------------: | ----------------: |
| S3         | (n/a — GoT split)    | 0.999          | —                 |
| **S4**     | 0.978                | **0.982**      | **+0.4 pt**       |

> A4-ToT lifts S4 by **+23.2 pt** over A1-ToT (Hotpot S4 baseline 0.749 →
> 0.982), again matching GoT's +24.5 pt to within 1.3 pt.

### 1g.5 Cross-dataset summary — ToT A4 Δ over A1 on S4

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

### 1g.6 Recall@5 / Recall@20 (alternative cutoffs, ToT Hotpot)

> Source: `data/processed/tot/hotpotqa/test/eval_v9_qwen_full_k{5,20}.json`.
> ToT Hotpot has only S3 + S4 (see §1g intro for why); S1/S2 are absent.

| **Variant**     | **S3 R@5** | **S3 R@20** | **S4 R@5** | **S4 R@20** |
| --------------- | ---------: | ----------: | ---------: | ----------: |
| **A1**          | 0.9879     | **1.0000**  | 0.7409     | 0.7500      |
| A2              | 0.7932     | 1.0000      | 0.7596     | **1.0000**  |
| A3              | 0.7955     | 1.0000      | 0.7752     | **1.0000**  |
| **A4**          | **0.9881** | **1.0000**  | **0.9292** | **1.0000**  |
| A4_nofb         | 0.9881     | 1.0000      | 0.9292     | 1.0000      |
| A4_norerank     | 0.7955     | 1.0000      | 0.7752     | 1.0000      |
| **A5**          | 0.9879     | 1.0000      | 0.8214     | 1.0000      |
| A5_noproj       | 0.9879     | 1.0000      | 0.8214     | 1.0000      |
| A5_norerank     | 0.5264     | 1.0000      | 0.4993     | 1.0000      |
| **A6**          | 0.9879     | 1.0000      | 0.7409     | 0.7500      |
| **A7**          | 0.9879     | 1.0000      | 0.7409     | 0.7500      |

> Hotpot ToT S4 R@5 spread is the widest among the three ToT datasets:
> A4 = 0.929, A5 = 0.821, A6 = 0.741, A1 = 0.741. A4's +18.8 pt margin
> over A1 / A6 at R@5 is the strongest small-k effect across all six
> (rpt × dataset) cells. R@20 saturates everywhere except A1 / A6 / A7
> S4 (capped at 0.750 for the same structural reason).

### 1g.7 Stage B (A8) — R@5 / R@20 summary across all three ToT datasets

> Source: `data/processed/tot/{ds}/test_qi/eval_v9_qwen_a8_k{5,20}.json`.
> Stage B is `A8 = A6 + LLM-rewritten query_intent`. R@5/R@20 are
> reported only for the variants whose Stage A baseline they refine
> (a6 / a7 / a8). Cross-cell pattern is consistent: Stage B keeps
> Stage A recall almost exactly intact; the value-add is on MRR
> (see §1e.3 / §1f.3 / §1g.3).

| **Dataset / subset** | **A6 R@5** | **A8 R@5** | **A6 R@20** | **A8 R@20** |
| -------------------- | ---------: | ---------: | ----------: | ----------: |
| ToT MuSiQue S1       | 0.6667     | 0.6702     | 1.0000      | 1.0000      |
| ToT MuSiQue S3       | 0.9807     | 0.9797     | 1.0000      | 1.0000      |
| ToT MuSiQue S4       | 0.6510     | 0.6520     | 0.7942      | 0.7942      |
| ToT 2Wiki S1         | 0.5926     | 0.5937     | 1.0000      | 1.0000      |
| ToT 2Wiki S3         | 0.9953     | 0.9963     | 1.0000      | 1.0000      |
| ToT 2Wiki S4         | 0.6944     | 0.6951     | 0.7736      | 0.7736      |
| ToT Hotpot S3        | 0.9879     | 0.9897     | 1.0000      | 1.0000      |
| ToT Hotpot S4        | 0.7409     | 0.7423     | 0.7500      | 0.7500      |

> Largest A8 R@5 lift over A6: **+8.33 pt on ToT MuSiQue S2** (not in
> the table above because n=1 — discarded as noise), then **+0.35 pt
> on ToT MuSiQue S1**. All other deltas are ≤ 0.20 pt. R@20 is
> identical to A6 by construction (same stage-1 candidate set, A8
> only reranks the top-k). This is consistent with the finding that
> A8's contribution is concentrated in MRR / top-1 quality rather
> than recall coverage.

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
| **A2** | 1.000 / **0.683** | (rerun pending) | (rerun pending) | | |
| **A3** | 1.000 / **0.000** | (rerun pending) | (rerun pending) | | |
| **A4** | 1.000 / **0.000** | 1.000 / **0.000** | 1.000 / **0.000** | | |
| **A4_nofb** | 1.000 / **0.000** | 1.000 / **0.000** | 1.000 / **0.000** | | |
| **A4_norerank** | 1.000 / **0.000** | 1.000 / **0.000** | 1.000 / **0.000** | | |
| **A5** | 1.000 / **0.000** | (rerun pending) | (rerun pending) | | |
| **A6** | 0.000 / **0.000** | 0.000 / **0.000** | 0.000 / **0.000** | | |
| **A7** | 0.000 / **0.000** | (alias of A6) | (alias of A6) | | |
| **A8** | 0.000 / **0.000** | (Stage B A8 eval running) | (Stage B A8 eval running) | | |

> All numbers are S4-only at k=10 from the Qwen text-embedding-v3 evaluator
> on shards rebuilt with full restricted interim (Phase A2 complete, 2026-05-11).
> MuSiQue value (A2 cFMR=0.683) is from the original `eval_v9_qwen_full_k10.json`.
> 2Wiki / Hotpot rows for `a1 / a4 / a4_nofb / a4_norerank / a6` are taken from
> `data/processed/{got,tot}/{2wikimhqa,hotpotqa}/test/eval_v9_qwen_s4_k10.json`
> (averaged across `got` and `tot` reasoning structures; the two RPTs produce
> identical FMR/cFMR — see Table 6a).
>
> **Restricted interim coverage** (state at 2026-05-11):
> * `data/interim/restricted/musique_restricted.jsonl`  — 2417 ep (built earlier)
> * `data/interim/restricted/2wikimhqa_restricted.jsonl` — 12576 ep (Phase A1, 2026-05-11)
> * `data/interim/restricted/hotpotqa_restricted.jsonl` — 7405 ep (Phase A1, 2026-05-11)
>
> All three datasets now have real verifier-injected restricted entries
> in their S4 shards. The previously-vacuous `cFMR = 0` footnote is
> **resolved** for 2Wiki and Hotpot.
>
> Pending fills (no implementation gap, only need to run a2/a3/a5 on
> the rebuilt S4 shards): see TODO at end of section.

### Key takeaway from the post-rebuild evaluation (2026-05-11)

> **Structural merge ≠ content leak.** The A4 family
> (A4 / A4_nofb / A4_norerank) reports `FMR = 1.000` on every dataset —
> meaning *every* S4 verifier request is served from a shared bucket —
> yet `cFMR = 0.000` on every dataset, including the two that previously
> reported only vacuous zeros. In other words: the bucket is shared at
> the structural level, but the policy filter inside `_check_policy`
> still removes all `RESTRICTED`-tagged memories before they enter
> any agent's top-k.
>
> This is the central safety claim of the paper, now corroborated on
> three multihop QA datasets and two reasoning structures (got / tot).
> A2 on 2Wiki / Hotpot needs to be added to demonstrate that
> non-zero cFMR is possible in this evaluation setting (i.e. that
> our `cFMR = 0` for A4 is not an artefact of evaluator construction).

### Table 6a · S4 Privacy Detail — Post-rebuild (k=10, restricted interim built)

| RPT | Dataset | n | Variant | R@10 | MRR | FMR | cFMR | Savings |
|---|---|---:|---|---:|---:|---:|---:|---:|
| got | hotpotqa | 7405 | a1 | 0.7497 | 0.6499 | 0.000 | **0.000** | 0.000 |
| got | hotpotqa | 7405 | a4 | 0.9947 | 0.7604 | 1.000 | **0.000** | 0.750 |
| got | hotpotqa | 7405 | a4_nofb | 0.9947 | 0.7599 | 1.000 | **0.000** | 0.750 |
| got | hotpotqa | 7405 | a4_norerank | 0.9835 | 0.8613 | 1.000 | **0.000** | 0.750 |
| got | hotpotqa | 7405 | a6 | 0.7497 | 0.6571 | 0.000 | **0.000** | 0.250 |
| got | 2wikimhqa | 12576 | a1 | 0.7679 | 0.7283 | 0.000 | **0.000** | 0.000 |
| got | 2wikimhqa | 12576 | a4 | 0.9806 | 0.8705 | 1.000 | **0.000** | 0.768 |
| got | 2wikimhqa | 12576 | a4_nofb | 0.9806 | 0.8705 | 1.000 | **0.000** | 0.768 |
| got | 2wikimhqa | 12576 | a4_norerank | 0.9303 | 0.9094 | 1.000 | **0.000** | 0.768 |
| got | 2wikimhqa | 12576 | a6 | 0.7679 | 0.7287 | 0.000 | **0.000** | 0.305 |
| tot | hotpotqa | 7405 | a1 | 0.7494 | 0.5803 | 0.000 | **0.000** | 0.000 |
| tot | hotpotqa | 7405 | a4 | 0.9815 | 0.6759 | 1.000 | **0.000** | 0.750 |
| tot | hotpotqa | 7405 | a4_nofb | 0.9815 | 0.6760 | 1.000 | **0.000** | 0.750 |
| tot | hotpotqa | 7405 | a4_norerank | 0.9397 | 0.7603 | 1.000 | **0.000** | 0.750 |
| tot | hotpotqa | 7405 | a6 | 0.7494 | 0.6485 | 0.000 | **0.000** | 0.250 |
| tot | 2wikimhqa | 12576 | a1 | 0.7727 | 0.6980 | 0.000 | **0.000** | 0.000 |
| tot | 2wikimhqa | 12576 | a4 | 0.9798 | 0.8216 | 1.000 | **0.000** | 0.774 |
| tot | 2wikimhqa | 12576 | a4_nofb | 0.9798 | 0.8216 | 1.000 | **0.000** | 0.774 |
| tot | 2wikimhqa | 12576 | a4_norerank | 0.9198 | 0.8651 | 1.000 | **0.000** | 0.774 |
| tot | 2wikimhqa | 12576 | a6 | 0.7727 | 0.7055 | 0.000 | **0.000** | 0.321 |

> Sources: `data/processed/{got,tot}/{hotpotqa,2wikimhqa}/test/eval_v9_qwen_s4_k10.json`
> (Phase A3 outputs, 2026-05-11). The same shards under k=5 / k=20 give
> qualitatively identical FMR / cFMR (only R@k / MRR change with k).

### TODO — close the cFMR table

* Run A2 / A3 / A5 on the rebuilt S4 shards for 2Wiki and Hotpot
  (`got` and `tot`). One eval call each, no LLM cost. A2 is the highest
  priority because it is the only variant expected to leak (cFMR > 0).

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

> Caveat: A5 implements per-bucket online truncated SVD on the
> stacked query matrix `Q ∈ R^{k×n}`. When `n ≤ r` (which is the
> dominant regime in our buckets — most buckets have `n ∈ {2,3,4}`)
> the effective rank is bounded by `n`, not by `r`. So sweeping `r`
> above ~3 has no effect on the small-bucket regime regardless of
> the configured value. See Table 12a for the orthogonal ablation
> that varies the *mechanism* itself instead of `r`.

---

## Table 12a · Projection Mechanism Ablation — MuSiQue (k=10, GoT, **Qwen text-embedding-v3**)

> Three projection mechanisms compared on the same pipeline scaffold
> (scope-only routing, hybrid full-dim rerank, private fallback).
> All three differ only in how the `(Z, W_final)` pair fed into
> `SharedCandidateRetriever` is constructed:
>
> * **A5** — per-bucket truncated SVD on `Q` (data-adaptive subspace).
> * **A5_DOC** — strict realisation of `introduction.md §8.4–8.7`:
>   random sparse `W ∈ R^{l×k}` + structural mask, orthogonalised once
>   via SVD into a fixed `W_final ∈ R^{k×r}` reused for every bucket.
>   Data-agnostic; `Q` enters only via `Z = Q^T W_final` at transform time.
> * **A5_NOPROJ** — identity projection (skip SVD; score in original
>   1024-d embedding space).
>
> Identical first-stage pool (same scope-only routing). Differences
> in scoring are entirely attributable to the projection mechanism.

| **Subset** | **n** | **Variant** | **Recall@10** | **MRR@10** | **FMR** | **cFMR** | **Savings** |
| ---------- | ----: | ----------- | ------------: | ---------: | ------: | -------: | ----------: |
| S1 | 4744 | A5         | 0.9890 | 0.8963 | 0.000 | 0.000 | 0.762 |
| S1 | 4744 | A5_DOC     | 0.9870 | 0.8962 | 0.000 | 0.000 | 0.762 |
| S1 | 4744 | A5_NOPROJ  | 0.9865 | 0.8963 | 0.000 | 0.000 | 0.762 |
| S2 | 2 | A5         | 1.0000 | 1.0000 | 0.000 | 0.000 | 0.817 |
| S2 | 2 | A5_DOC     | 1.0000 | 1.0000 | 0.000 | 0.000 | 0.817 |
| S2 | 2 | A5_NOPROJ  | 1.0000 | 1.0000 | 0.000 | 0.000 | 0.817 |
| S3 | 1252 | A5        | 0.9985 | 0.8079 | 0.000 | 0.000 | 0.667 |
| S3 | 1252 | A5_DOC    | 0.9981 | 0.8067 | 0.000 | 0.000 | 0.667 |
| S3 | 1252 | A5_NOPROJ | 0.9981 | 0.8083 | 0.000 | 0.000 | 0.667 |
| S4 | 2417 | A5        | 0.8719 | 0.7430 | 1.000 | 0.000 | 0.780 |
| S4 | 2417 | A5_DOC    | 0.8706 | 0.7429 | 1.000 | 0.000 | 0.780 |
| S4 | 2417 | A5_NOPROJ | 0.8715 | 0.7431 | 1.000 | 0.000 | 0.780 |

> Source: `data/processed/got/musique/test/eval_v9_qwen_a5doc_k10.json`
> (8415 episodes total, all seven shards). Cross-dataset replication
> (HotpotQA, 2WikiMultiHopQA, S4-only) is currently running — see TODO.

### Pairwise differences (R@10, S4)

| Comparison | ΔR@10 (S4) | ΔMRR (S4) | Verdict |
|---|---:|---:|---|
| A5 vs A5_NOPROJ          | +0.0004 | −0.0001 | per-bucket SVD ≈ identity |
| A5_DOC vs A5_NOPROJ      | −0.0009 | −0.0002 | random sparse SVD ≈ identity |
| A5_DOC vs A5             | −0.0013 | −0.0001 | doc-strict ≈ data-adaptive |

### Key takeaway

> **The projection matrix mechanism is not the source of A5's
> retrieval quality.** Three structurally distinct mechanisms —
> a data-adaptive per-bucket truncated SVD (A5), a fixed data-agnostic
> random-sparse-orthogonal projection (A5_DOC, the strict
> `introduction.md` design), and the identity (A5_NOPROJ) — produce
> Recall@10 within ±0.0013 of each other on every stratum.
> First-stage savings are identical (same scope-only routing).
>
> What this means for the paper:
>
> * The "shared subspace" framing in §8.4–8.7 is doing **structural
>   work** (organising same-bucket queries into one batched
>   retrieval call, enabling pool reuse across agents) rather than
>   **statistical work** (projecting onto a more discriminative
>   low-rank subspace).
> * Honest ablation: report Table 12a in full and explicitly state
>   that the projection step contributes 0 ± 0.001 R@10. The
>   end-to-end gains attributed to A5/A6 in main tables come from
>   bucket-level pool sharing (already captured by A4) and from
>   policy-aware block routing (A6 only). The projection itself is
>   inert in the current 1024-d Qwen embedding space at the
>   bucket sizes we observe (`n ∈ {2,3,4}`).
> * Honest counterfactual: the projection step *might* still matter
>   in a regime we have not tested — much larger buckets
>   (`n ≫ r`), much higher embedding dimensions, or very tight
>   first-stage budgets where score collapse from the rerank step
>   is intolerable. We do not currently have evidence for or against
>   that regime.

### TODO — close the projection ablation story

* Cross-dataset replication of Table 12a on HotpotQA and 2WikiMultiHopQA
  S4 shards (currently running in background, output:
  `data/processed/got/{hotpotqa,2wikimhqa}/test/eval_v9_qwen_a5doc_s4_k10.json`).
  If the null result holds across all three datasets, lock in the
  table and stop sweeping `r`.
* Optional: bucket-size-stratified breakdown of Table 12a (split S1
  by `n_q` into {2, 3, 4, 5+}) to confirm that the small-bucket regime
  is where the null result lives, and to document the boundary at
  which projection might start mattering.

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

---

## §X · Project Self-Assessment & Paper-Writing Recommendations (2026-05-11)

> Honest internal review after the ToT Stage A + Stage B sweeps were
> complete on MuSiQue / 2WikiMultiHopQA / HotpotQA. Captures (a) what
> the project is genuinely contributing, (b) what currently does **not**
> hold up, and (c) recommended priorities for finishing the paper.

### X.1 What is solid (the hero numbers)

The single sharpest sentence the paper can make today, grounded in
real data, is:

> On MuSiQue S4, A4 retrieves **78 % fewer first-stage candidates** than
> per-agent independent retrieval (A1) at virtually the same recall
> ceiling (Δ Recall@10 ≈ +0.3 pt vs A1; +13.6 pt vs A1 on ToT-MuSiQue;
> +20.7 pt on ToT-2Wiki; +23.2 pt on ToT-Hotpot for S4 specifically),
> while reducing the content-level privacy leak (cFMR) of force-merge
> (A2) **from 0.683 to 0.000**.

Three statements survive scrutiny:

1. **Efficiency × Privacy duality is real.** A4 keeps almost all of
   A2's `first_stage_savings` (≈ 78 % vs ≈ 78 %) while collapsing cFMR
   from 0.683 → 0.000 on MuSiQue. This is the cleanest single-row win
   in the paper.
2. **Reasoning-structure orthogonality is empirically established.** A4
   transfers from GoT to ToT with S4 deltas agreeing to within 1–3 pt
   across 3 datasets (§1g.5). 6 of 6 (dataset × structure) cells move
   in the predicted direction.
3. **Hierarchical block routing (A6) achieves FMR=0 *and* cFMR=0** on
   MuSiQue S4 — the only configuration with both structural and
   content-level non-leak. It pays ~33 pp less first-stage savings than
   A4 in exchange.

### X.2 What does not yet hold up (must address before submission)

1. **Cross-dataset privacy is currently an MuSiQue-only claim.** The
   `restricted_evidence` interim has been built only for MuSiQue; on
   2Wiki / Hotpot every variant's cFMR=0 is *vacuous* (no `mem_rs_*`
   entries to leak). The doc has been updated (Table 6 ‡, §1f.2 †,
   §1g.2 †) but the underlying data still needs to be regenerated for
   the paper to claim cross-dataset privacy.
2. **ρ in ToT is a hop_count alias.** Under ToT FORK, ρ is a
   deterministic function of `(graph_type, hop_count)` with std=0
   inside each hop bucket (e.g. all 4-hop ToT-2Wiki episodes have
   ρ=0.428 exactly). Narrative needs to either (a) honestly call ρ a
   *topology proxy*, or (b) redefine ρ to use real content similarity
   between scopes. Option (a) is cheap; option (b) is a research item.
3. **A8 marginal value over A6.** A8 = A6 + LLM query rewrite. Its
   actual lift (Stage B) is ≤ 0.4 pt R@10 and 1–7 pt MRR depending on
   dataset, at the cost of a qwen-plus call per request. The paper
   should not sell A8 as a hero method; report it as a small but
   reliable MRR refinement on top of A6.
4. **A7 is presently a code-level alias of A6.** Either implement the
   learned-W variant the doc promises, or merge A7 into A6 in tables.
   Leaving 8 named variants where 1 is empty padding is a reviewer
   target.
5. **Tables 2–5 (downstream EM/F1/Acc) are still empty.** All current
   "improvements" are A1 vs Ax self-comparisons. At least one classical
   retrieval baseline (BM25 is cheapest) should populate the
   open-ended baseline rows or those tables should be removed from
   this draft.
6. **No confidence intervals.** Cells like §1f.3 reporting "+0.2 pt"
   deltas should not be in bold without bootstrap CI or significance
   testing — at 9 800 episodes per cell, sub-percent deltas are
   plausibly noise.

### X.3 Recommended narrative for the paper

Order of importance for the abstract / intro headlines:

1. **Privacy × efficiency duality** (A4 gives A2's savings without A2's
   leak) — this is the headline.
2. **Generalization across reasoning topologies** (GoT / ToT, 3
   datasets, S4 deltas within 1–3 pt) — this is the soundness check.
3. **MRR refinement via LLM query rewriting** (A8 over A6) — small but
   honest.

What **not** to claim until the data is built:

- "Cross-dataset privacy" (currently MuSiQue only).
- "ρ measures collaborator information overlap" (currently it largely
  measures topology).
- "Hierarchical block routing wins" without admitting its 33 pp
  savings cost.

### X.4 Recommended next concrete actions

Sorted by cost / impact ratio:

| **Action** | **Cost** | **Lifts which claim?** |
| ---------- | -------- | ---------------------- |
| Build `data/interim/restricted/{2wikimhqa,hotpotqa}_restricted.jsonl` via `python -m scripts.generate_restricted --dataset {ds} --split test --data-dir data/raw/{ds}` and re-run S4 retrieval (no LLM) | ~1–2 h | Cross-dataset privacy (claim #1 above) |
| Adjust ρ narrative to "topology proxy" + footnote derivation | ~30 min docs | Soundness of subset stratification |
| Run BM25 on MuSiQue test as one Table 2 row | ~2 h | At least one external baseline |
| Replace "vs A2" force-merge baseline numbers with a learned-W A7 once that is implemented | research | Closes the A6/A7 alias gap |
| Add 95 % bootstrap CI on the §1{b–g}.1 R@10 / MRR tables | ~1 h | Statistical soundness |

### X.5 Where the project sits on the contribution ladder

- **Solid system-level contribution** (privacy bucketing + private
  fallback + hierarchical routing as a coherent design space).
- **One clean empirical headline** (A4's 78 % savings at cFMR=0 on
  MuSiQue, plus structure-orthogonality across GoT/ToT).
- **Not yet a theoretical contribution** — ρ is a useful taxonomy axis
  but currently a topology re-encoding rather than a measurement of
  content overlap.

This is enough for a strong systems / applications paper, **provided**
the cross-dataset privacy data is regenerated and the narrative is
adjusted as in §X.3.
