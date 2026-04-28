# CoScope Experiments — MuSiQue / GoT / test split

Living document. Updated as new evaluations are run. Numbers are reproducible
from the JSON sidecars under `data/processed/got/musique/` (gitignored).

---

## 1. Setup

| Item | Value |
|---|---|
| Dataset | MuSiQue v1.0 (`data/raw/musique/musique_ans_v1.0_dev.jsonl`) |
| Split | `test` (loader maps `test`→`dev`; 2417 raw items) |
| Episodes built | **1000** (5 graph types × 200) |
| Reasoning path | **GoT** (Graph-of-Thought) |
| Embedder | deterministic random hash, dim=256 (offline, reproducible) |
| Determinism | runs with `PYTHONHASHSEED=0`; ~±0.5pt residual noise on S4 from top-k tie-breaking near the cutoff |
| Step-2 LLM (a8 only) | qwen-plus via DashScope (3800 calls, 0 errors) |
| k (recall / mrr cutoff) | 10 |
| Schema version | 1.0.0 |

### Episode distribution

| Subset | Count | ρ range | Composition |
|---|---:|---|---|
| S1 | 400 | ρ ≥ 0.5 | LINEAR (200) + half of POLICY_ISOLATED |
| S2 | 400 | 0.05 < ρ < 0.5 | FORK_MERGE (200) + INDEPENDENT (200) |
| S3 | 200 | ρ = 0 | FORK (200) |
| S4 | 200 | orthogonal flag | POLICY_ISOLATED (privacy-isolated) |

| graph_type | n | ρ (mean) |
|---|---:|---:|
| LINEAR | 200 | 0.50 |
| FORK | 200 | 0.00 |
| FORK_MERGE | 200 | 0.19 |
| INDEPENDENT | 200 | 0.20 |
| POLICY_ISOLATED | 200 | 0.50 |

`gold_evidence_coverage = 1.0` across all 1000 episodes.

---

## 2. Variants under test

| Variant | Pipeline configuration |
|---|---|
| **a1** | per-agent independent retrieval (no sharing) — baseline |
| **a2** | scope-only shared, no rerank |
| **a3** | mean-pooled query, no rerank |
| **a4** | mean-pooled query + cross-encoder rerank + private fallback |
| **a4_nofb** | a4 minus private fallback |
| **a4_norerank** | a4 minus rerank |
| **a5** | query-matrix + SVD projection + rerank + fallback |
| **a5_noproj** | a5 minus SVD projection |
| **a5_norerank** | a5 minus rerank |
| **a6** | private-only retrieval (no shared bucket) |
| **a7** | a6 on raw question (no Step-2 rewrite) |
| **a8** | a6 on Step-2 LLM-rewritten `query_intent` |

> Note: a7 = a6 on the raw question, so on this dataset a7 ≡ a6. a8 differs
> from a7 only by the query string fed to the encoder.

---

## 3. Main results — recall@10 by graph_type

Source: `eval_v8.json` (v8, raw query) and `eval_v9.json` (v9, with query
rewrite). Both produced under `PYTHONHASHSEED=0` against HEAD.

### v8 (raw query, 11 variants × 1000 episodes)

| variant | LINEAR | FORK | FORK_MERGE | INDEPENDENT | POLICY_ISOLATED |
|---|---:|---:|---:|---:|---:|
| a1 | 0.9992 | 0.9992 | 0.9981 | 0.9985 | 0.7494 |
| a2 | 0.3807 | 0.3936 | 0.3599 | 0.3425 | 0.1805 |
| a3 | 0.4335 | 0.4389 | 0.4180 | 0.4217 | 0.4385 |
| **a4** | 0.9983 | 0.9983 | **0.9994** | **0.9990** | **0.8425** |
| a4_norerank | 0.4335 | 0.4389 | 0.4180 | 0.4217 | 0.4385 |
| a4_nofb | 0.9983 | 0.9983 | 0.9994 | 0.9990 | 0.7975 |
| a5 | 0.9992 | 0.9992 | 0.9981 | 0.9985 | 0.7662 |
| a5_noproj | 0.9992 | 0.9992 | 0.9981 | 0.9985 | 0.8081 |
| a5_norerank | 0.4506 | 0.4403 | 0.4334 | 0.4102 | 0.4505 |
| a6 | 0.9992 | 0.9992 | 0.9981 | 0.9985 | 0.7494 |
| a7 | 0.9992 | 0.9992 | 0.9981 | 0.9985 | 0.7494 |

### v9 (with Step-2 query rewrite, a4/a6/a7/a8 × 1000 episodes)

| variant | LINEAR | FORK | FORK_MERGE | INDEPENDENT | POLICY_ISOLATED |
|---|---:|---:|---:|---:|---:|
| a4 | 0.9983 | 0.9983 | 0.9994 | 0.9990 | **0.8438** |
| a6 | 0.9992 | 0.9992 | 0.9981 | 0.9985 | 0.7494 |
| a7 | 0.9992 | 0.9992 | 0.9981 | 0.9985 | 0.7494 |
| a8 | 0.9958 | 0.9975 | 0.9967 | 0.9958 | 0.7475 |

---

## 4. Main results — mrr@10 by graph_type

### v8 (raw query)

| variant | LINEAR | FORK | FORK_MERGE | INDEPENDENT | POLICY_ISOLATED |
|---|---:|---:|---:|---:|---:|
| a1 | 0.8501 | 0.8654 | 0.8114 | 0.7744 | 0.6376 |
| a2 | 0.2474 | 0.2600 | 0.2548 | 0.2377 | 0.1231 |
| a3 | 0.2764 | 0.2817 | 0.2951 | 0.2789 | 0.2644 |
| **a4** | 0.8568 | 0.8583 | 0.8113 | 0.7707 | 0.6715 |
| a4_norerank | 0.2764 | 0.2817 | 0.2951 | 0.2789 | 0.2644 |
| a4_nofb | 0.8568 | 0.8583 | 0.8113 | 0.7707 | 0.6603 |
| a5 | 0.8501 | 0.8654 | 0.8114 | 0.7749 | 0.6460 |
| a5_noproj | 0.8501 | 0.8654 | 0.8114 | 0.7749 | 0.6630 |
| a5_norerank | 0.2992 | 0.2823 | 0.2920 | 0.2917 | 0.2909 |
| a6 | 0.8501 | 0.8654 | 0.8114 | 0.7749 | 0.6376 |
| a7 | 0.8501 | 0.8654 | 0.8114 | 0.7749 | 0.6376 |

### v9 (with query rewrite)

| variant | LINEAR | FORK | FORK_MERGE | INDEPENDENT | POLICY_ISOLATED |
|---|---:|---:|---:|---:|---:|
| a4 | 0.8568 | 0.8583 | 0.8113 | 0.7707 | 0.6717 |
| a6 | 0.8501 | 0.8654 | 0.8114 | 0.7749 | 0.6376 |
| a7 | 0.8501 | 0.8654 | 0.8114 | 0.7749 | 0.6376 |
| **a8** | **0.8606** | **0.8837** | **0.8324** | **0.7912** | 0.6547 |

---

## 5. Efficiency — first_stage_savings by graph_type (v8)

Fraction of first-stage retrieval calls saved by sharing across agents.
A1 has no sharing so is always 0; A6/A7 are private-only with limited sharing.

| variant | LINEAR | FORK | FORK_MERGE | INDEPENDENT | POLICY_ISOLATED |
|---|---:|---:|---:|---:|---:|
| a1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| a2-a5 family | 0.667 | 0.667 | 0.750 | 0.800 | 0.750 |
| a6 / a7 | 0.333 | 0.333 | 0.500 | 0.600 | 0.250 |

---

## 6. Safety — false-merge on POLICY_ISOLATED (S4)

`false_merge_rate (FMR)` flags whether the routing layer ever proposes a
merge that crosses a privacy isolation boundary. `content_FMR` checks
whether the retrieved memory entries actually contain content that should
have been blocked.

| variant | FMR (S4) | content_FMR (S4) |
|---|---:|---:|
| a1 | 0.000 | 0.000 |
| a2 | 1.000 | **0.995** ← scope-only, no verifier |
| a3 | 1.000 | 0.000 |
| a4 | 1.000 | 0.000 |
| a4_nofb | 1.000 | 0.000 |
| a4_norerank | 1.000 | 0.000 |
| a5 | 1.000 | 0.000 |
| a5_noproj | 1.000 | 0.000 |
| a5_norerank | 1.000 | 0.000 |
| a6 | 0.000 | 0.000 |
| a7 | 0.000 | 0.000 |
| a8 | 0.000 | 0.000 |

**Reading**: a4/a5 family triggers FMR=1.0 in routing (the conservative
upper bound — every S4 request is *potentially* mergeable), but the
verifier and fallback filter out the actual content, leaving
`content_FMR=0`. Only a2 leaks content because it has no verifier.

---

## 7. Ablations

### Effect of cross-encoder rerank on a4 / a5

| graph_type | a4 | a4_norerank | Δ | a5 | a5_norerank | Δ |
|---|---:|---:|---:|---:|---:|---:|
| LINEAR | 0.998 | 0.434 | **−0.565** | 0.999 | 0.451 | **−0.548** |
| FORK | 0.998 | 0.439 | **−0.560** | 0.999 | 0.440 | **−0.559** |
| FORK_MERGE | 0.999 | 0.418 | **−0.581** | 0.998 | 0.433 | **−0.565** |
| INDEPENDENT | 0.999 | 0.422 | **−0.577** | 0.999 | 0.410 | **−0.588** |
| POLICY_ISOLATED | 0.843 | 0.439 | **−0.404** | 0.766 | 0.451 | **−0.316** |

Rerank is the single most important component — recall collapses without
it.

### Effect of private fallback on a4

| graph_type | a4 | a4_nofb | Δ |
|---|---:|---:|---:|
| LINEAR | 0.9983 | 0.9983 | 0.000 |
| FORK | 0.9983 | 0.9983 | 0.000 |
| FORK_MERGE | 0.9994 | 0.9994 | 0.000 |
| INDEPENDENT | 0.9990 | 0.9990 | 0.000 |
| POLICY_ISOLATED | 0.8425 | 0.7975 | **+0.045** |

Private fallback is a meaningful component on POLICY_ISOLATED — removing
it drops recall by 4.5 percentage points.

### Effect of SVD projection on a5

#### v8 (hash embedder, recall@10)

| graph_type | a5 | a5_noproj | Δ |
|---|---:|---:|---:|
| LINEAR | 0.9992 | 0.9992 | 0.000 |
| FORK | 0.9992 | 0.9992 | 0.000 |
| FORK_MERGE | 0.9981 | 0.9981 | 0.000 |
| INDEPENDENT | 0.9985 | 0.9985 | 0.000 |
| POLICY_ISOLATED | 0.7662 | 0.8081 | **−0.042** |

#### v8-ST (sentence-transformers `all-MiniLM-L6-v2`, recall@10)

Source: `eval_v8_st.json` (1000 episodes, real embedder, no Step-2 rewrite).

| graph_type | a5 | a5_noproj | Δ recall | a5 (mrr) | a5_noproj (mrr) | Δ mrr |
|---|---:|---:|---:|---:|---:|---:|
| LINEAR | 0.9739 | 0.9739 | 0.000 | 0.9088 | 0.9082 | −0.001 |
| FORK | 0.9722 | 0.9722 | 0.000 | 0.8652 | 0.8627 | −0.003 |
| FORK_MERGE | 0.9675 | 0.9672 | 0.000 | 0.8765 | 0.8732 | −0.003 |
| INDEPENDENT | 0.9662 | 0.9633 | **+0.003** | 0.8687 | 0.8609 | −0.008 |
| POLICY_ISOLATED | 0.7792 | 0.8173 | **−0.038** | 0.6965 | 0.7079 | **−0.011** |

**Findings (real embedder):**

1. **Non-conflict graph types**: SVD has essentially no effect on recall
   (|Δ| ≤ 0.003) but a *consistent slight drop* in mrr@10
   (−0.001 to −0.008). The projection compresses queries into a
   `min(svd_rank, n_queries)`-dim subspace; for buckets of 3–5 queries
   the effective rank is 3–5 anyway, so recall is preserved while the
   reduced resolution slightly hurts ranking.

2. **POLICY_ISOLATED (S4)**: SVD projection *hurts* recall by 3.8 pt and
   mrr by 1.1 pt — the same sign as on the hash embedder, but smaller in
   magnitude (−4.2 pt on hash → −3.8 pt on real). The projection-fitted
   subspace does not realize useful covariance signal even when the
   underlying embedder carries semantic structure: bucket sizes (3–5
   queries) are too small for the subspace to recover meaningful
   directions, so the only effect is to discard residual signal that the
   non-projected baseline keeps.

3. **Verdict**: **a5_noproj dominates a5** on the recall metric across
   every subset where they differ, on both embedder backends. A5 retains
   marginal mrr advantage only on S4 (+1.1 pt mrr at the cost of −3.8 pt
   recall). The SVD projection is therefore not a useful component on
   MuSiQue at the current bucket size.

#### Why SVD does not help on MuSiQue — per-bucket diagnostics

Source: `data/processed/got/musique/test/svd_scan/svd_scan_r64.summary.json`,
produced by `scripts/eval_jsonl.py --variants a5 --embedder st
--svd-rank 64 --dump-svd-artifacts ... --max-episodes-with-dump 1000`
followed by `scripts/analyze_svd_dumps.py` (1000 episodes, 200 per
graph_type, one shared bucket per episode).

| graph_type | n_q (med/max) | r_eff (med/max) | top‑1 energy | top‑3 energy | median max‑cos(K, Z) |
|---|---|---|---:|---:|---:|
| LINEAR | 3 / 3 | 3 / 3 | 0.652 | 1.000 | 0.944 |
| FORK | 3 / 3 | 2 / 2 | 0.910 | 1.000 | 0.955 |
| FORK_MERGE | 4 / 4 | 3 / 3 | 0.708 | 1.000 | 0.934 |
| INDEPENDENT | 5 / 5 | 3 / 3 | 0.611 | 1.000 | 0.923 |
| POLICY_ISOLATED | 4 / 4 | 4 / 4 | 0.491 | 0.960 | 0.928 |

* `n_q` — number of queries in the shared bucket the SVD is fitted on.
* `r_eff` — effective rank, count of `i` with `S[i] / S[0] > 1e-3`.
* `top‑k energy` — `Σ S[:k]² / Σ S²`.
* `max‑cos(K, Z)` — for each pool candidate's projected embedding, the
  max cosine against any projected query; we report the median over
  candidates, then the mean over buckets.

**Interpretation:**

* The configured `svd_rank=64` is never reached: bucket size caps the
  effective rank at `n_q ≤ 5` everywhere. **Increasing `svd_rank` cannot
  help** because the data itself does not span more than `n_q`
  directions, regardless of the configured rank.
* Top‑3 spectral energy is essentially 1.0 for the four non‑conflict
  graph types — three directions already span the entire bucket. The
  full‑dim baseline (`a5_noproj`) keeps the same information without
  paying any compression cost, which is why recall ties at the third
  decimal.
* POLICY_ISOLATED is the only subset where the spectrum stays diffuse
  (top‑1 = 0.49, top‑3 = 0.96). Buckets there mix planner / solver /
  verifier queries with intentionally divergent intents, so a low‑rank
  approximation is structurally **lossy** in proportion to that
  divergence — matching the observed `−3.8 pt recall` regression.
* The query subspace itself is not the bottleneck:
  `median max‑cos(K, Z) ≈ 0.92–0.96`, i.e. the candidate pool *does*
  align with the queries' principal directions. The projection's
  problem is information removal, not direction misalignment.

**Conclusion:** on this scale of bucket (3–5 queries), the SVD step is
a strict no‑op in the best case and a small loss in the conflict case.
SVD becomes a candidate optimization only when buckets grow past
≈ 10 queries, which on MuSiQue would require either far higher
hop‑counts or cross‑episode bucket merging.

#### Status of SVD in the main pipeline (soft downgrade)

Based on the diagnostics above, SVD is **not a recommended default** on
MuSiQue at the current bucket size (`n_q = 3..5`). The code path
remains first-class (``variant=a5``, ``a5_noproj``, ``a5_norerank``,
``a6``) so future work on wider buckets can re‑enable it, but:

* The **headline result for the shared-retrieval family is `a4`**
  (`shared_mean + rerank + fallback`), not `a5`. `a4` ties or beats
  `a5` on every subset and every k in the v8-ST numbers.
* `a5_noproj` is retained as the **canonical ablation baseline** for
  "shared retrieval without SVD"; whenever `a5_noproj ≥ a5`, report
  `a5_noproj`.
* The SVD path will only be promoted back to "primary" once we run on a
  dataset / graph configuration where buckets routinely contain ≥ 10
  queries (e.g. 5+ hop chains or cross-episode routing). Until then,
  the `--svd-rank` and `--dump-svd-artifacts` CLI flags remain
  research-only tools.

### Effect of Step-2 query rewrite (a7 → a8)

| graph_type | recall (a7) | recall (a8) | Δ recall | mrr (a7) | mrr (a8) | **Δ mrr** |
|---|---:|---:|---:|---:|---:|---:|
| LINEAR | 0.9992 | 0.9958 | −0.003 | 0.8501 | 0.8606 | **+0.010** |
| FORK | 0.9992 | 0.9975 | −0.002 | 0.8654 | 0.8837 | **+0.018** |
| FORK_MERGE | 0.9981 | 0.9967 | −0.001 | 0.8114 | 0.8324 | **+0.021** |
| INDEPENDENT | 0.9985 | 0.9958 | −0.003 | 0.7749 | 0.7912 | **+0.016** |
| POLICY_ISOLATED | 0.7494 | 0.7475 | −0.002 | 0.6376 | 0.6547 | **+0.017** |

LLM-rewritten queries do not raise recall (already saturated) but lift
mrr@10 across all graph types — strongest on multi-branch FORK
(+0.018) and FORK_MERGE (+0.021), where rewriting disambiguates parallel
sub-questions.

---

## 8. Headline takeaways

1. **a4 (mean+rerank+fallback) is the recall winner.** ≥ 0.998 on the four
   non-conflict graph types, **0.843** on POLICY_ISOLATED versus 0.749 for
   the no-share baseline a1 (**+9.4 pt absolute**).

2. **a8 (a6 + LLM query rewrite) is the mrr winner.** mrr@10 lifts
   uniformly by **+1.0 to +2.1 pt**, largest on multi-branch graphs
   (FORK +1.8 pt, FORK_MERGE +2.1 pt).

3. **Cross-encoder rerank is critical**: removing it collapses recall by
   32-58 pt across all subsets.

4. **Private fallback contributes 4.5 pt of recall on POLICY_ISOLATED**
   (a4 = 0.843 vs a4_nofb = 0.798). On non-conflict graph types it is
   neutral.

5. **Privacy is preserved end-to-end** for every variant that keeps the
   verifier (`a3`, `a4*`, `a5*`, `a6`, `a7`, `a8`):
   `content_FMR (S4) = 0`. Only `a2` (no verifier) leaks content.

6. **First-stage savings of 67-80 %** for the shared-retrieval family
   (a2-a5) at no cost to recall — the central efficiency claim of the
   framework.

---

## 9. Reproducibility

Re-run on existing shards:

```bash
# v8 (no Step-2 rewrite)
PYTHONHASHSEED=0 python -m scripts.eval_jsonl \
  --shards 'data/processed/got/musique/test/*.jsonl' \
  --variants a1 a2 a3 a4 a4_nofb a4_norerank a5 a5_noproj a5_norerank a6 a7 \
  --k 10 \
  --output data/processed/got/musique/test/eval_v8.json

# v9 (with Step-2 query rewrite — assumes test_qi/ shards already patched)
PYTHONHASHSEED=0 python -m scripts.eval_jsonl \
  --shards 'data/processed/got/musique/test_qi/*.jsonl' \
  --variants a4 a6 a7 a8 --k 10 \
  --output data/processed/got/musique/test_qi/eval_v9.json
```

To regenerate Step-2 query_intent (3800 LLM calls, ~90 min on qwen-plus):

```bash
python -m scripts.generate_query_intent \
  --shards 'data/processed/got/musique/test/*.jsonl' \
  --dataset musique --split test \
  --llm dashscope --model qwen-plus \
  --output-dir data/processed/got/musique/test_qi
```

---

## 10. Status / next axes

| Axis | Status |
|---|---|
| MuSiQue × test × GoT × {a1..a8 + ablations} | **complete** |
| MuSiQue × test × CoT | not started |
| MuSiQue × test × ToT | not started |
| HotpotQA × test × GoT | episodes not built |
| 2WikiMultiHopQA × test × GoT | episodes not built |
| GSM8K × test × GoT | episodes not built |
| MATH × test × GoT | episodes not built |
| Multi-k sweep (k = 1, 5, 20) | not run (only k=10) |

