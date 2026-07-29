# Experiment Guide

Install the benchmark dependencies before running live experiments:

```bash
python -m pip install -e ".[dev,bench]"
```

All runners load the repository-root `.env`. Gold answers are withheld from
the runtime and used only by post-generation scoring. Benchmark corpus
paragraphs are marked `retrieval_only`, so an LLM cannot receive them through
automatic runtime-state visibility.

## AIME 2024/2025 and MBPP-Plus

First download and verify the pinned datasets:

```bash
python -m coscope.scripts.download_benchmarks
```

Run a small multi-agent validation:

```bash
python -m coscope.scripts.run_benchmark_suite \
  --benchmarks aime2024,aime2025,mbpp_plus \
  --limit 2 \
  --aime-max-output-tokens 8192 \
  --mbpp-max-output-tokens 4096 \
  --output new_results/aime_mbpp_plus_2.json
```

MBPP-Plus scoring requires a running Docker daemon. It uses official EvalPlus
base and augmented tests; generated code is never executed directly on the
host. The same benchmark names are accepted by `run_sharing_ablation` and
`run_reasoning_modes`. Use `--benchmark-limits
aime2024=30,aime2025=30,mbpp_plus=378` when running the three complete,
unequal-sized splits together.

The original 2048-token AIME calibration exhausted the completion budget
before the required marker. A later eight-benchmark run found further cap hits
at 4096 for AIME, 2048 for MBPP-Plus, and 1024 for MuSiQue. Live runners
therefore apply no output cap by default. Set a global or benchmark-specific
cap only when a fixed compute budget is part of the experiment. Record and
report cap hits; do not repair truncated outputs after scoring.

The repository-root `.env` also leaves `COSCOPE_LLM_MAX_TOKENS` unset. With no
CLI or environment override, the DeepSeek request omits `max_tokens` and uses
the model/provider's intrinsic output limit.

## Full-run preflight and unequal split sizes

Validate every local example without making model or embedding calls:

```bash
python -m coscope.scripts.preflight_experiment \
  --full \
  --workflow sharing_ablation \
  --output new_results/full_experiment_preflight.json
```

The report checks loaded counts, unique IDs, required fields, context-source
IDs, MBPP hidden-test boundaries, data inventory, expected metric families,
and planned LLM/task/EvalPlus work. A full eight-benchmark sharing ablation is
large; inspect this estimate before starting provider-backed execution.

Use per-benchmark limits when splits differ:

```bash
python -m coscope.scripts.run_sharing_ablation \
  --benchmarks aime2024,aime2025,mbpp_plus \
  --limit 5 \
  --benchmark-limits aime2024=30,aime2025=30,mbpp_plus=378
```

`--limit` remains the default for benchmarks without an override. Every live
runner records the resolved `limits_by_benchmark` in its JSON report.

## Canonical factorial experiment

The primary experiment is the complete Cartesian product:

```text
2 reasoning modes (CoT, ToT)
x 3 sharing policies (CoScope, full-sharing, no-sharing)
x 8 benchmarks
```

Every condition keeps the same planner → solver → verifier topology. In a CoT
condition all three agents use CoT. In a ToT condition all three use full ToT
with three branches and one value/vote call.

Run the no-provider full-data preflight first:

```bash
python -m coscope.scripts.preflight_experiment \
  --workflow factorial \
  --full \
  --reasoning-modes cot,tot \
  --sharing-policies coscope,full_sharing,no_sharing \
  --output new_results/full_factorial_preflight.json
```

The full live launch is deliberately guarded:

```bash
python -m coscope.scripts.run_factorial_experiment \
  --full \
  --confirm-full-run \
  --reasoning-modes cot,tot \
  --sharing-policies coscope,full_sharing,no_sharing \
  --bootstrap-samples 5000 \
  --output new_results/factorial_cot_tot_full.json
```

For each example, CoT uses `3 agents x 1 call x 3 policies = 9` LLM calls.
ToT uses `3 agents x 4 calls x 3 policies = 36`, for 45 total. ToT creates
nine branch retrieval requests per sharing condition and submits them in one
scope-safe batch. Shared first-stage recall may be reused, while every
agent/branch independently authorizes, reranks, falls back, and assembles its
context.

GoT occupies the same mode registry and can later be added with
`--reasoning-modes cot,tot,got`; it is intentionally absent from the default
matrix. The older suite, sharing-only, and single-reasoner mode runners remain
useful diagnostic experiments but are not the canonical factorial result.

## Durable cloud execution

Use the cloud runner for a long full experiment. It persists every
benchmark/example/mode/policy condition independently, so a restart does not
repeat successful provider calls:

```bash
python -m coscope.scripts.run_cloud_factorial \
  --full --confirm-full-run \
  --workers 4 \
  --purge-details-after-success \
  --env-file /etc/coscope/coscope.env \
  --data-root /var/lib/coscope/data \
  --state-dir /var/lib/coscope/runs/factorial-v1
```

SQLite WAL is the checkpoint source of truth. `status.json` is updated
atomically, while `events.jsonl` records task starts, successes, retries, and
failures without storing secrets. Only one coordinator can own a state
directory; `--workers` creates bounded parallel task workers inside it.
Successful rows are immutable during resume. Running rows are reclaimed only
after the previous coordinator lease becomes stale.
Because provider calls and SQLite cannot share a transaction, a hard kill may
repeat an in-flight condition whose response was not checkpointed; it cannot
repeat a condition already marked `succeeded`. At most `--workers` conditions
are exposed to this unavoidable provider-side ambiguity.

With `--purge-details-after-success`, the runner retains compact metric inputs
instead of full non-code task details. Once official MBPP scoring and final
aggregation finish, it atomically writes `final_metrics.json` plus
`COMPLETED.json`, then removes the checkpoint, WAL, event log, and detailed
results. The completion marker makes subsequent starts idempotent.

Inspect progress without stopping the run:

```bash
python -m coscope.scripts.experiment_status \
  /var/lib/coscope/runs/factorial-v1
```

See `deploy/ali-root/README.md` for GitHub, systemd, secret placement, dataset
transfer, clean shutdown, and result-export instructions.

## GSM8K threshold sweep

This runner evaluates a three-agent workflow (`planner`, `solver`, and
`verifier`) while changing only the public-query grouping threshold. All three
share the question scope but have different restricted role-knowledge scopes.
Raw working text remains private; only summaries flow downstream:

```bash
python -m coscope.scripts.run_mas_benchmark \
  --limit 5 \
  --seed 20260729 \
  --thresholds 0.75,0.82,0.90 \
  --output new_results/mas_gsm8k_threshold_sweep.json
```

`--limit 1319` evaluates the complete local test split but performs three LLM
calls per task and threshold. Confirm the provider budget before a large run.

The scoped one-example regression achieved 1/1 at every threshold. Thresholds
`0.75`, `0.82`, and `0.90` produced respectively 1, 2, and 3 public retrieval
groups, with savings of 66.7%, 33.3%, and 0%. Every point also performed three
private specialist lookups. This confirms threshold control and private
fallback coexist; one example cannot compare accuracy.

## Multi-benchmark MAS

```bash
python -m coscope.scripts.run_benchmark_suite \
  --limit 2 \
  --seed 20260729 \
  --threshold 0.82 \
  --output new_results/mas_benchmark_suite_2.json
```

The local validation covers HotpotQA, 2WikiMultiHopQA, MuSiQue, and MATH.
Official task evidence remains in a common scope so answer quality is not
confounded by synthetic evidence denial. Planner, solver, and verifier still
have distinct role-restricted knowledge scopes. Retrieval is batched first;
execution is sequential, and each pending request refreshes its view after
upstream summaries are published.
Answer scoring follows each benchmark: normalized EM/F1 for multi-hop QA,
maximum over MuSiQue aliases, and normalized symbolic equivalence for MATH.
The post-change one-example regression is intended to validate data flow, not
rank models. Supporting-fact and joint scores remain unreported because the
generator does not emit official supporting-fact identifiers.

## Sharing ablation

The paired ablation compares the same examples, agents, provider models, and
verifier-only final-answer rule under three information-flow policies:

- `coscope`: common evidence is reusable, role knowledge is restricted, raw
  thinking is private, and only explicit summaries flow downstream;
- `full_sharing`: role knowledge and raw upstream thinking are team-visible;
- `no_sharing`: each agent receives a separate evidence copy and neither
  retrieval nor runtime state is shared.

```bash
python -m coscope.scripts.run_sharing_ablation \
  --limit 5 \
  --seed 20260729 \
  --threshold 0.75 \
  --max-output-tokens 1024 \
  --musique-max-output-tokens 2048 \
  --math-max-output-tokens 2048 \
  --bootstrap-samples 5000 \
  --output new_results/sharing_ablation.json
```

MATH receives a larger output cap because a 768-token calibration truncated
valid reasoning before its required final-answer marker. The runner reports
that calibration as failure; it does not recover answers from another agent.
Use `merge_sharing_ablation` to combine disjoint benchmark shards without
rerunning provider calls. `--replace math=report.json` replaces a calibration
shard, while `--append report.json` adds disjoint examples and rejects
duplicate IDs. The command recomputes aggregates and paired bootstrap
intervals.

## CoT and retrieval-aware ToT comparison

```bash
python -m coscope.scripts.run_reasoning_modes \
  --limit 2 \
  --seed 20260729 \
  --output new_results/reasoning_mode_comparison_2.json
```

By default this covers all five local benchmarks; use, for example,
`--benchmarks gsm8k,math` to select a subset. CoT issues one retrieval request
and one generation. ToT creates three branch nodes, constructs one public
retrieval intent per branch, and sends all three requests through one
`retrieve_batch` call. Compatible requests share public first-stage recall;
each branch still authorizes, reranks, falls back, and assembles context
independently. ToT then performs three branch generations and one value/vote
selection.

This batching concerns retrieval computation, not the LLM API. Branch
generations remain separate calls. Reports include retrieval request count,
groups, store queries, query savings, group IDs, public queries, and
per-branch context sources. GoT runtime code remains for compatibility but is
not part of active experiments.

## Reported metrics

Experiments record:

- official answer accuracy or EM/F1 and task-success rate;
- AIME integer accuracy and MBPP/MBPP+ base/plus pass@1;
- LLM prompt, completion, reasoning, cached, and total tokens;
- embedding calls/input tokens and MAS overhead;
- retrieval groups, savings, Recall@K, MRR@K, and fallback calls;
- duplicate context, selected items, and unauthorized exposure;
- retrieval, generation, and end-to-end latency;
- reasoning nodes, retained thoughts, and LLM calls for mode comparisons.
- per-agent effective scopes, selected context sources, private working-memory
  IDs, published-summary IDs, and intended recipients.

Reasoning tokens are already included in completion tokens. Results under
ignored `new_results/` contain answers and share-safe summaries, never raw
private working text or credentials.

Official references:

- [GSM8K repository](https://github.com/openai/grade-school-math)
- [HotpotQA evaluation](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
- [2WikiMultiHopQA repository](https://github.com/Alab-NII/2wikimultihop)
- [MuSiQue repository](https://github.com/stonybrooknlp/musique)
- [MATH grading in PRM800K](https://github.com/openai/prm800k/tree/main/prm800k/grading)
- [Chain-of-Thought paper](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts reference implementation](https://github.com/princeton-nlp/tree-of-thought-llm)
- [Graph of Thoughts reference implementation](https://github.com/spcl/graph-of-thoughts)
