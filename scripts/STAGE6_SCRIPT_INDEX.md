# Stage 6 Script Index

This file records the Stage 6 script merge.

## Canonical Script Rule

The unified project keeps one active script path:

```text
CoScope_unified/scripts/
```

Scripts should run from the unified project root with:

```text
python -m scripts.<name>
```

Shell wrappers resolve the project root relative to the script location:

```text
cd "$(dirname "$0")/.."
```

This avoids hard-coding either the old local package name or the senior
collaborator's server path.

## Active Paper-Aligned Entrypoints

Build:

```text
scripts/build_all.py
scripts/rebuild_s4_only.py
main.py
```

Evaluation:

```text
scripts/eval_jsonl.py
scripts/eval_k5_k20_sweep.sh
scripts/run_stage_a_rpt.sh
scripts/run_stage_b_dataset.sh
scripts/run_cot_s4_eval.py
```

Trace and reporting:

```text
scripts/render_trace.py
scripts/export_trace_bundle.py
scripts/render_eval_brief.py
scripts/run_trace_job.py
```

Data utilities:

```text
scripts/audit_jsonl.py
scripts/check_prompt_isolation.py
scripts/compute_rho.py
scripts/generate_query_intent.py
scripts/generate_restricted.py
scripts/smoke_test.py
scripts/split_dev.py
```

## Merge Decisions

### Kept Unified `build_all.py`

The unified version is closer to the paper framework because it supports
`reasoning_path_type in {got,cot,tot}` while avoiding duplicate CoT/ToT episode
generation. CoT uses `LINEAR`; ToT uses `FORK`; all three reasoning paths can
append `POLICY_ISOLATED` for S4.

### Kept Unified `eval_jsonl.py`

The old project's `evaluate_episode_variants.py` was not restored as an active
script. The current paper path is:

```text
scripts.eval_jsonl -> evaluation/jsonl_runner.py
```

`scripts/run_cot_s4_eval.py` now delegates to this path, so the project does
not keep two evaluation drivers.

### Merged Old CoT Trace Utilities

The following old CoT scripts were imported earlier and kept active:

```text
scripts/render_trace.py
scripts/export_trace_bundle.py
scripts/render_eval_brief.py
scripts/run_trace_job.py
scripts/run_cot_s4_eval.py
```

They were adapted to unified imports and canonical paths.

## Stage 6 Fixes

Updated script defaults and examples:

```text
scripts/audit_jsonl.py
scripts/check_prompt_isolation.py
scripts/compute_rho.py
scripts/run_trace_job.py
```

Updated shell wrappers to use the current project directory instead of the old
server path:

```text
scripts/coscope_status.sh
scripts/run_2wiki_qwen.sh
scripts/run_a8_eval_only.sh
scripts/run_build_2wiki.sh
scripts/run_hotpotqa_qwen.sh
scripts/run_k20_only.sh
scripts/run_stage_a_qwen.sh
scripts/run_stage_a_qwen_resume.sh
scripts/run_stage_b.sh
scripts/run_stage_b_dataset.sh
```

## Not Kept As Active Scripts

```text
CoScope/coscope/scripts/evaluate_episode_variants.py
```

Reason: superseded by `scripts/eval_jsonl.py`, which is the current
retrieval-only evaluation path used by the paper framework.

## Verification

Stage 6 verification checks:

```text
No old `coscope.*` imports under scripts/
No old `coscope/data` active script paths
No old `/home/ninghanwen/lixin/CoScope` shell working directory
AST parse check for Python scripts
Help/import smoke checks for representative script entrypoints
```
