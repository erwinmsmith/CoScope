# Stage 7 Results Archive

Stage 7 separates current reproducible data/results from historical artifacts.

## Active Layout

Current project outputs should use:

```text
data/processed/   # reproducible experiment inputs
data/eval/        # current evaluation outputs
data/exports/     # current trace bundles and markdown exports
```

Historical artifacts should use:

```text
data/archive/cot_old/      # old CoT project artifacts
data/archive/senior_old/   # old senior GoT/ToT artifacts, if available
```

## Current Reproducible Inputs

The active processed tree currently contains only the promoted CoT shards:

```text
data/processed/cot/musique/dev/s1_linear.jsonl
data/processed/cot/musique/dev/s4_policy_isolated.jsonl
data/processed/cot/2wikimhqa/dev/s1_linear.jsonl
data/processed/cot/hotpotqa/dev/s1_linear.jsonl
data/processed/cot/gsm8k/test/s1_linear.jsonl
data/processed/cot/gsm8k/test/s2_linear.jsonl
```

These are current inputs because they follow the unified
`DatasetPipeline -> data/processed/{rpt}/{dataset}/{split}` layout.

## Current Evaluation And Export Roots

The following roots now exist for new runs:

```text
data/eval/
data/exports/
```

They are intentionally empty except for `.gitkeep`. New outputs from scripts
such as `scripts.run_cot_s4_eval`, `scripts.eval_jsonl`, and
`scripts.export_trace_bundle` should write here.

## Historical CoT Archive

Old CoT smoke, manifest-reuse, evaluation, and export artifacts remain under:

```text
data/archive/cot_old/
```

They are not active experiment inputs.

## Senior Archive

No files were found under:

```text
CoScope(1)/CoScope/data
```

Therefore Stage 7 created:

```text
data/archive/senior_old/
```

as a reserved archive location, but copied no senior historical results.

## Git Tracking Rule

Large real data files remain ignored by `.gitignore`. The project tracks only:

- data layout notes
- archive README files
- `.gitkeep` skeleton files

This preserves the folder contract without accidentally committing corpora,
processed shards, or large historical results.

