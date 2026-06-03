# Stage 5 Data Index

This file records the Stage 5 data merge from the old CoT project into the
unified CoScope workspace.

## Canonical Rule

The paper-aligned project flow is:

```text
dataio/loaders -> DatasetPipeline -> data/processed/{rpt}/{dataset}/{split}
scripts.eval_jsonl -> retrieval-only evaluation reports
```

Therefore the unified tree keeps one canonical processed layout:

```text
data/processed/cot/{dataset}/{split}/{subset}_{graph_type}.jsonl
```

Old experiment directory names such as `processed_qwen_full` and
`processed_qwen_gsm8k_cot` were not preserved as active paths. Their current
CoT shards were promoted into the canonical layout instead.

## Promoted Raw Inputs

Copied into `data/raw`:

```text
data/raw/musique/musique_ans_v1.0_dev.jsonl
data/raw/musique/musique_ans_v1.0_train.jsonl
data/raw/2wikimhqa/dev.json
data/raw/2wikimhqa/test.json
data/raw/2wikimhqa/train.json
data/raw/hotpotqa/distractor_train_00000.parquet
data/raw/hotpotqa/distractor_train_00001.parquet
data/raw/hotpotqa/distractor_validation.parquet
data/raw/gsm8k/test.parquet
data/raw/gsm8k/train.parquet
data/raw/math/test.parquet
data/raw/math/train.parquet
```

Skipped:

```text
data/raw/**/.cache
```

Reason: cache files are local HuggingFace artifacts, not paper inputs.

## Promoted Processed CoT Shards

Copied into canonical processed paths:

```text
data/processed/cot/musique/dev/s1_linear.jsonl
data/processed/cot/musique/dev/s4_policy_isolated.jsonl
data/processed/cot/2wikimhqa/dev/s1_linear.jsonl
data/processed/cot/hotpotqa/dev/s1_linear.jsonl
data/processed/cot/gsm8k/test/s1_linear.jsonl
data/processed/cot/gsm8k/test/s2_linear.jsonl
```

These are the current Qwen-built CoT shards from the older project. They are
the only old processed CoT files promoted to active paths.

## Promoted Manifests

Copied into canonical manifest paths:

```text
data/manifests/musique_dev_ids.txt
```

## Archived Historical Artifacts

Historical smoke-test, manifest-reuse, evaluation, and export artifacts were
kept under:

```text
data/archive/cot_old/
```

Archived processed test outputs:

```text
data/archive/cot_old/processed/processed_manifest_reuse
data/archive/cot_old/processed/processed_manifest_seed
data/archive/cot_old/processed/processed_qwen_smoke
data/archive/cot_old/processed/processed_test
data/archive/cot_old/processed/processed_test2
data/archive/cot_old/processed/processed_test3
data/archive/cot_old/processed/processed_test4
data/archive/cot_old/processed/processed_test5
```

Archived historical results:

```text
data/archive/cot_old/eval
data/archive/cot_old/exports
```

Reason: these are useful provenance, but they are not the current reproducible
paper path. Keeping them out of `data/processed` prevents accidental use in new
experiments.

## Not Migrated

The following old `CoScope/coscope/data` directories were intentionally not
copied:

```text
agents
auth
config
core
cot
got
loaders
memory
output
pipeline
scripts
split
tot
__pycache__
```

Reason: these are stale code/cache trees that had drifted into `data`. The
unified project already has canonical source modules outside `data`.

## Stage 5 Decision

For active data, keep the logic closest to the paper framework:

- one raw input root: `data/raw`
- one processed root: `data/processed/{rpt}/{dataset}/{split}`
- one manifest root: `data/manifests`
- old outputs only as archive provenance: `data/archive/cot_old`

This avoids keeping two active copies of the same CoT shard under different
directory names.
