# CoScope Unified Migration Notes

## Current Status

Stage 0 through Stage 7 are complete.

This directory was created as the unified project workspace for merging:

- `CoScope(1)/CoScope`: senior collaborator's GoT/ToT-rich implementation
- `CoScope/coscope`: current owner's CoT-focused implementation and experiment artifacts

The original source directories are intentionally left unchanged.

## Stage 0: Baseline Copy

Baseline source:

```text
CoScope(1)/CoScope
```

Unified working directory:

```text
CoScope_unified
```

Initial action:

```text
Copied CoScope(1)/CoScope -> CoScope_unified
```

Rationale:

- `CoScope(1)/CoScope` currently has the richer active structure.
- It includes GoT/ToT support, `dataio`, `llm`, `memory/builders`, `evaluation/jsonl_runner.py`, stage scripts, project metadata, and documentation.
- It is the safer base for integrating the older CoT-specific work.

## Stage 1: Migration Ledger

This file tracks what is imported, skipped, or manually merged during unification.

### Base Project

Imported as the initial baseline:

```text
CoScope(1)/CoScope
```

### Pending Source Project

Not yet merged:

```text
CoScope/coscope
```

Important pending areas from `CoScope/coscope`:

- CoT-specific implementation details
- CoT experiment outputs and trace exports
- Legacy rollout clients and rho utilities
- Trace rendering and export scripts
- Older docs describing system-layer rollout design

## Planned Merge Areas

### Code

Review and selectively merge:

```text
graph/cot/
construction/
retrieval/
rollout/
scripts/
docs/
```

### Path Migrations

Old paths from `CoScope/coscope` should not be copied blindly. They map to newer locations:

```text
utils/loaders      -> dataio/loaders
utils/output       -> dataio
utils/split        -> evaluation/split
auth               -> memory/crud/auth
memory/*_builder.py -> memory/builders/
```

### Data And Results

Historical CoT data should be migrated later into an archive-style layout, for example:

```text
data/archive/cot_old/
```

Current reproducible inputs and outputs should remain under:

```text
data/processed/
data/eval/
data/exports/
data/manifests/
```

## Merge Rules

- Do not overwrite newer baseline files without inspecting differences.
- Prefer `CoScope_unified` as the only directory edited during migration.
- Preserve `CoScope/coscope` and `CoScope(1)/CoScope` as source snapshots.
- Keep one canonical implementation for retrieval, memory, data loading, and evaluation.
- Record every manual merge in this file.

## Stage 2: Module-Level Diff Plan

Completed as a read-only comparison between:

```text
CoScope/coscope
CoScope_unified
```

Key findings:

- `CoScope_unified` already has the richer CoT chain builder.
- `CoScope_unified` already moved CoT prompt strings to `prompts/graph/cot.py`.
- The old CoT `graph/cot/prompt_templates.py` path is useful for legacy script compatibility.
- The old `RestrictedBuilder` had a useful raw-item fallback for S4 smoke tests.
- The old `EpisodeBuilder` preserved useful metadata: `qa_type`, `task_shared_item_count`, and `expects_team_shared_evidence`.
- The old `PolicyValidator` used `expects_team_shared_evidence` to suppress inappropriate team-shared warnings for episodes that do not expect task-shared evidence.

Files reviewed for Stage 2:

```text
graph/cot/chain_builder.py
graph/cot/rho_calculator.py
graph/cot/prompt_templates.py
prompts/graph/cot.py
construction/episode_builder.py
construction/dataset_pipeline.py
memory/builders/restricted_builder.py
memory/crud/auth/policy_validator.py
```

Decision:

- Keep `CoScope_unified` as canonical for CoT graph construction and prompt layout.
- Patch in old CoT edge-case behavior where it is still valuable.
- Do not copy old `utils`, `auth`, or root-level `memory/*_builder.py` paths into the unified tree.

## Stage 3: CoT Merge

Merged from `CoScope/coscope` into `CoScope_unified`:

```text
memory/builders/restricted_builder.py
construction/episode_builder.py
memory/crud/auth/policy_validator.py
graph/cot/prompt_templates.py
```

Details:

- Added raw-item fallback restricted generation for S4 / `POLICY_ISOLATED` episodes when no offline restricted JSONL exists.
- Preserved `CoScope_unified` deterministic restricted `memory_id` generation and class-level restricted index cache.
- Restored episode metadata fields:
  - `qa_type`
  - `task_shared_item_count`
  - `expects_team_shared_evidence`
- Restored policy-validator behavior that only warns about missing `task_shared_episodic` evidence when the episode actually expects team-shared evidence.
- Added `graph/cot/prompt_templates.py` as a compatibility re-export of canonical prompts from `prompts/graph/cot.py`.

Verification:

```text
AST parse check: passed
scripts/smoke_test.py with PYTHONPATH=.: passed
CoT LINEAR synthetic build: passed
CoT POLICY_ISOLATED synthetic build: passed
```

Note:

```text
python -m py_compile ... was blocked by an existing __pycache__ write-permission issue.
The follow-up AST parse check was run with python -B and passed.
```

Files intentionally not overwritten:

```text
graph/cot/chain_builder.py
graph/cot/rho_calculator.py
prompts/graph/cot.py
construction/dataset_pipeline.py
```

Rationale:

- The unified versions are newer and already support the GoT/CoT/ToT shared structure.
- The useful old CoT behavior was small and could be merged without replacing full files.

## Stage 4: Scripts And Rollout Utilities

Completed migration of low-risk trace/export utilities and legacy rollout import
compatibility shims.

### Migrated Scripts

Copied from `CoScope/coscope/scripts` and updated for the unified package layout:

```text
scripts/render_trace.py
scripts/export_trace_bundle.py
scripts/render_eval_brief.py
scripts/run_trace_job.py
scripts/run_cot_s4_eval.py
```

Changes:

- Replaced old `coscope.*` imports with unified imports such as `main`,
  `scripts.export_trace_bundle`, and `scripts.eval_jsonl`.
- Updated default paths from `coscope/data/...` to `data/...`.
- Adapted `run_trace_job.py` to the current `main.py` CLI by removing old
  unsupported options such as id-manifest/resume/audit flags.
- Adapted `run_cot_s4_eval.py` to use the current `scripts.eval_jsonl` report
  pipeline instead of the legacy `evaluate_episode_variants.py`.
- Extended `render_eval_brief.py` so it can render both legacy summaries and
  current `eval_jsonl` reports with `stratified.cells`.

### Rollout Compatibility Shims

Added compatibility re-exports for old rollout import paths:

```text
rollout/template_llm_client.py -> llm.template.TemplateLLMClient
rollout/dashscope_client.py    -> llm.dashscope.DashScopeClient
rollout/llm_client.py          -> core.interfaces.LLMClient / LLMResponse
rollout/rho_v3.py              -> rollout.rho_calculator.compute_rho
```

Rationale:

- `CoScope_unified` already has canonical `llm/` and `rollout/rho_calculator.py`
  modules.
- Keeping small shims preserves old script compatibility without duplicating
  client implementations.

### Not Migrated

```text
scripts/evaluate_episode_variants.py
```

Reason:

- The unified project already has `scripts/eval_jsonl.py` and
  `evaluation/jsonl_runner.py`, which are the newer Mode-B evaluation path.
- `run_cot_s4_eval.py` now delegates to `scripts.eval_jsonl`.

### Verification

```text
AST parse check: passed
Import check for migrated scripts and rollout shims: passed
scripts/render_trace.py on synthetic CoT JSONL: passed
scripts/export_trace_bundle.py on synthetic CoT JSONL: passed
scripts/render_eval_brief.py on synthetic eval_jsonl summary: passed
scripts/run_trace_job.py --help: passed
scripts/run_cot_s4_eval.py --help: passed
scripts/run_cot_s4_eval.py --skip-build on synthetic CoT S4 shard: passed
```

Temporary verification outputs:

```text
tests/tmp/stage4_trace/
tests/tmp/stage4_s4/
```

## Stage 5: Data And Historical Result Merge

Completed data merge from `CoScope/coscope/data` into the unified tree.

Detailed index:

```text
data/STAGE5_DATA_INDEX.md
```

### Canonical Active Data

Promoted current CoT shards into the paper-aligned layout:

```text
data/processed/cot/{dataset}/{split}/{subset}_{graph_type}.jsonl
```

Promoted raw dataset files into:

```text
data/raw/
```

Promoted manifests into:

```text
data/manifests/
```

### Archived Historical Artifacts

Moved old smoke-test, manifest-reuse, evaluation, and export artifacts under:

```text
data/archive/cot_old/
```

### Not Migrated

Skipped stale code/cache directories that existed under the old `data` folder,
including:

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

Rationale:

- The unified source tree already has canonical code modules outside `data`.
- Active processed data should follow the current `DatasetPipeline` and
  `scripts.eval_jsonl` paper path.
- Old `processed_qwen_*` directory names were not kept as active paths; their
  current CoT shards were promoted into `data/processed/cot/...`.
- Historical outputs remain available as provenance without creating duplicate
  active files.

### Verification

```text
Canonical processed CoT line counts: passed
No .pyc files under unified data: passed
scripts/render_trace.py on canonical CoT shard: passed
```

Temporary verification output:

```text
tests/tmp/stage5_trace.md
```

## Stage 6: Script Merge

Completed script merge and path normalization.

Detailed index:

```text
scripts/STAGE6_SCRIPT_INDEX.md
```

### Active Script Directory

The unified project keeps one active script directory:

```text
scripts/
```

Entrypoints should be run as:

```text
python -m scripts.<name>
```

### Merge Decisions

Kept the unified script versions where they are closer to the paper framework,
especially:

```text
scripts/build_all.py
scripts/eval_jsonl.py
scripts/run_stage_a_rpt.sh
scripts/run_stage_b_dataset.sh
```

Rationale:

- `build_all.py` supports the shared GoT/CoT/ToT construction path.
- CoT and ToT graph-type overrides avoid duplicate episode generation.
- `eval_jsonl.py` is the current retrieval-only evaluation driver used by the
  paper path.

Merged old CoT trace/report scripts as active utilities:

```text
scripts/render_trace.py
scripts/export_trace_bundle.py
scripts/render_eval_brief.py
scripts/run_trace_job.py
scripts/run_cot_s4_eval.py
```

### Not Kept As Active

```text
CoScope/coscope/scripts/evaluate_episode_variants.py
```

Reason:

- It is superseded by `scripts/eval_jsonl.py` and
  `evaluation/jsonl_runner.py`.
- Keeping both would create two active evaluation logics.

### Stage 6 Fixes

- Fixed `scripts/audit_jsonl.py` to read JSONL with explicit UTF-8 encoding.
- Updated `scripts/check_prompt_isolation.py` default scan root from old
  `coscope` to the current project.
- Updated `scripts/compute_rho.py` to follow
  `data/processed/{rpt}/{dataset}/{split}`.
- Replaced hard-coded server `cd /home/ninghanwen/lixin/CoScope` lines in shell
  wrappers with script-relative project-root resolution.
- Confirmed active scripts have no old `coscope.*` imports or old
  `coscope/data` paths.

### Verification

```text
Script stale-path scan: passed
Python AST parse check under scripts/: passed
Representative --help checks: passed
scripts/smoke_test.py: passed
scripts/render_trace.py on canonical CoT shard: passed
```

## Stage 7: Data And Results Archive Layout

Completed archive layout normalization for data and results.

Detailed index:

```text
data/STAGE7_RESULTS_ARCHIVE.md
```

### Active Layout

Current reproducible experiment data/results should use:

```text
data/processed/
data/eval/
data/exports/
```

Historical artifacts should use:

```text
data/archive/cot_old/
data/archive/senior_old/
```

### Decisions

- Kept promoted CoT shards under `data/processed/cot/...`.
- Kept old CoT smoke/eval/export artifacts under `data/archive/cot_old/...`.
- Created `data/eval/` and `data/exports/` as current output roots.
- Created `data/archive/senior_old/` as the reserved location for senior
  historical GoT/ToT results.
- Copied no senior historical results because `CoScope(1)/CoScope/data`
  contains no local files in this workspace.

### Git Ignore Update

Updated `.gitignore` so large real data remains ignored while the project can
track:

```text
data/*.md
data/eval/.gitkeep
data/exports/.gitkeep
data/archive/.gitkeep
data/archive/cot_old/README.md
data/archive/senior_old/README.md
```

### Verification

```text
Senior data inventory: empty
Active processed CoT inventory: passed
Archive inventory: passed
No .pyc files under data/: passed
Current output roots data/eval and data/exports exist: passed
```

## Next Stage

Stage 8 can focus on documentation cleanup and path normalization:

1. Update remaining docs that still mention old `coscope/data/...` paths.
2. Normalize examples to `data/processed/{rpt}/{dataset}/{split}`.
3. Decide whether historical result tables should point to archive paths or be
   regenerated from current evaluation outputs.
