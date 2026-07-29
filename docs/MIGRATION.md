# Migration from the Legacy Experiment Workspace

## Active source

The repository root is now the package root:

```text
pyproject.toml
coscope/
tests/
docs/
```

This preserves the public repository's `coscope` package skeleton while
replacing its original retrieval-only model with the scope-aware runtime.

## Replaced concepts

| Legacy concept | Runtime replacement |
|---|---|
| CoT/ToT/GoT as agent graph types | `AgentTopology` plus independent `ReasoningMode` |
| String scope hierarchy | multi-dimensional `ScopeDescriptor` |
| Retrieve globally, filter later | `EffectiveView` before search |
| Full query on shared path | explicit public/private intent split |
| Mean-query buckets | scope signature + medoid complete-link grouping |
| Gold task-shared memories | runtime artifacts with lifecycle and promotion |
| One final candidate list | per-agent rerank, fallback, and `ContextPacket` |
| Retrieval-count-only evaluation | retrieval, context, safety, runtime, task metrics |

## Local-only assets

The following directories are retained for reference but ignored:

- `CoScope_unified/`
- `raw/`
- `eval/`
- `logs/`
- `new_results/`

No files were deleted. Historical results remain inspectable but are not valid
evidence for the redesigned runtime unless replayed through an explicit
compatibility adapter with their oracle assumptions disclosed.

## Compatibility policy

Compatibility should be implemented at the boundary, under `coscope/replay/`.
Core runtime types must not import legacy Episode, GraphType, dataset builder,
or oracle ground-truth types.
