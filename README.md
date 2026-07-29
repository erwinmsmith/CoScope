# CoScope

CoScope is a **scope-aware context and retrieval runtime for multi-agent LLM
systems**. It decides what information an agent may see, when that information
may enter context, and which public retrieval work can be safely reused.

This repository follows the package skeleton of
[`erwinmsmith/CoScope`](https://github.com/erwinmsmith/CoScope), but the runtime
model has been redesigned around the current specification.

## What changed

`AgentTopology` and `ReasoningMode` are separate concepts:

- agent topology describes which agents exist and how they communicate;
- CoT and ToT describe the modes used by active experiments. GoT remains only
  as a compatibility runtime.

`AgentClass` separately defines the durable knowledge and context contract for
a role. It controls knowledge domains, readable and publishable artifacts,
private/public thinking, and channel budgets. `AgentInstance` applies that
class within one run.

Retrieval is scope-first. Every request resolves an `EffectiveView` before
vector search. Only an explicitly declared `public_intent` is embedded on the
shared path. Requests may share a first-stage search only when their scope
signature is compatible and their public vectors satisfy medoid and pairwise
similarity thresholds. Final ranking, private fallback, and context assembly
remain agent-specific.

## Runtime flow

```text
Agent + ReasoningNode
        ↓
EffectiveView (hard policy boundary)
        ↓
Public/private intent split
        ↓
Scope signature + safe query grouping
        ↓
Shared public retrieval
        ↓
Per-agent authorization + rerank + private fallback
        ↓
Deduplicated, conflict-aware ContextPacket
        ↓
Mock or live executor
        ↓
Private unembedded working state + explicit share-safe summary
        ↓
Draft/proposed Artifact → verify → explicit promotion
```

## Package layout

```text
coscope/
├── core/        # agents, topology, artifacts, memories, events
├── runtime/     # kernel, run state, executor contracts
├── reasoning/   # independent CoT/ToT runtimes; compatibility GoT support
├── scope/       # descriptors, policies, effective views, signatures
├── retrieval/   # intent planning, grouping, retrieval, rerank, fallback
├── context/     # selection, pollution controls, budgets, packets
├── memory/      # store, lifecycle, commit, promotion
├── adapters/    # embedding and LLM provider contracts
├── replay/      # retrieval replay and mock execution
├── evaluation/  # retrieval, context, safety, runtime, task metrics
└── tracing/     # redaction-safe runtime traces
```

The previous `CoScope_unified/` experiment workspace and its raw data, logs,
and result files remain local-only and are excluded from version control. They
are not imported by the new runtime.

## Quick start

```bash
python -m pip install -e ".[dev]"
cp .env.example .env
python -m coscope.scripts.download_benchmarks
python -m coscope.scripts.smoke_test
pytest
```

## Live model configuration

Edit the repository-root `.env` to configure both providers:

```dotenv
COSCOPE_RUNTIME_MODE=live
DEEPSEEK_API_KEY=sk-...
COSCOPE_LLM_MODEL=deepseek-v4-flash

DASHSCOPE_API_KEY=sk-...
COSCOPE_EMBEDDING_MODEL=text-embedding-v3
COSCOPE_EMBEDDING_DIMENSION=1024
```

The default DashScope endpoint targets China (Beijing). Set
`COSCOPE_EMBEDDING_BASE_URL` to the endpoint matching the API key's region.
Both DashScope `text-embedding-v3` and the attached Zhipu `embedding-3` are
remote embedding APIs, not locally downloadable models. To use Zhipu instead:

```dotenv
ZHIPU_API_KEY=...
COSCOPE_EMBEDDING_PROVIDER=zhipu
COSCOPE_EMBEDDING_MODEL=embedding-3
COSCOPE_EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
COSCOPE_EMBEDDING_DIMENSION=1024
```

Create a provider-backed runtime and executor with:

```python
from coscope import CoScopeRuntime

runtime = CoScopeRuntime.from_env()
executor = runtime.live_executor()
```

After adding both keys, make one minimal call to each provider:

```bash
python -m coscope.scripts.check_providers
```

```python
from coscope import AgentInstance, CoScopeRuntime, MemoryEntry
from coscope.reasoning import ReasoningConfig, ReasoningMode
from coscope.scope import ScopeDescriptor, Visibility

runtime = CoScopeRuntime()
runtime.create_run("demo", run_id="run_demo")
runtime.register_agent(
    AgentInstance(
        "researcher",
        "researcher",
        knowledge_permissions=frozenset({"project_docs"}),
    )
)
_, node = runtime.start_reasoning(
    "researcher", ReasoningConfig(ReasoningMode.COT)
)

public_scope = ScopeDescriptor(
    frozenset({"project_docs"}),
    "run_demo/task",
    Visibility.TEAM_SHARED,
)
runtime.ingest_memory(
    MemoryEntry("The release deadline is Friday.", public_scope, "document", "doc_1")
)

request = runtime.create_request(
    run_id="run_demo",
    agent_id="researcher",
    reasoning_node_id=node.node_id,
    full_query="When is the release deadline?",
    public_intent="release deadline",
)
result = runtime.retrieve_batch([request])[request.request_id]
packet = runtime.assemble_context(request, result)
```

## Current execution modes

- **Replay:** runs prebuilt requests without invoking an LLM.
- **Simulated:** executes runtime control flow through `MockExecutor`.
- **Live:** DeepSeek `deepseek-v4-flash` generation and DashScope
  `text-embedding-v3` retrieval are configured through `.env`.

See [Architecture](docs/ARCHITECTURE.md) and [Migration](docs/MIGRATION.md).
For live token accounting, multi-benchmark scoring, threshold sweeps, and the
implemented CoT/retrieval-aware-ToT comparison, see
[Experiment Guide](docs/EXPERIMENTS.md).
Dataset provenance, pinned revisions, licenses, and MBPP-Plus sandboxing are
documented in [Benchmark Datasets](docs/DATASETS.md).

The canonical benchmark launch uses
`coscope.scripts.run_factorial_experiment`: all planner/solver/verifier agents
uniformly use CoT or ToT under each of CoScope, full-sharing, and no-sharing.
Run `coscope.scripts.preflight_experiment --workflow factorial --full` before
any provider-backed full launch.

For a resumable parallel cloud launch, use
`coscope.scripts.run_cloud_factorial`. It checkpoints each condition in
SQLite and keeps provider secrets, benchmark data, and experiment state
outside Git. The Aliyun systemd deployment is documented in
[deploy/ali-root/README.md](deploy/ali-root/README.md).
