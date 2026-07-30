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

Edit the repository-root `.env` to configure DeepSeek generation and the
local CPU embedding backend:

```dotenv
COSCOPE_RUNTIME_MODE=live
DEEPSEEK_API_KEY=sk-...
COSCOPE_LLM_MODEL=deepseek-v4-flash

COSCOPE_EMBEDDING_PROVIDER=fastembed
COSCOPE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
COSCOPE_EMBEDDING_DIMENSION=384
COSCOPE_EMBEDDING_CACHE_DIR=fastembed_cache
COSCOPE_EMBEDDING_THREADS=2
COSCOPE_EMBEDDING_RESULT_CACHE_SIZE=4096

COSCOPE_MEMORY_PROVIDER=qdrant
COSCOPE_QDRANT_URL=http://127.0.0.1:6333
COSCOPE_QDRANT_COLLECTION=coscope_memory
```

FastEmbed downloads a quantized ONNX model once, verifies its SHA-256 identity,
shares one CPU session across experiment workers, and uses a bounded result
cache for repeated text across factorial arms. Set
`COSCOPE_EMBEDDING_LOCAL_FILES_ONLY=true` after the cache is populated.
DashScope `text-embedding-v3` and Zhipu `embedding-3` remain optional remote
providers. To use Zhipu instead:

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
- **Live:** DeepSeek `deepseek-v4-flash` generation and local FastEmbed
  `BAAI/bge-small-en-v1.5` retrieval are configured through `.env`.

See [Architecture](docs/ARCHITECTURE.md) and [Migration](docs/MIGRATION.md).
For live token accounting, multi-benchmark scoring, threshold sweeps, and the
implemented CoT/retrieval-aware-ToT comparison, see
[Experiment Guide](docs/EXPERIMENTS.md).
Dataset provenance, pinned revisions, licenses, and MBPP-Plus sandboxing are
documented in [Benchmark Datasets](docs/DATASETS.md).

The canonical benchmark launch uses
`coscope.scripts.run_factorial_experiment`: all planner/solver/verifier agents
uniformly use CoT or ToT under each combination of CoScope/full-sharing/
no-sharing and batched/independent retrieval. Formal cloud runs require
Qdrant; all memory writes, reads, scope filters, and vector searches go through
the database.
Run `coscope.scripts.preflight_experiment --workflow factorial --full` before
any provider-backed full launch.

For a resumable parallel cloud launch, use
`coscope.scripts.run_cloud_factorial`. It checkpoints each condition in
SQLite and keeps provider secrets, benchmark data, and experiment state
outside Git. The Aliyun systemd deployment is documented in
[deploy/ali-root/README.md](deploy/ali-root/README.md).
