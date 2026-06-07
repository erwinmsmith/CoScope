# CoScope

**Collaborative Memory Retrieval Framework for Multi-Agent Systems**

CoScope is a framework for efficient collaborative memory retrieval in multi-agent systems (MAS). It enables multiple agents to share first-stage retrieval while preserving individual agent needs through personalized reranking and private fallback mechanisms.

## Key Features

- **Modular Architecture**: Clean separation of concerns with encoder, router, matrix, projection, retriever, reranker, fallback, and fusion modules
- **Collaborative Retrieval**: Share first-stage retrieval across agents with overlapping memory scopes
- **Query Matrix Mechanism**: Unified batch retrieval using query matrices with projection and SVD subspace extraction
- **Scope-Aware Routing**: Intelligent request bucketing based on scope overlap, memory type, and policy compatibility
- **Agent-Specific Reranking**: Personalized result ranking based on agent role and state
- **Private Fallback**: Protected recall through private scope retrieval
- **Prompt Management**: Separated prompt templates for easy customization
- **Memory CRUD**: High-level memory operations interface
- **LangChain/LangGraph Integration**: Seamlessly integrate with existing agent frameworks

## Architecture

```
CoScope/                 # repo root (no top-level package wrapper)
├── core/                # Types, interfaces, scope IDs, artifact slots
├── config/              # Configuration management (YAML + env)
│
├── engine/              # CoScope facade — entry point for all retrieval
│
├── retrieval/           # Retrieval pipeline (performance-tuning surface)
│   ├── encoder/         # Request normalisation
│   ├── router/          # Scope-based bucketing
│   ├── matrix/          # Query matrix construction
│   ├── projection/      # Shared subspace projection (SVD)
│   ├── retriever/       # Candidate retrieval
│   ├── reranker/        # Agent-specific reranking
│   ├── fallback/        # Private-scope fallback
│   └── fusion/          # Evidence fusion
│
├── memory/              # Memory storage and CRUD
│   ├── store.py         # In-memory backend (pluggable via protocol)
│   ├── *_builder.py     # Workspace / task-shared / private builders
│   └── crud/
│       ├── manager.py   # High-level CRUD operations
│       ├── types.py
│       └── auth/        # Access control, policy validator, S4 FMR check
│
├── llm/                 # LLM client layer (DB/service-swappable)
│   ├── base.py          # LLMClient protocol + LLMResponse (re-export)
│   ├── dashscope.py     # DashScopeClient (Qwen API)
│   └── template.py      # TemplateLLMClient (deterministic, for testing)
│
├── dataio/                  # Data-access and serialisation layer
│   ├── serializer.py    # Episode <-> JSONL round-trip
│   ├── stats_reporter.py
│   ├── validator.py
│   └── loaders/         # Raw-dataset loaders (MuSiQue, HotpotQA, ...)
│
├── prompts/             # Prompt management
│   ├── registry.py      # Central DB-pluggable registry (set_backend)
│   ├── manager.py       # Template manager (JSON/YAML file loading)
│   └── templates.py     # Role prompt templates
│
├── graph/               # GoT / CoT / ToT graph builders + prompt strings
├── rollout/             # Offline LLM rollout engine; artifact traces
├── construction/        # Episode construction from raw datasets
│                        #   - io/loaders reads raw data
│                        #   - construction/ builds Episodes from it
├── evaluation/
│   ├── metrics.py       # Recall, precision, FMR, rho
│   ├── runner.py        # In-process variant runner
│   ├── jsonl_runner.py  # Batch JSONL evaluation
│   └── split/           # Subset assignment (S1-S4) and split management
│
├── agents/              # Agent builders (planner / solver / verifier)
├── scripts/             # Thin CLI scripts — arg parsing + orchestration only
└── examples/            # Usage examples
```

> **DB integration hooks** — six stable extension points for wiring the
> external database product:
> `memory.store` (backend protocol) · `memory.crud.auth` (permission layer) ·
> `prompts.registry.set_backend` · `embedding.*` (adapter) ·
> `llm.LLMClient` (model service) · `retrieval.retriever` (vector DB)

See `ARCHITECTURE.md` for the full migration history and the ordered
next-action plan.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/CoScope.git
cd CoScope

# Install core dependencies
pip install -e .

# Install LangChain integration
pip install -e ".[langchain]"

# Install LangGraph integration
pip install -e ".[langgraph]"
```

## Quick Start

```python
from engine import CoScope
from core.types import MemoryType

# Initialize CoScope
coscope = CoScope()

# Register agents
coscope.create_agent(agent_id="planner_1", role="planner")
coscope.create_agent(agent_id="solver_1", role="solver")

# Add memories
coscope.add_memory(
    content="The constraint is to complete within 2 hours.",
    scope_id="task/shared",
    memory_type=MemoryType.SEMANTIC,
)

# Create and execute retrieval requests
requests = [
    coscope.create_request(agent_id="planner_1", query="What are the constraints?"),
    coscope.create_request(agent_id="solver_1", query="What evidence is available?"),
]

results = coscope.retrieve(requests)

for result in results:
    print(f"Agent: {result.agent_id}")
    for candidate in result.candidates[:5]:
        print(f"  - {candidate.memory.content}")
```

## Configuration

### config.yaml

Structural configuration for the retrieval pipeline:

```yaml
retrieval:
  query_embedding_dim: 1024
  variant: "a4"
  shared_top_k: 50
  rerank_top_k: 20

scope_hierarchy:
  task_shared:
    - id: "task/{task_id}/shared"
      scope_type: "task_shared"
      default_policy:
        visibility: ["team"]
```

### .env

Runtime configuration and secrets:

```bash
COSCOPE_OPENAI_API_KEY=your-api-key
COSCOPE_LOG_LEVEL=INFO
COSCOPE_MEMORY_BACKEND=inmemory
```

## Advanced Usage

### Using Memory CRUD

```python
from memory.crud import MemoryCRUD, MemoryQuery

crud = MemoryCRUD(memory_store, embedding_provider)

# Create
memory = crud.create(
    content="Important fact",
    scope_id="task/shared",
    memory_type=MemoryType.SEMANTIC,
)

# Query
results = crud.query(MemoryQuery(text="Important", limit=10))

# Stats
stats = crud.get_stats()
```

### Using Prompt Management

```python
from prompts import PromptManager, RoleTemplates

manager = PromptManager()

# Get template
template = manager.get_template("role/planner")

# Render
rendered = manager.render("role/planner", query="What constraints?", task_context="...")
```

### Custom Retrieval Components

```python
from retrieval.router import OverlapAwareRouter
from retrieval.reranker import RoleAwareReranker
from retrieval.fusion import ReciprocalRankFusion

# Use overlap-aware routing
router = OverlapAwareRouter(min_overlap_score=0.3)

# Use role-aware reranking
reranker = RoleAwareReranker()

# Use RRF fusion
fusion = ReciprocalRankFusion(k=60)
```

### Experiment Variants

The retrieval pipeline supports the no-training ablation modes described in the
design document:

```python
# A1: per-agent independent retrieval
results = coscope.retrieve(requests, variant="a1")

# A3: scope-only shared retrieval with mean query embedding
results = coscope.retrieve(requests, variant="a3")

# A4: shared mean retrieval + personalized rerank + private fallback
results = coscope.retrieve(requests, variant="a4")
```

You can also set the default in `config.yaml`:

```yaml
retrieval:
  variant: "a4"
```

### Minimal Evaluation

```python
from evaluation import evaluate_retrieval

results = coscope.retrieve(requests, variant="a4")
stats = coscope.get_stats()["pipeline_stats"]

gold_by_request = {
    requests[0].request_id: ["mem_gold_1"],
    requests[1].request_id: ["mem_gold_1", "mem_gold_2"],
}

report = evaluate_retrieval(
    results,
    gold_by_request,
    k=10,
    pipeline_stats=stats,
    conflict_request_ids=["req_verifier_s4"],
)

print(report.to_dict())
```

The report includes `recall_at_k`, `evidence_hit_rate`, `mrr_at_k`,
`fallback_rate`, `first_stage_savings`, `false_merge_rate`, and
`content_false_merge_rate`.

To compare all no-training variants in one pass:

```python
from evaluation import evaluate_variants, format_variant_table

runs = evaluate_variants(
    coscope,
    requests,
    gold_by_request,
    k=10,
    conflict_request_ids=["req_verifier_s4"],
)

print(format_variant_table(runs))
```

There is also a runnable toy example:

```bash
python -m examples.evaluate_variants
```

For a slightly broader synthetic suite covering S1/S2/S3/S4-style cases:

```bash
python -m examples.evaluate_synthetic
```

Programmatic use:

```python
from evaluation import evaluate_synthetic_suite, format_synthetic_table

summaries = evaluate_synthetic_suite(k=3)
print(format_synthetic_table(summaries))
```

## Agent Framework Integration

### LangChain

```python
from agents import CoScopeRetrievalTool

tool = coscope.get_retrieval_tool(agent_id="planner_1")
# Use as a LangChain tool
```

### LangGraph

```python
from agents import CoScopeGraphBuilder

builder = CoScopeGraphBuilder(pipeline, memory_manager)
builder.add_retrieve_node("retrieve")
builder.add_edge("retrieve", "end")

agent = builder.build()
```

## License

MIT License
