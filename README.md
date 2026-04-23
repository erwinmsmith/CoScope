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
- **GoT-MAS Dataset Pipeline**: Build Graph-of-Thought multi-agent datasets (GSM8K, MATH, HotpotQA, 2WikiMultiHopQA, MusiqueQA) with memory scope construction, episode building, and rho evaluation

## Architecture

```
coscope/
├── config/              # Configuration management (YAML + env)
├── core/                # Core types and interfaces
├── retrieval/           # Retrieval pipeline modules
│   ├── encoder/        # Request normalization
│   ├── router/        # Scope-based bucketing
│   ├── matrix/         # Query matrix construction
│   ├── projection/     # Shared subspace projection
│   ├── retriever/     # Candidate retrieval
│   ├── reranker/       # Agent-specific reranking
│   ├── fallback/       # Private scope fallback
│   └── fusion/         # Evidence fusion
├── memory/             # Memory storage and CRUD
│   └── crud/          # High-level memory operations
├── prompts/           # Prompt template management
├── agents/            # Agent integrations (LangChain/LangGraph)
├── examples/          # Usage examples
└── data/              # GoT-MAS dataset construction pipeline
    ├── agents/         # Planner, Solver, Verifier agent builders
    ├── auth/           # Access control, policy validation, S4 checker
    ├── core/           # Core types (GoTGraph, GoTNode, etc.)
    ├── cot/            # Chain-of-Thought node types
    ├── got/            # Graph-of-Thought graph builder, templates, rho calculator
    ├── loaders/        # Dataset loaders (GSM8K, MATH, HotpotQA, Musique, Wiki)
    ├── memory/         # Memory scope builders (workspace, task-shared, private, restricted)
    ├── output/         # Serialization, stats reporter, validator
    ├── pipeline/       # Dataset pipeline and episode builder
    ├── scripts/        # Build, compute-rho, smoke-test utilities
    └── split/          # Train/dev/test split manager
```

## Installation

```bash
# Clone the repository
git clone https://github.com/erwinmsmith/CoScope.git
cd CoScope

# Install core dependencies
pip install -e .

# Install LangChain integration
pip install -e ".[langchain]"

# Install LangGraph integration
pip install -e ".[langgraph]"
```

## Quick Start

### Memory Retrieval

```python
from coscope import CoScope
from coscope.core.types import MemoryType

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

### Dataset Construction (GoT-MAS Pipeline)

```python
from coscope.data.pipeline import EpisodeBuilder
from coscope.data.loaders import GSM8KLoader

# Load dataset
loader = GSM8KLoader()
raw_items = loader.load(split="train", limit=100)

# Build episodes with GoT graph
builder = EpisodeBuilder()
episodes = builder.build_all(raw_items, dataset="gsm8k")

for ep in episodes:
    print(f"Episode {ep.episode_id}: {len(ep.nodes)} nodes, {len(ep.memories)} memories")
```

## Configuration

### config.yaml

Structural configuration for the retrieval pipeline:

```yaml
retrieval:
  query_embedding_dim: 1024
  projection_dim: 256
  svd_rank: 64
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
from coscope.memory.crud import MemoryCRUD, MemoryQuery

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
from coscope.prompts import PromptManager, RoleTemplates

manager = PromptManager()

# Get template
template = manager.get_template("role/planner")

# Render
rendered = manager.render("role/planner", query="What constraints?", task_context="...")
```

### Custom Retrieval Components

```python
from coscope.retrieval.router import OverlapAwareRouter
from coscope.retrieval.reranker import RoleAwareReranker
from coscope.retrieval.fusion import ReciprocalRankFusion

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

# A5: query matrix + unsupervised truncated SVD + rerank + fallback
results = coscope.retrieve(requests, variant="a5")
```

You can also set the default in `config.yaml`:

```yaml
retrieval:
  variant: "a5"
```

### Minimal Evaluation

```python
from coscope.evaluation import evaluate_retrieval

results = coscope.retrieve(requests, variant="a5")
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

The report includes `recall_at_k`, `mrr_at_k`,
`first_stage_savings`, and `false_merge_rate`.

To compare all no-training variants in one pass:

```python
from coscope.evaluation import evaluate_variants, format_variant_table

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
python -m coscope.examples.evaluate_variants
```

For a slightly broader synthetic suite covering S1/S2/S3/S4-style cases:

```bash
python -m coscope.examples.evaluate_synthetic
```

Programmatic use:

```python
from coscope.evaluation import evaluate_synthetic_suite, format_synthetic_table

summaries = evaluate_synthetic_suite(k=3)
print(format_synthetic_table(summaries))
```

## Agent Framework Integration

### LangChain

```python
from coscope.agents import CoScopeRetrievalTool

tool = coscope.get_retrieval_tool(agent_id="planner_1")
# Use as a LangChain tool
```

### LangGraph

```python
from coscope.agents import CoScopeGraphBuilder

builder = CoScopeGraphBuilder(pipeline, memory_manager)
builder.add_retrieve_node("retrieve")
builder.add_edge("retrieve", "end")

agent = builder.build()
```

## License

MIT License
