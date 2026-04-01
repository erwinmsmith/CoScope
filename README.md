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
└── examples/          # Usage examples
```

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
