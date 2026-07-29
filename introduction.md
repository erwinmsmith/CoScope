# CoScope: Scope-Aware Context and Retrieval Runtime

CoScope is not a dataset-specific retrieval script and does not model CoT,
ToT, or GoT as arrangements of business agents. It is the runtime layer that
decides what information each agent may see, when that information may enter
context, and which public retrieval computation may be safely reused.

Its goals are:

1. improve collaboration through complementary evidence;
2. reduce context pollution from irrelevant, duplicate, stale, conflicting,
   or unverified content;
3. enforce knowledge, runtime, ownership, lifecycle, and permission boundaries.

## Two independent structures

`AgentTopology` defines the agents, roles, communication edges, tools, and
permissions. `ReasoningMode` defines how one agent organizes reasoning state:

- CoT: a linear state chain;
- ToT: branch-private alternatives with scoring, pruning, and explicit merge;
- GoT: a dependency graph with multi-parent nodes and controlled propagation.

An application can therefore use a fixed Planner → Researcher → Solver
topology while assigning a different reasoning mode to each agent.

## Runtime principle

CoScope is scope-first and retrieval-second:

```text
resolve EffectiveView
→ split public and private intent
→ group compatible public requests
→ retrieve once from the public-view intersection
→ authorize and rerank independently per agent
→ retrieve extra private/local evidence when needed
→ assemble an independent ContextPacket
→ execute an LLM/tool
→ write an artifact with lifecycle and provenance
```

Sharing computation must never expand access. Private intent cannot enter the
shared vector, branch-local drafts cannot cross branches, and promotion into a
shared scope must be explicit and policy checked.

## Current implementation

The repository implements the offline and simulated runtime contracts:

- multidimensional scope descriptors and effective views;
- policy-first corpus filtering;
- scope signatures and safe public-query clustering;
- medoid shared retrieval and scope-bound cache;
- per-agent rerank and private fallback;
- context deduplication, conflict handling, pollution guards, and budgets;
- artifact lifecycle, verification, and promotion;
- independent CoT, ToT, and GoT state runtimes;
- replay, mock execution, tracing, and evaluation primitives.

The live LLM adapter remains provider-neutral. End-to-end task experiments
must generate queries and runtime memories dynamically; gold decomposition and
gold task-shared memories may be used only in explicitly labeled replay
benchmarks.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation details.
