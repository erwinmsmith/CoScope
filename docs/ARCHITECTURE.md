# Architecture

## System boundary

CoScope does not define an application's business agents. It is the runtime
between an agent orchestrator and LLM, tool, vector-store, and knowledge-base
adapters. It manages reasoning state, effective visibility, safe retrieval
reuse, context construction, runtime memory, and audit events.

## Agent classes and dynamic context

`AgentClass` is the reusable permission and context blueprint.
`AgentInstance` carries only run-specific identity and state. A class declares:

- knowledge domains and tool capabilities;
- readable and publishable artifact types;
- whether private or public thinking is allowed;
- default total and per-channel context budgets.

The benchmark defines separate planner, solver, and verifier classes. They
share `benchmark_common` but respectively receive restricted
`planning_knowledge`, `solving_knowledge`, and `verification_knowledge`.

Runtime working text is written as unembedded `DRAFT` memory in a node- or
branch-private scope. It is visible to its owner but cannot enter vector
retrieval. Only an explicitly extracted reasoning summary can be committed to
a team or recipient-restricted scope. Runtime summaries are injected directly
and remain unembedded until a later indexing/promotion policy opts them in.
Pending downstream requests refresh their `EffectiveView` before context
assembly, so newly published state enters only the authorized downstream
context.

## Independent structures

`AgentTopology` answers who executes and communicates. `ReasoningMode` answers
how a particular agent expands its internal state:

- CoT performs one linear generation and records one thought node.
- ToT generates three isolated alternatives, evaluates them together, and
  prunes two branches using the evaluator's value/vote scores.
- GoT generates three source operations and runs a fourth operation that
  aggregates all three parents into one answer.

No reasoning mode creates business agents.

The benchmark performs batched scope-safe knowledge retrieval, then executes
`planner → solver → verifier`. Planner publishes a team summary, solver
publishes a verifier-only summary, and verifier sees both; none can see
another agent's raw working text. The general benchmark suite combines answers
by normalized majority vote with verifier tie-breaking. The sharing ablation
instead always scores the verifier output so the no-sharing control cannot
gain information through an external vote.

The canonical factorial evaluation keeps this topology fixed and independently
varies a uniform reasoning mode, sharing policy, and retrieval execution mode.
All three agents use CoT in the CoT conditions. All three use ToT in the ToT
conditions. Batched conditions submit the nine ToT branch requests to one
scope-safe grouping pass; independent conditions issue nine uncached
vector-store searches. GoT remains registered as a compatible future mode but
is not enabled in the default matrix.

## Scope-first retrieval

1. `PolicyEngine` evaluates tenant, workspace, domain, visibility, ownership,
   runtime region, and explicit permissions.
2. `ScopeEngine` resolves an exact `EffectiveView`.
3. `QueryPlanner` requires an explicitly share-safe public intent.
4. `scope_signature` binds public scope IDs, policy versions, index snapshots,
   memory types, embedding version, and retrieval parameters.
5. `RequestGrouper` uses medoid similarity plus a complete-link pairwise floor;
   it does not use transitive connected components.
6. Shared retrieval searches only the intersection of public views. Formal
   experiments apply this intersection as a Qdrant payload filter before
   vector ranking.
7. Every agent independently rechecks authorization, reranks, performs local
   fallback over `effective_view - shared_view`, and assembles context.

## Context and memory

`ContextManager` combines visible runtime state and retrieved evidence, removes
duplicates, resolves explicitly keyed conflicts, rejects expired/quarantined
content, and enforces a token budget.

General artifacts follow:

```text
draft → proposed → verified → committed
```

Private or branch-local content cannot become shared merely by being written.
`PromotionService` requires a verified state and explicit `promote` permission.
The benchmark's reasoning summaries use an explicit class-authorized publish
path; raw working text never follows that path.

## Security invariants

- Unauthorized content never enters the searchable corpus view.
- Shared vectors contain public intent only.
- Reuse never expands an individual view.
- Final ranking and context are agent-specific.
- Uncommitted branch state does not cross branches.
- Shared promotion is explicit and policy checked.
- Trace events reject raw content, full queries, and private intent.

## Deliberate exclusions

Legacy oracle dataset builders, topology-derived “CoT/ToT/GoT agents,”
query-matrix/SVD claims, and per-question gold-memory construction are not part
of the runtime. They may be adapted behind replay tooling but cannot define
production semantics.
