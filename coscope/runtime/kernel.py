"""Unified CoScope runtime kernel."""

from __future__ import annotations

import uuid

from coscope.adapters import (
    DeterministicEmbedding,
    EmbeddingAdapter,
    LLMAdapter,
    build_embedding,
    build_llm,
)
from coscope.config import CoScopeSettings
from coscope.context import ContextManager, ContextPacket
from coscope.core import (
    AgentInstance,
    AgentTopology,
    Artifact,
    ArtifactState,
    ArtifactType,
    MemoryEntry,
    RuntimeEvent,
    UsageLedger,
)
from coscope.memory import (
    CommitService,
    MemoryStore,
    PromotionService,
    RuntimeMemoryStore,
    build_qdrant_store,
)
from coscope.reasoning import (
    CoTRuntime,
    GoTRuntime,
    NodeStatus,
    ReasoningConfig,
    ReasoningMode,
    ReasoningNode,
    ReasoningRuntime,
    ToTRuntime,
)
from coscope.retrieval import (
    QueryPlanner,
    RequestGrouper,
    RetrievalCoordinator,
    RetrievalRequest,
    RetrievalResult,
)
from coscope.runtime.executor import ExecutionOutput, RuntimeExecutor, RuntimeInvocation
from coscope.runtime.llm_executor import LLMExecutor
from coscope.runtime.state import RunState
from coscope.scope import (
    Permission,
    PolicyEngine,
    ScopeDescriptor,
    ScopeEngine,
    Visibility,
    scope_signature,
)
from coscope.tracing import TraceRecorder


class CoScopeRuntime:
    """Scope-aware runtime for replay, simulation, and provider adapters."""

    def __init__(
        self,
        embedder: EmbeddingAdapter | None = None,
        llm: LLMAdapter | None = None,
        *,
        settings: CoScopeSettings | None = None,
        usage_ledger: UsageLedger | None = None,
        memory_store: MemoryStore | None = None,
    ):
        self.usage = usage_ledger or UsageLedger()
        self.embedder = embedder or DeterministicEmbedding()
        self.llm = llm
        self.settings = settings
        self.topology = AgentTopology()
        self.policy = PolicyEngine()
        self.scope_engine = ScopeEngine(self.policy)
        self.memory = (
            memory_store
            if memory_store is not None
            else RuntimeMemoryStore()
        )
        retrieval_settings = settings.retrieval if settings else None
        self.retrieval = RetrievalCoordinator(
            self.memory,
            self.embedder,
            grouper=(
                RequestGrouper(
                    medoid_threshold=retrieval_settings.medoid_threshold,
                    minimum_pairwise_similarity=(
                        retrieval_settings.minimum_pairwise_similarity
                    ),
                    max_group_size=retrieval_settings.max_group_size,
                )
                if retrieval_settings
                else None
            ),
            shared_candidate_k=(
                retrieval_settings.shared_candidate_k if retrieval_settings else 50
            ),
        )
        self.context = ContextManager(self.memory)
        self.query_planner = QueryPlanner()
        self.commit_service = CommitService(self.memory, self.embedder)
        self.promotion_service = PromotionService(self.memory, self.policy)
        self.traces = TraceRecorder()
        self.runs: dict[str, RunState] = {}
        self.reasoning: dict[tuple[str, str], ReasoningRuntime] = {}
        self.nodes: dict[tuple[str, str], ReasoningNode] = {}

    @classmethod
    def from_env(cls, env_file: str = ".env") -> CoScopeRuntime:
        """Create an offline or live runtime from environment configuration."""
        settings = CoScopeSettings.from_env(env_file)
        return cls.from_settings(settings)

    @classmethod
    def from_settings(
        cls,
        settings: CoScopeSettings,
        *,
        memory_namespace: str | None = None,
    ) -> CoScopeRuntime:
        """Create a runtime from already validated settings."""
        usage = UsageLedger()
        embedder: EmbeddingAdapter = (
            build_embedding(settings.embedding, usage)
            if settings.live
            else DeterministicEmbedding(
                settings.embedding.dimension
                if settings.memory.provider == "qdrant"
                else 128
            )
        )
        memory_store = (
            build_qdrant_store(
                settings.memory,
                dimension=int(
                    getattr(embedder, "dimension", settings.embedding.dimension)
                ),
                namespace=memory_namespace,
            )
            if settings.memory.provider == "qdrant"
            else RuntimeMemoryStore()
        )
        return cls(
            embedder=embedder,
            llm=build_llm(settings.llm, usage) if settings.live else None,
            settings=settings,
            usage_ledger=usage,
            memory_store=memory_store,
        )

    def close(self, *, purge_memory: bool = False) -> None:
        """Release task-local memory, optionally removing its DB namespace."""
        try:
            if purge_memory:
                self.memory.clear()
        finally:
            # Live factorial tasks construct an SDK client per isolated runtime.
            # Closing it here prevents finished HTTP connections from remaining
            # in CLOSE_WAIT for the lifetime of the experiment coordinator.
            close = getattr(self.llm, "close", None)
            if callable(close):
                close()

    def live_executor(
        self, *, system_instructions: tuple[str, ...] = ()
    ) -> LLMExecutor:
        if self.llm is None:
            raise RuntimeError(
                "no live LLM configured; set COSCOPE_RUNTIME_MODE=live "
                "and create the runtime with CoScopeRuntime.from_env()"
            )
        return LLMExecutor(self.llm, system_instructions=system_instructions)

    def create_run(
        self, task_id: str, *, run_id: str | None = None, metadata: dict | None = None
    ) -> RunState:
        resolved_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        if resolved_id in self.runs:
            raise ValueError(f"duplicate run: {resolved_id}")
        run = RunState(resolved_id, task_id, metadata=dict(metadata or {}))
        self.runs[resolved_id] = run
        self._trace("run_created", resolved_id, attributes={"task_id": task_id})
        return run

    def register_agent(self, agent: AgentInstance) -> None:
        self.topology.add_agent(agent)

    def connect_agents(self, source_id: str, target_id: str) -> None:
        self.topology.connect(source_id, target_id)

    def start_reasoning(
        self,
        agent_id: str,
        config: ReasoningConfig,
        *,
        runtime_id: str | None = None,
    ):
        if agent_id not in self.topology.agents:
            raise KeyError(f"unknown agent: {agent_id}")
        resolved_id = runtime_id or f"{agent_id}:{config.mode.value}"
        runtime: ReasoningRuntime
        if config.mode == ReasoningMode.COT:
            runtime = CoTRuntime(agent_id, config)
            root = runtime.create_root()
        elif config.mode == ReasoningMode.TOT:
            runtime = ToTRuntime(agent_id, config)
            root = runtime.create_root()
        else:
            runtime = GoTRuntime(agent_id, config)
            root = runtime.add_node("root")
        self.reasoning[(agent_id, resolved_id)] = runtime
        self._register_nodes(agent_id, runtime.nodes())
        return runtime, root

    def refresh_reasoning_nodes(
        self, agent_id: str, runtime: ReasoningRuntime
    ) -> None:
        self._register_nodes(agent_id, runtime.nodes())

    def ingest_memory(self, entry: MemoryEntry) -> MemoryEntry:
        """System ingestion path for external/public sources."""
        if entry.vector is None:
            entry.vector = self.embedder.embed(entry.content)
        self.memory.add(entry)
        return entry

    def write_artifact(
        self,
        actor_id: str,
        artifact: Artifact,
        scope: ScopeDescriptor,
        *,
        runtime_region: str | None = None,
        searchable: bool = True,
    ) -> MemoryEntry:
        actor = self.topology.agents[actor_id]
        decision = self.policy.decide(
            actor, scope, Permission.WRITE, runtime_region=runtime_region
        )
        if not decision.allowed:
            raise PermissionError(f"write denied: {decision.reason}")
        if artifact.owner_agent_id != actor_id:
            raise PermissionError("an actor cannot write another agent's artifact")
        entry = self.commit_service.write(
            artifact,
            scope,
            searchable=searchable,
        )
        self._trace(
            "memory_written",
            str(artifact.metadata.get("run_id", "unknown")),
            actor_id,
            {"memory_id": entry.memory_id, "scope_id": scope.scope_id},
        )
        return entry

    def verify_memory(self, memory_id: str) -> MemoryEntry:
        entry = self._require_memory(memory_id)
        if entry.state != ArtifactState.PROPOSED:
            raise ValueError("only proposed memory can be verified")
        entry.state = ArtifactState.VERIFIED
        self.memory.replace(entry)
        return entry

    def promote_memory(
        self, actor_id: str, memory_id: str, target_scope: ScopeDescriptor
    ) -> MemoryEntry:
        return self.promotion_service.promote(
            self.topology.agents[actor_id],
            self._require_memory(memory_id),
            target_scope,
        )

    def create_request(
        self,
        *,
        run_id: str,
        agent_id: str,
        reasoning_node_id: str,
        full_query: str,
        public_intent: str | None = None,
        private_intent: str | None = None,
        full_query_is_public: bool = False,
        required_facets: tuple[str, ...] = (),
        context_budget: int | None = None,
        retrieval_budget: int = 20,
        memory_types: frozenset[str] = frozenset(),
    ) -> RetrievalRequest:
        if run_id not in self.runs:
            raise KeyError(f"unknown run: {run_id}")
        agent = self.topology.agents[agent_id]
        node = self.nodes[(agent_id, reasoning_node_id)]
        intent = self.query_planner.plan(
            full_query,
            public_intent=public_intent,
            private_intent=private_intent,
            full_query_is_public=full_query_is_public,
        )
        view = self.scope_engine.resolve(
            agent, self.memory.scopes(), runtime_region=node.runtime_region
        )
        vector = self.embedder.embed(intent.public_intent)
        signature = scope_signature(
            view,
            memory_types=memory_types,
            embedding_model_version=self.embedder.model_version,
            retrieval_parameters={
                "candidate_k": self.retrieval.shared_retriever.candidate_k,
                "store_revision": self.memory.revision,
            },
        )
        return RetrievalRequest(
            request_id=f"req_{uuid.uuid4().hex[:16]}",
            run_id=run_id,
            agent_id=agent_id,
            reasoning_node_id=reasoning_node_id,
            full_query=intent.full_query,
            public_intent=intent.public_intent,
            private_intent=intent.private_intent,
            public_vector=vector,
            scope_signature=signature,
            effective_view=view,
            required_facets=required_facets,
            context_budget=(
                context_budget
                if context_budget is not None
                else agent.context_policy.default_context_budget
            ),
            retrieval_budget=retrieval_budget,
            role=agent.role,
            state_summary=str(agent.runtime_state.get("summary", "")),
            memory_types=memory_types,
            readable_artifact_types=(
                agent.context_policy.readable_artifact_types
            ),
            channel_budgets=dict(agent.context_policy.channel_budgets),
        )

    def refresh_context_view(self, request: RetrievalRequest) -> RetrievalRequest:
        """Refresh a pending request after runtime memories change.

        Retrieval candidates remain unchanged; newly published runtime state can
        enter context only if the requesting agent's current policy allows it.
        """
        agent = self.topology.agents[request.agent_id]
        node = self.nodes[(request.agent_id, request.reasoning_node_id)]
        request.effective_view = self.scope_engine.resolve(
            agent,
            self.memory.scopes(),
            runtime_region=node.runtime_region,
        )
        request.scope_signature = scope_signature(
            request.effective_view,
            memory_types=request.memory_types,
            embedding_model_version=self.embedder.model_version,
            retrieval_parameters={
                "candidate_k": self.retrieval.shared_retriever.candidate_k,
                "store_revision": self.memory.revision,
            },
        )
        return request

    def record_thinking(
        self,
        *,
        actor_id: str,
        reasoning_node_id: str,
        private_content: str,
        public_summary: str | None = None,
        recipients: frozenset[str] | None = frozenset(),
    ) -> tuple[MemoryEntry, MemoryEntry | None]:
        """Store raw working text privately and optionally publish a summary.

        ``recipients=None`` publishes to the whole team. A non-empty recipient
        set creates a directed restricted scope. The raw text is never copied
        into the published memory.
        """
        actor = self.topology.agents[actor_id]
        node = self.nodes[(actor_id, reasoning_node_id)]
        policy = actor.context_policy
        if not policy.allow_private_thinking:
            raise PermissionError(
                f"agent class {actor.agent_class_id} cannot store private thinking"
            )
        private_scope = ScopeDescriptor(
            frozenset({"runtime_thinking"}),
            node.runtime_region,
            (
                Visibility.BRANCH_PRIVATE
                if node.reasoning_mode == ReasoningMode.TOT
                else Visibility.NODE_LOCAL
            ),
            owner_id=actor_id,
            memory_types=frozenset({"thinking"}),
        )
        private_entry = self.write_artifact(
            actor_id,
            Artifact(
                private_content,
                ArtifactType.REASONING_SUMMARY,
                "private_thinking",
                reasoning_node_id,
                actor_id,
                state=ArtifactState.DRAFT,
                metadata={"context_flow": "private_thinking"},
            ),
            private_scope,
            runtime_region=node.runtime_region,
            searchable=False,
        )
        if not public_summary:
            return private_entry, None
        if not policy.allow_public_thinking:
            raise PermissionError(
                f"agent class {actor.agent_class_id} cannot publish thinking"
            )
        if (
            policy.publishable_artifact_types
            and ArtifactType.REASONING_SUMMARY
            not in policy.publishable_artifact_types
        ):
            raise PermissionError(
                f"agent class {actor.agent_class_id} cannot publish "
                "reasoning_summary artifacts"
            )
        if recipients is not None:
            unknown = recipients - self.topology.agents.keys()
            if unknown:
                raise KeyError(f"unknown thinking recipients: {sorted(unknown)}")
        permissions: dict[str, frozenset[Permission]] = {
            actor_id: frozenset(
                {Permission.READ, Permission.WRITE, Permission.SHARE}
            )
        }
        if recipients is not None:
            for recipient in recipients:
                permissions[recipient] = frozenset({Permission.READ})
        public_scope = ScopeDescriptor(
            frozenset({"runtime_thinking"}),
            f"{node.runtime_region}/published/{uuid.uuid4().hex[:12]}",
            (
                Visibility.TEAM_SHARED
                if recipients is None
                else Visibility.RESTRICTED
            ),
            owner_id=actor_id,
            permissions=permissions,
            memory_types=frozenset({"thinking"}),
            provenance={"source_node": reasoning_node_id},
            trust_level="agent_summary",
        )
        public_entry = self.write_artifact(
            actor_id,
            Artifact(
                public_summary,
                ArtifactType.REASONING_SUMMARY,
                "published_thinking",
                reasoning_node_id,
                actor_id,
                state=ArtifactState.COMMITTED,
                metadata={
                    "context_flow": "public_summary",
                    "recipients": (
                        "team" if recipients is None else sorted(recipients)
                    ),
                },
            ),
            public_scope,
            runtime_region=node.runtime_region,
            searchable=False,
        )
        return private_entry, public_entry

    def retrieve_batch(
        self, requests: list[RetrievalRequest]
    ) -> dict[str, RetrievalResult]:
        results = self.retrieval.retrieve_batch(requests)
        self._trace_retrieval_results(requests, results)
        return results

    def retrieve_independently(
        self,
        requests: list[RetrievalRequest],
    ) -> dict[str, RetrievalResult]:
        """Execute one uncached vector-store query per request."""
        results = self.retrieval.retrieve_independently(requests)
        self._trace_retrieval_results(requests, results)
        return results

    def _trace_retrieval_results(
        self,
        requests: list[RetrievalRequest],
        results: dict[str, RetrievalResult],
    ) -> None:
        for request in requests:
            result = results[request.request_id]
            self._trace(
                "retrieval_completed",
                request.run_id,
                request.agent_id,
                {
                    "request_id": request.request_id,
                    "group_id": result.group_id,
                    "candidate_ids": [
                        candidate.memory.memory_id for candidate in result.candidates
                    ],
                    "fallback": result.fallback_triggered,
                },
            )

    def assemble_context(
        self, request: RetrievalRequest, result: RetrievalResult
    ) -> ContextPacket:
        return self.context.assemble(request, result)

    def execute_batch(
        self,
        run_id: str,
        invocations: list[RuntimeInvocation],
        executor: RuntimeExecutor,
    ) -> dict[str, ExecutionOutput]:
        requests = [
            self.create_request(
                run_id=run_id,
                agent_id=item.agent_id,
                reasoning_node_id=item.reasoning_node_id,
                full_query=item.full_query,
                public_intent=item.public_intent,
                private_intent=item.private_intent,
                required_facets=item.required_facets,
                context_budget=item.context_budget,
                retrieval_budget=item.retrieval_budget,
                memory_types=item.memory_types,
            )
            for item in invocations
        ]
        results = self.retrieve_batch(requests)
        outputs: dict[str, ExecutionOutput] = {}
        for invocation, request in zip(invocations, requests, strict=True):
            agent = self.topology.agents[invocation.agent_id]
            node = self.nodes[(invocation.agent_id, invocation.reasoning_node_id)]
            packet = self.assemble_context(request, results[request.request_id])
            node.status = NodeStatus.RUNNING
            output = executor.invoke(agent, node, packet)
            outputs[request.request_id] = output
            if output.artifacts:
                output_scope = invocation.output_scope
                if output_scope is None:
                    raise ValueError(
                        "output_scope is required when an executor returns artifacts"
                    )
                for artifact in output.artifacts:
                    artifact.metadata.setdefault("run_id", run_id)
                    self.write_artifact(
                        invocation.agent_id,
                        artifact,
                        output_scope,
                        runtime_region=node.runtime_region,
                    )
            node.status = NodeStatus.COMPLETED
        return outputs

    def _register_nodes(self, agent_id: str, nodes: list[ReasoningNode]) -> None:
        for node in nodes:
            self.nodes[(agent_id, node.node_id)] = node

    def _require_memory(self, memory_id: str) -> MemoryEntry:
        entry = self.memory.get(memory_id)
        if entry is None:
            raise KeyError(f"unknown memory: {memory_id}")
        return entry

    def _trace(
        self,
        event_type: str,
        run_id: str,
        actor_id: str | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.traces.record(
            RuntimeEvent(event_type, run_id, actor_id, attributes=attributes or {})
        )
