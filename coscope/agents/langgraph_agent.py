"""
LangGraph Agent Integration for CoScope.

Provides integration with LangGraph for stateful multi-agent coordination.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from coscope.retrieval.pipeline import RetrievalPipeline
    from coscope.memory.store import MemoryManager

logger = logging.getLogger(__name__)

_langgraph_available = False

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    _langgraph_available = True
except ImportError:
    logger.debug("LangGraph not available. Install with: pip install langgraph")


# ============================================================
# LangGraph State Definition
# ============================================================


class CoScopeAgentState(TypedDict, total=False):
    """
    State schema for CoScope LangGraph agents.

    This defines the shared state that flows through the graph.
    """

    # Agent identification
    agent_id: str
    agent_role: str

    # Current task
    task: str
    task_id: Optional[str]

    # Memory context
    retrieved_memories: List[Dict[str, Any]]
    selected_memory_id: Optional[str]

    # Agent state
    planning_step: str
    reasoning_branch: List[str]
    completed_steps: List[str]

    # Session context
    session_id: str
    task_shared_memory: Dict[str, Any]

    # Messages
    messages: List[Dict[str, Any]]

    # Results
    result: Optional[str]
    error: Optional[str]


# ============================================================
# Node Definitions
# ============================================================


class CoScopeGraphNodes:
    """
    Pre-defined nodes for CoScope LangGraph integration.

    These nodes can be composed into custom agent graphs.
    """

    def __init__(
        self,
        pipeline: "RetrievalPipeline",
        memory_manager: "MemoryManager",
    ):
        self.pipeline = pipeline
        self.memory_manager = memory_manager

    def retrieve_node(
        self, state: CoScopeAgentState
    ) -> CoScopeAgentState:
        """
        Node that retrieves relevant memories for the current task.
        """
        from coscope.core.types import AgentConfig, AgentRole, AgentState, MemoryType, RetrievalRequest, ScopeSpec
        from coscope.core.types import VisibilityLevel, PolicyConstraints

        agent_id = state.get("agent_id", "unknown")
        task = state.get("task", "")
        agent_role_str = state.get("agent_role", "custom")

        try:
            role = AgentRole(agent_role_str)
        except ValueError:
            role = AgentRole.CUSTOM

        # Create agent config
        agent_config = AgentConfig(
            agent_id=agent_id,
            role=role,
            allowed_scopes=["task/shared", "session/current", "workspace/default"],
            allowed_memory_types=[
                MemoryType.EPISODIC,
                MemoryType.ARTIFACT,
                MemoryType.SHARED,
            ],
            policy=PolicyConstraints(
                visibility=[VisibilityLevel.TEAM],
                max_clearance=3,
            ),
        )

        # Create retrieval request
        request = RetrievalRequest(
            agent_id=agent_id,
            role=role,
            query=task,
            scope=ScopeSpec(
                shared_scopes=["task/shared", "session/current"],
                workspace_scopes=["workspace/default"],
            ),
            memory_types=[MemoryType.EPISODIC, MemoryType.ARTIFACT, MemoryType.SHARED],
            policy=PolicyConstraints(
                visibility=[VisibilityLevel.TEAM],
                max_clearance=3,
            ),
            state=AgentState(
                plan_node=state.get("planning_step"),
                known_evidence_ids=[],
                tool_context={},
                reasoning_branch=state.get("reasoning_branch", []),
            ),
        )

        # Execute retrieval
        results = self.pipeline.retrieve([request])

        if results:
            result = results[0]
            memories = [
                {
                    "memory_id": c.memory.memory_id,
                    "content": c.memory.content,
                    "type": c.memory.memory_type.value,
                    "score": c.score,
                    "source": c.source,
                }
                for c in result.candidates[:10]
            ]
            state["retrieved_memories"] = memories
        else:
            state["retrieved_memories"] = []

        return state

    def plan_node(self, state: CoScopeAgentState) -> CoScopeAgentState:
        """
        Node that plans the next steps based on retrieved memories.
        """
        # Placeholder for planning logic
        # In a real implementation, this would use an LLM
        state["planning_step"] = "planned"
        return state

    def execute_node(self, state: CoScopeAgentState) -> CoScopeAgentState:
        """
        Node that executes the planned action.
        """
        # Placeholder for execution logic
        state["completed_steps"] = state.get("completed_steps", []) + ["executed"]
        return state

    def reflect_node(self, state: CoScopeAgentState) -> CoScopeAgentState:
        """
        Node that reflects on the results and updates memory.
        """
        # Placeholder for reflection logic
        return state

    def should_retrieve_again(
        self, state: CoScopeAgentState
    ) -> Literal["retrieve", "end"]:
        """
        Conditional edge to decide if more retrieval is needed.
        """
        # Simple condition: if we have enough memories, stop
        if len(state.get("retrieved_memories", [])) >= 5:
            return "end"
        return "retrieve"


# ============================================================
# CoScope Agent Graph Builder
# ============================================================


class CoScopeGraphBuilder:
    """
    Builder for creating CoScope-integrated LangGraph agents.

    This provides a fluent API for composing agent graphs with
    CoScope retrieval capabilities.
    """

    def __init__(
        self,
        pipeline: "RetrievalPipeline",
        memory_manager: "MemoryManager",
        checkpointer: Optional[Any] = None,
    ):
        if not _langgraph_available:
            raise ImportError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

        from langgraph.graph import StateGraph, END

        self.pipeline = pipeline
        self.memory_manager = memory_manager
        self.checkpointer = checkpointer or MemorySaver()

        self.graph: Optional[StateGraph] = None
        self._nodes: Dict[str, Callable] = {}
        self._edges: List[Tuple[str, str]] = []
        self._conditional_edges: Dict[str, Callable] = {}

    def add_node(
        self,
        name: str,
        func: Callable[[CoScopeAgentState], CoScopeAgentState],
    ) -> "CoScopeGraphBuilder":
        """Add a node to the graph."""
        self._nodes[name] = func
        return self

    def add_retrieve_node(
        self,
        name: str = "retrieve",
    ) -> "CoScopeGraphBuilder":
        """Add a CoScope retrieval node."""
        nodes = CoScopeGraphNodes(self.pipeline, self.memory_manager)
        self._nodes[name] = nodes.retrieve_node
        return self

    def add_edge(
        self,
        source: str,
        destination: str,
    ) -> "CoScopeGraphBuilder":
        """Add a directed edge between nodes."""
        self._edges.append((source, destination))
        return self

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[CoScopeAgentState], str],
        path_map: Optional[Dict[str, str]] = None,
    ) -> "CoScopeGraphBuilder":
        """Add a conditional edge."""
        self._conditional_edges[source] = (condition, path_map or {})
        return self

    def set_entry_point(self, node: str) -> "CoScopeGraphBuilder":
        """Set the entry point node."""
        self._entry_point = node
        return self

    def build(self, name: str = "coscope_agent") -> Any:
        """
        Build and compile the graph.

        Returns:
            Compiled LangGraph agent
        """
        from langgraph.graph import StateGraph, END

        # Create graph
        graph = StateGraph(CoScopeAgentState)

        # Add nodes
        for node_name, node_func in self._nodes.items():
            graph.add_node(node_name, node_func)

        # Add edges
        for source, dest in self._edges:
            graph.add_edge(source, dest)

        # Add conditional edges
        for source, (condition, path_map) in self._conditional_edges.items():
            graph.add_conditional_edges(source, condition, path_map)

        # Set entry point
        entry = getattr(self, "_entry_point", list(self._nodes.keys())[0])
        graph.set_entry_point(entry)

        # Set finish point
        if END not in self._edges and not self._conditional_edges:
            graph.add_edge(entry, END)

        # Compile
        compiled = graph.compile(checkpointer=self.checkpointer)

        self.graph = graph
        return compiled

    @classmethod
    def create_simple_agent(
        cls,
        pipeline: "RetrievalPipeline",
        memory_manager: "MemoryManager",
    ) -> Any:
        """
        Create a simple CoScope agent graph with default nodes.

        The graph has:
        retrieve -> plan -> execute -> reflect -> end
        """
        builder = cls(pipeline, memory_manager)

        # Add default nodes
        nodes = CoScopeGraphNodes(pipeline, memory_manager)
        builder.add_node("retrieve", nodes.retrieve_node)
        builder.add_node("plan", nodes.plan_node)
        builder.add_node("execute", nodes.execute_node)
        builder.add_node("reflect", nodes.reflect_node)

        # Set flow
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "plan")
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", "reflect")
        builder.add_edge("reflect", END)

        return builder.build()


# ============================================================
# Multi-Agent Coordination
# ============================================================


class MultiAgentCoordinator:
    """
    Coordinates multiple CoScope agents in a LangGraph workflow.

    This enables complex multi-agent scenarios where agents can:
    - Share retrieved memories
    - Coordinate via shared state
    - Route tasks between agents
    """

    def __init__(
        self,
        pipelines: Dict[str, "RetrievalPipeline"],
        memory_managers: Dict[str, "MemoryManager"],
    ):
        self.pipelines = pipelines
        self.memory_managers = memory_managers

    def create_supervisor_graph(
        self,
        agent_configs: Dict[str, Any],
    ) -> Any:
        """
        Create a supervisor-managed multi-agent graph.

        The supervisor can route tasks to specialized agents
        and aggregate their results.
        """
        if not _langgraph_available:
            raise ImportError("LangGraph is required for multi-agent coordination")

        from langgraph.graph import StateGraph, END, START

        class SupervisorState(TypedDict, total=False):
            task: str
            assigned_agent: Optional[str]
            agent_results: Dict[str, Any]
            final_result: Optional[str]

        def supervisor_node(state: SupervisorState) -> SupervisorState:
            """Route task to appropriate agent."""
            task = state.get("task", "")
            agent_results = state.get("agent_results", {})

            # Simple routing based on keywords
            if "plan" in task.lower() or "constraint" in task.lower():
                assigned = "planner"
            elif "verify" in task.lower() or "check" in task.lower():
                assigned = "verifier"
            elif "critic" in task.lower() or "conflict" in task.lower():
                assigned = "critic"
            else:
                assigned = "solver"

            state["assigned_agent"] = assigned
            return state

        def agent_result_node(state: SupervisorState) -> SupervisorState:
            """Collect results from assigned agent."""
            # In a real implementation, this would invoke the actual agent
            agent = state.get("assigned_agent")
            if agent and agent not in state.get("agent_results", {}):
                state["agent_results"] = {
                    **state.get("agent_results", {}),
                    agent: {"status": "completed"},
                }
            return state

        def should_continue(state: SupervisorState) -> Literal["agent", "end"]:
            """Check if task is complete."""
            if len(state.get("agent_results", {})) >= 1:
                return "end"
            return "agent"

        # Build graph
        graph = StateGraph(SupervisorState)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("agent", agent_result_node)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            should_continue,
            {"agent": "agent", "end": END},
        )
        graph.add_edge("agent", END)

        return graph.compile()


# ============================================================
# Utility Functions
# ============================================================


def is_langgraph_available() -> bool:
    """Check if LangGraph is available."""
    return _langgraph_available


def create_default_checkpointer(
    backend: Literal["memory", "sqlite"] = "memory",
    **kwargs,
) -> Any:
    """
    Create a checkpointer for LangGraph state persistence.
    """
    if not _langgraph_available:
        raise ImportError("LangGraph is required for checkpointer")

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    elif backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = kwargs.get("path", "./coscope_data/checkpoints/checkpoints.db")
        return SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    else:
        raise ValueError(f"Unknown checkpointer backend: {backend}")
