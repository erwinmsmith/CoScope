"""LLM execution strategies aligned with CoT, ToT, and GoT operations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from coscope.adapters.llm import LLMAdapter
from coscope.reasoning.cot_runtime import CoTRuntime
from coscope.reasoning.got_runtime import GoTRuntime
from coscope.reasoning.modes import ReasoningMode
from coscope.reasoning.node import NodeStatus, ReasoningNode
from coscope.reasoning.tot_runtime import ToTRuntime

TOT_BRANCH_STRATEGIES = (
    "derive a direct step-by-step solution",
    "try an alternative decomposition and check edge cases",
    "work backward from the requested answer and verify evidence",
)
GOT_SOURCE_OPERATIONS = (
    "generate a primary solution path",
    "extract and connect the strongest evidence path",
    "generate a countercheck focused on contradictions or arithmetic errors",
)


@dataclass
class ReasoningOutcome:
    output: str
    mode: ReasoningMode
    llm_calls: int
    node_count: int
    generated_thoughts: int
    metadata: dict[str, object]


def execute_live_strategy(
    mode: ReasoningMode,
    runtime: CoTRuntime | ToTRuntime | GoTRuntime,
    root: ReasoningNode,
    llm: LLMAdapter,
    *,
    question: str,
    authorized_context: str,
    answer_instruction: str,
    tot_branches: list[ReasoningNode] | None = None,
    tot_branch_contexts: dict[str, str] | None = None,
    got_sources: list[ReasoningNode] | None = None,
    got_source_contexts: dict[str, str] | None = None,
) -> ReasoningOutcome:
    if mode == ReasoningMode.COT and isinstance(runtime, CoTRuntime):
        return _execute_cot(
            runtime,
            root,
            llm,
            question,
            authorized_context,
            answer_instruction,
        )
    if mode == ReasoningMode.TOT and isinstance(runtime, ToTRuntime):
        return _execute_tot(
            runtime,
            root,
            llm,
            question,
            authorized_context,
            answer_instruction,
            branches=tot_branches,
            branch_contexts=tot_branch_contexts,
        )
    if mode == ReasoningMode.GOT and isinstance(runtime, GoTRuntime):
        return _execute_got(
            runtime,
            root,
            llm,
            question,
            authorized_context,
            answer_instruction,
            sources=got_sources,
            source_contexts=got_source_contexts,
        )
    raise TypeError(f"runtime does not match reasoning mode {mode.value}")


def _invoke(
    llm: LLMAdapter,
    system: str,
    user: str,
) -> str:
    return llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    ).text


def _base_prompt(question: str, context: str, instruction: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Authorized context:\n{context or '(none)'}\n\n"
        f"Output rule:\n{instruction}"
    )


def _execute_cot(
    runtime: CoTRuntime,
    root: ReasoningNode,
    llm: LLMAdapter,
    question: str,
    context: str,
    instruction: str,
) -> ReasoningOutcome:
    root.status = NodeStatus.RUNNING
    output = _invoke(
        llm,
        "Use a single coherent chain of intermediate reasoning steps, then answer.",
        _base_prompt(question, context, instruction),
    )
    root.local_state["thought"] = output
    root.status = NodeStatus.COMPLETED
    return ReasoningOutcome(
        output,
        ReasoningMode.COT,
        llm_calls=1,
        node_count=len(runtime.nodes()),
        generated_thoughts=1,
        metadata={"chain_depth": 1},
    )


def _execute_tot(
    runtime: ToTRuntime,
    root: ReasoningNode,
    llm: LLMAdapter,
    question: str,
    context: str,
    instruction: str,
    *,
    branches: list[ReasoningNode] | None = None,
    branch_contexts: dict[str, str] | None = None,
) -> ReasoningOutcome:
    active_branches = (
        branches
        if branches is not None
        else runtime.branch(root.node_id, runtime.config.branching_factor)
    )
    candidate_outputs = []
    for index, branch in enumerate(active_branches):
        branch.status = NodeStatus.RUNNING
        output = _invoke(
            llm,
            "Generate one independent Tree-of-Thought candidate. "
            "Do not use another branch's state.",
            _base_prompt(
                question,
                (branch_contexts or {}).get(branch.node_id, context),
                (
                    "Strategy: "
                    f"{TOT_BRANCH_STRATEGIES[index % len(TOT_BRANCH_STRATEGIES)]}. "
                    f"{instruction}"
                ),
            ),
        )
        branch.local_state["thought"] = output
        branch.status = NodeStatus.COMPLETED
        candidate_outputs.append(output)

    candidates = "\n\n".join(
        f"BRANCH {index + 1}:\n{output}"
        for index, output in enumerate(candidate_outputs)
    )
    final = _invoke(
        llm,
        "Act as the Tree-of-Thought value/vote evaluator. Compare candidates, "
        "select the most correct branch, verify it, and return the final answer.",
        (
            f"Question:\n{question}\n\n{candidates}\n\n"
            "First write SELECTED_BRANCH: <number>.\n"
            f"Then {instruction}"
        ),
    )
    selected_match = re.search(r"SELECTED_BRANCH:\s*([1-9][0-9]*)", final)
    selected_index = (
        min(int(selected_match.group(1)), len(active_branches)) - 1
        if selected_match
        else 0
    )
    for index, branch in enumerate(active_branches):
        runtime.score(branch.node_id, 1.0 if index == selected_index else 0.0)
    pruned = runtime.prune(1)
    return ReasoningOutcome(
        final,
        ReasoningMode.TOT,
        llm_calls=len(active_branches) + 1,
        node_count=len(runtime.nodes()),
        generated_thoughts=len(active_branches),
        metadata={
            "branching_factor": len(active_branches),
            "selected_branch": selected_index + 1,
            "pruned_node_ids": pruned,
        },
    )


def _execute_got(
    runtime: GoTRuntime,
    root: ReasoningNode,
    llm: LLMAdapter,
    question: str,
    context: str,
    instruction: str,
    *,
    sources: list[ReasoningNode] | None = None,
    source_contexts: dict[str, str] | None = None,
) -> ReasoningOutcome:
    active_sources = (
        sources
        if sources is not None
        else [
            root,
            runtime.add_node("evidence_path"),
            runtime.add_node("countercheck_path"),
        ]
    )
    thoughts = []
    for node, operation in zip(
        active_sources,
        GOT_SOURCE_OPERATIONS,
        strict=True,
    ):
        node.status = NodeStatus.RUNNING
        output = _invoke(
            llm,
            "Execute one independent Graph-of-Thought operation.",
            _base_prompt(
                question,
                (source_contexts or {}).get(node.node_id, context),
                f"{operation}. {instruction}",
            ),
        )
        node.local_state["thought"] = output
        runtime.complete(node.node_id)
        thoughts.append(output)

    aggregate = runtime.add_node(
        "aggregate",
        parent_ids=[node.node_id for node in active_sources],
    )
    parent_text = "\n\n".join(
        f"PARENT {index + 1}:\n{thought}"
        for index, thought in enumerate(thoughts)
    )
    aggregate.status = NodeStatus.RUNNING
    final = _invoke(
        llm,
        "Execute a Graph-of-Thought aggregate operation. Reconcile all parent "
        "thoughts, resolve conflicts, and produce one verified answer.",
        f"Question:\n{question}\n\n{parent_text}\n\n{instruction}",
    )
    aggregate.local_state["thought"] = final
    aggregate.local_state["merged_from"] = [
        node.node_id for node in active_sources
    ]
    runtime.complete(aggregate.node_id)
    return ReasoningOutcome(
        final,
        ReasoningMode.GOT,
        llm_calls=len(active_sources) + 1,
        node_count=len(runtime.nodes()),
        generated_thoughts=len(active_sources),
        metadata={
            "aggregate_node": aggregate.node_id,
            "parent_node_ids": [node.node_id for node in active_sources],
        },
    )
