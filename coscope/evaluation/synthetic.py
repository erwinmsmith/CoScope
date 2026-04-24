"""
Synthetic retrieval cases for quick ablation checks.

These cases are intentionally small and deterministic enough to smoke-test the
experiment flow before wiring in full dataset episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from coscope.core.types import (
    MemoryType,
    PolicyConstraints,
    RetrievalRequest,
    ScopeSpec,
    VisibilityLevel,
)
from coscope.engine import CoScope
from coscope.evaluation.metrics import GoldMap
from coscope.evaluation.runner import (
    DEFAULT_VARIANTS,
    VariantRun,
    evaluate_variants,
    format_variant_table,
)


@dataclass
class SyntheticCase:
    """A runnable synthetic retrieval episode."""

    case_id: str
    label: str
    coscope: CoScope
    requests: List[RetrievalRequest]
    gold_by_request: GoldMap
    conflict_request_ids: List[str] = field(default_factory=list)


@dataclass
class SyntheticVariantSummary:
    """Aggregated metrics for one variant across synthetic cases."""

    variant: str
    case_runs: Dict[str, VariantRun]

    @property
    def recall_at_k(self) -> float:
        return _mean(run.report.recall_at_k for run in self.case_runs.values())

    @property
    def mrr_at_k(self) -> float:
        return _mean(run.report.mrr_at_k for run in self.case_runs.values())

    @property
    def first_stage_savings(self) -> float:
        return _mean(run.report.first_stage_savings for run in self.case_runs.values())

    @property
    def false_merge_rate(self) -> float:
        false_merges = sum(run.report.false_merge_count for run in self.case_runs.values())
        conflict_total = sum(run.report.conflict_request_count for run in self.case_runs.values())
        return false_merges / conflict_total if conflict_total else 0.0

    @property
    def first_stage_actual(self) -> int:
        return sum(run.report.first_stage_actual for run in self.case_runs.values())

    @property
    def first_stage_independent(self) -> int:
        return sum(run.report.first_stage_independent for run in self.case_runs.values())


def build_synthetic_suite() -> List[SyntheticCase]:
    """Build S1/S2/S3/S4 synthetic cases."""
    return [
        build_s1_high_overlap(),
        build_s2_partial_overlap(),
        build_s3_low_overlap(),
        build_s4_policy_conflict(),
    ]


def evaluate_synthetic_suite(
    cases: Optional[Sequence[SyntheticCase]] = None,
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    k: int = 3,
) -> List[SyntheticVariantSummary]:
    """Run variant evaluation over every synthetic case and aggregate by variant."""
    cases = list(cases or build_synthetic_suite())
    by_variant: Dict[str, Dict[str, VariantRun]] = {variant: {} for variant in variants}

    for case in cases:
        runs = evaluate_variants(
            case.coscope,
            case.requests,
            case.gold_by_request,
            variants=variants,
            k=k,
            conflict_request_ids=case.conflict_request_ids,
        )
        for run in runs:
            by_variant[run.variant][case.case_id] = run

    return [
        SyntheticVariantSummary(variant=variant, case_runs=by_variant[variant])
        for variant in variants
    ]


def format_synthetic_table(
    summaries: Sequence[SyntheticVariantSummary],
    *,
    digits: int = 4,
) -> str:
    """Format aggregated synthetic results as a Markdown table."""
    headers = [
        "Variant",
        "Recall@k",
        "MRR@k",
        "Savings",
        "FMR",
        "First-stage",
        "Cases",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary.variant.upper(),
                _fmt(summary.recall_at_k, digits),
                _fmt(summary.mrr_at_k, digits),
                _fmt(summary.first_stage_savings, digits),
                _fmt(summary.false_merge_rate, digits),
                f"{summary.first_stage_actual}/{summary.first_stage_independent}",
                str(len(summary.case_runs)),
            ]
        )
    return _markdown_table(headers, rows)


def format_synthetic_case_tables(
    summaries: Sequence[SyntheticVariantSummary],
    *,
    digits: int = 4,
) -> str:
    """Format one comparison table per synthetic case."""
    case_ids = []
    for summary in summaries:
        for case_id in summary.case_runs:
            if case_id not in case_ids:
                case_ids.append(case_id)

    sections = []
    for case_id in case_ids:
        runs = [summary.case_runs[case_id] for summary in summaries if case_id in summary.case_runs]
        sections.append(f"### {case_id}\n{format_variant_table(runs, digits=digits)}")
    return "\n\n".join(sections)


def build_s1_high_overlap() -> SyntheticCase:
    """S1: all agents share the same scope/type/policy."""
    coscope = _base_coscope()
    _register_team_agent(coscope, "planner_s1", "planner")
    _register_team_agent(coscope, "solver_s1", "solver")
    _register_team_agent(coscope, "critic_s1", "critic")

    gold = coscope.add_memory(
        "S1 alpha shared evidence for all agents.",
        "task/s1/shared",
        MemoryType.SEMANTIC,
        confidence=0.95,
        visibility=[VisibilityLevel.TEAM],
    )
    _add_distractor(coscope, "task/s1/shared", "S1 distractor")

    requests = [
        coscope.create_request("planner_s1", "S1 alpha shared evidence"),
        coscope.create_request("solver_s1", "S1 alpha evidence"),
        coscope.create_request("critic_s1", "S1 shared evidence"),
    ]
    gold_map = {request.request_id: [gold.memory_id] for request in requests}
    return SyntheticCase("S1", "high_overlap", coscope, requests, gold_map)


def build_s2_partial_overlap() -> SyntheticCase:
    """S2: two agents share evidence, one requires private evidence."""
    coscope = _base_coscope()
    _register_team_agent(coscope, "planner_s2", "planner")
    _register_team_agent(coscope, "solver_s2", "solver")
    _register_team_agent(coscope, "private_s2", "solver")

    shared_gold = coscope.add_memory(
        "S2 shared alpha evidence for planner and solver.",
        "task/s2/shared",
        MemoryType.SEMANTIC,
        confidence=0.95,
        visibility=[VisibilityLevel.TEAM],
    )
    private_gold = coscope.add_memory(
        "S2 private beta evidence for the private solver.",
        "agent/private_s2/private",
        MemoryType.SEMANTIC,
        confidence=0.95,
        visibility=[VisibilityLevel.TEAM],
    )
    _add_distractor(coscope, "task/s2/shared", "S2 shared distractor")
    _add_distractor(coscope, "agent/private_s2/private", "S2 private distractor")

    private_scope = ScopeSpec(private_scopes=["agent/private_s2/private"])
    requests = [
        coscope.create_request("planner_s2", "S2 shared alpha evidence"),
        coscope.create_request("solver_s2", "S2 shared evidence"),
        coscope.create_request(
            "private_s2",
            "S2 private beta evidence",
            scope=private_scope,
            memory_types=[MemoryType.SEMANTIC],
        ),
    ]
    gold_map = {
        requests[0].request_id: [shared_gold.memory_id],
        requests[1].request_id: [shared_gold.memory_id],
        requests[2].request_id: [private_gold.memory_id],
    }
    return SyntheticCase("S2", "partial_overlap", coscope, requests, gold_map)


def build_s3_low_overlap() -> SyntheticCase:
    """S3: each agent has a separate scope, so sharing should be minimal."""
    coscope = _base_coscope()
    agents = [("planner_s3", "planner"), ("solver_s3", "solver"), ("critic_s3", "critic")]
    requests = []
    gold_map: Dict[str, List[str]] = {}

    for idx, (agent_id, role) in enumerate(agents, 1):
        scope_id = f"task/s3/{agent_id}/shared"
        _register_team_agent(coscope, agent_id, role, scopes=[scope_id])
        gold = coscope.add_memory(
            f"S3 gold evidence {idx} for {agent_id}.",
            scope_id,
            MemoryType.SEMANTIC,
            confidence=0.95,
            visibility=[VisibilityLevel.TEAM],
        )
        _add_distractor(coscope, scope_id, f"S3 distractor {idx}")
        request = coscope.create_request(agent_id, f"S3 gold evidence {idx}")
        requests.append(request)
        gold_map[request.request_id] = [gold.memory_id]

    return SyntheticCase("S3", "low_overlap", coscope, requests, gold_map)


def build_s4_policy_conflict() -> SyntheticCase:
    """S4: same scope/type but verifier policy must not be merged."""
    coscope = _base_coscope()
    team_policy = _team_policy()
    restricted_policy = PolicyConstraints(
        visibility=[VisibilityLevel.RESTRICTED],
        max_clearance=5,
        audit_required=True,
    )

    _register_team_agent(coscope, "planner_s4", "planner", policy=team_policy)
    _register_team_agent(coscope, "solver_s4", "solver", policy=team_policy)
    _register_team_agent(coscope, "verifier_s4", "verifier", policy=restricted_policy)

    shared_gold = coscope.add_memory(
        "S4 team-visible alpha evidence.",
        "task/s4/shared",
        MemoryType.SEMANTIC,
        confidence=0.95,
        visibility=[VisibilityLevel.TEAM],
    )
    restricted_gold = coscope.add_memory(
        "S4 restricted verifier audit evidence.",
        "task/s4/shared",
        MemoryType.SEMANTIC,
        confidence=0.95,
        visibility=[VisibilityLevel.RESTRICTED],
    )
    _add_distractor(coscope, "task/s4/shared", "S4 distractor")

    requests = [
        coscope.create_request("planner_s4", "S4 alpha evidence"),
        coscope.create_request("solver_s4", "S4 alpha evidence"),
        coscope.create_request("verifier_s4", "S4 restricted audit evidence"),
    ]
    gold_map = {
        requests[0].request_id: [shared_gold.memory_id],
        requests[1].request_id: [shared_gold.memory_id],
        requests[2].request_id: [restricted_gold.memory_id],
    }
    return SyntheticCase(
        "S4",
        "policy_conflict",
        coscope,
        requests,
        gold_map,
        conflict_request_ids=[requests[2].request_id],
    )


def _base_coscope() -> CoScope:
    coscope = CoScope()
    coscope.data.pipeline.config.fallback_threshold = 2
    return coscope


def _team_policy() -> PolicyConstraints:
    return PolicyConstraints(visibility=[VisibilityLevel.TEAM], max_clearance=1)


def _register_team_agent(
    coscope: CoScope,
    agent_id: str,
    role: str,
    *,
    scopes: Optional[Sequence[str]] = None,
    policy: Optional[PolicyConstraints] = None,
) -> None:
    coscope.create_agent(
        agent_id=agent_id,
        role=role,
        allowed_scopes=list(scopes or [f"task/{agent_id.split('_')[-1]}/shared"]),
        allowed_memory_types=[MemoryType.SEMANTIC],
        policy=policy or _team_policy(),
    )


def _add_distractor(coscope: CoScope, scope_id: str, content: str) -> None:
    coscope.add_memory(
        content=content,
        scope_id=scope_id,
        memory_type=MemoryType.SEMANTIC,
        confidence=0.25,
        visibility=[VisibilityLevel.TEAM],
    )


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _fmt(value: Any, digits: int) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        str(value).ljust(widths[i]) for i, value in enumerate(headers)
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])
