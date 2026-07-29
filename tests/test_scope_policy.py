from coscope.core import AgentInstance, MemoryEntry
from coscope.scope import (
    EffectiveView,
    Permission,
    ScopeDescriptor,
    ScopeEngine,
    Visibility,
)


def test_effective_view_excludes_unauthorized_and_intersects_public_scopes():
    shared = ScopeDescriptor(
        frozenset({"docs"}),
        "run/task",
        Visibility.TEAM_SHARED,
    )
    private = ScopeDescriptor(
        frozenset({"docs"}),
        "run/agent/planner",
        Visibility.AGENT_PRIVATE,
        owner_id="planner",
    )
    restricted = ScopeDescriptor(
        frozenset({"docs"}),
        "run/audit",
        Visibility.RESTRICTED,
        permissions={"role:verifier": frozenset({Permission.READ})},
    )
    planner = AgentInstance(
        "planner", "planner", knowledge_permissions=frozenset({"docs"})
    )
    verifier = AgentInstance(
        "verifier", "verifier", knowledge_permissions=frozenset({"docs"})
    )
    engine = ScopeEngine()

    planner_view = engine.resolve(
        planner, [shared, private, restricted], runtime_region="run/agent/planner"
    )
    verifier_view = engine.resolve(
        verifier, [shared, private, restricted], runtime_region="run/audit"
    )

    assert planner_view.scope_ids == {shared.scope_id, private.scope_id}
    assert verifier_view.scope_ids == {shared.scope_id, restricted.scope_id}
    common = EffectiveView.shared_intersection([planner_view, verifier_view])
    assert common.scope_ids == {shared.scope_id}
    assert restricted.scope_id not in common.scope_ids


def test_branch_private_scope_is_region_isolated():
    agent = AgentInstance(
        "solver", "solver", knowledge_permissions=frozenset({"docs"})
    )
    branch_a = ScopeDescriptor(
        frozenset({"docs"}),
        "tot/branch_a",
        Visibility.BRANCH_PRIVATE,
        owner_id=agent.agent_id,
    )
    branch_b = ScopeDescriptor(
        frozenset({"docs"}),
        "tot/branch_b",
        Visibility.BRANCH_PRIVATE,
        owner_id=agent.agent_id,
    )
    view = ScopeEngine().resolve(
        agent, [branch_a, branch_b], runtime_region="tot/branch_a"
    )
    assert view.scope_ids == {branch_a.scope_id}


def test_scope_id_is_stable_and_content_independent():
    scope = ScopeDescriptor(
        frozenset({"docs"}),
        "run/task",
        Visibility.TEAM_SHARED,
    )
    first = MemoryEntry("one", scope, "test", "1")
    second = MemoryEntry("two", scope, "test", "2")
    assert first.scope.scope_id == second.scope.scope_id
