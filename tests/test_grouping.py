from math import cos, radians, sin

from coscope.retrieval import RequestGrouper, RetrievalRequest
from coscope.scope import EffectiveView


def _request(request_id: str, angle: float) -> RetrievalRequest:
    vector = (cos(radians(angle)), sin(radians(angle)))
    view = EffectiveView(
        scope_ids=frozenset({"public"}),
        public_scope_ids=frozenset({"public"}),
        private_scope_ids=frozenset(),
        tenant_id="t",
        workspace_id="w",
        policy_versions=frozenset({"1"}),
        index_snapshots=frozenset({"s"}),
    )
    return RetrievalRequest(
        request_id=request_id,
        run_id="run",
        agent_id=request_id,
        reasoning_node_id="node",
        full_query=request_id,
        public_intent=request_id,
        private_intent=None,
        public_vector=vector,
        scope_signature="same",
        effective_view=view,
    )


def test_grouping_does_not_chain_transitively():
    # A-B and B-C are similar, while A-C violates the pairwise floor.
    groups = RequestGrouper(
        medoid_threshold=0.90,
        minimum_pairwise_similarity=0.85,
    ).group([_request("a", 0), _request("b", 23), _request("c", 46)])

    assert sorted(len(group.requests) for group in groups) == [1, 2]


def test_different_scope_signatures_never_share():
    first = _request("a", 0)
    second = _request("b", 0)
    second.scope_signature = "different"
    groups = RequestGrouper().group([first, second])
    assert len(groups) == 2
