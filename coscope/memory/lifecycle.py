"""Validated artifact/memory lifecycle transitions."""

from typing import ClassVar

from coscope.core.artifact import ArtifactState


class LifecycleManager:
    _allowed: ClassVar[dict[ArtifactState, set[ArtifactState]]] = {
        ArtifactState.DRAFT: {
            ArtifactState.PROPOSED,
            ArtifactState.QUARANTINED,
            ArtifactState.REVOKED,
        },
        ArtifactState.PROPOSED: {
            ArtifactState.VERIFIED,
            ArtifactState.QUARANTINED,
            ArtifactState.REVOKED,
        },
        ArtifactState.VERIFIED: {
            ArtifactState.COMMITTED,
            ArtifactState.QUARANTINED,
            ArtifactState.REVOKED,
        },
        ArtifactState.COMMITTED: {
            ArtifactState.EXPIRED,
            ArtifactState.REVOKED,
        },
        ArtifactState.QUARANTINED: {
            ArtifactState.VERIFIED,
            ArtifactState.REVOKED,
        },
        ArtifactState.EXPIRED: set(),
        ArtifactState.REVOKED: set(),
    }

    def transition(self, current: ArtifactState, target: ArtifactState) -> ArtifactState:
        if target not in self._allowed[current]:
            raise ValueError(f"invalid lifecycle transition: {current.value} -> {target.value}")
        return target
