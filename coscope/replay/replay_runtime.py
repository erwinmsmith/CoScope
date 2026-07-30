"""Replay prebuilt requests without invoking an LLM."""

from coscope.context import ContextPacket
from coscope.retrieval import RetrievalRequest, RetrievalResult
from coscope.runtime import CoScopeRuntime


class ReplayRuntime:
    def __init__(self, runtime: CoScopeRuntime):
        self.runtime = runtime

    def run(
        self, requests: list[RetrievalRequest]
    ) -> tuple[dict[str, RetrievalResult], dict[str, ContextPacket]]:
        results = self.runtime.retrieve_batch(requests)
        packets = {
            request.request_id: self.runtime.assemble_context(
                request, results[request.request_id]
            )
            for request in requests
        }
        return results, packets
