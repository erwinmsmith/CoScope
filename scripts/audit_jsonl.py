"""
Systematic schema audit for CoScope JSONL shards.

Runs 20+ invariant checks covering:
  A. Top-level enum value casing
  B. hop_count / max-hop consistency
  C. policy_conflict ↔ POLICY_ISOLATED
  D. NodeType / GraphType casing
  E. memory_id uniqueness (per-ep + cross-ep)
  F. parent_artifact_ids resolvability
  G. task_shared_* visibility != owner
  H. Agent policy visibility/clearance validity
  I. RetrievalRequest scope shape
  J. Per-node slot completeness (PLANNER/SOLVER/VERIFIER)
  K. Artifact metadata completeness
  L. schema_version == 1.0.0
  M. produced_by_slot / produced_artifact_ids consistency
  N. topo_index is int
  O. required_clearance in 0..3
  P. AUDIT_REPORT ↔ POLICY_ISOLATED
  Q. ground_truth.memory_id resolvable
  R. created_at ISO-8601
  S. meta.rollout.llm present
  T. cross-shard (original_id, question, answer) consistency

Run:
    python -m scripts.audit_jsonl tests/tmp/processed/got/musique/dev
    python -m scripts.audit_jsonl data/processed/got
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple


GRAPH_TYPES = {"LINEAR", "FORK", "MERGE", "FORK_MERGE", "INDEPENDENT", "POLICY_ISOLATED"}
SUBSETS = {"S1", "S2", "S3", "S4"}
NODE_TYPES = {"PLANNER", "SOLVER", "VERIFIER"}
RP_TYPES = {"GoT", "CoT", "ToT"}
VIS = {"owner", "team", "session", "public", "restricted"}
ROLES = {"planner", "solver", "verifier", "custom"}
EXPECTED_SLOTS = {
    "PLANNER":  {"query_intent", "scratch", "plan"},
    "SOLVER":   {"query_intent", "scratch", "conclusion"},
    "VERIFIER": {"query_intent", "scratch", "audit_report"},
}
META_ARTIFACT_REQ = {
    "slot", "scope_layer", "source_node_id", "hop_index", "topo_index",
    "parent_artifact_ids", "required_clearance", "is_gold_evidence",
    "dataset", "episode_id",
}


def audit_episode(ep: dict, tag: str, seen_ids: set) -> Tuple[List[str], List[str]]:
    issues: List[str] = []
    warnings: List[str] = []
    I, W = issues.append, warnings.append

    # A. top-level enum values
    if ep.get("graph_type") not in GRAPH_TYPES:
        I(f"{tag}: graph_type={ep.get('graph_type')!r}")
    if ep.get("reasoning_path_type") not in RP_TYPES:
        I(f"{tag}: reasoning_path_type={ep.get('reasoning_path_type')!r}")
    if ep.get("rho_subset") not in SUBSETS:
        I(f"{tag}: rho_subset={ep.get('rho_subset')!r}")

    # B. hop_count
    solvers = [n for n in ep["got_graph"]["nodes"] if n["node_type"] == "SOLVER"]
    max_hop = max((n.get("hop_index") or 0 for n in solvers), default=0)
    if ep["graph_type"] == "LINEAR" and max_hop != ep["hop_count"]:
        W(f"{tag}: LINEAR hop_count={ep['hop_count']} but max solver hop={max_hop}")

    # C. policy_conflict ↔ POLICY_ISOLATED
    iso = ep["graph_type"] == "POLICY_ISOLATED"
    if iso != bool(ep.get("policy_conflict")):
        I(f"{tag}: policy_conflict={ep.get('policy_conflict')} but graph_type={ep['graph_type']}")
    if iso and not ep.get("s4_eligible"):
        W(f"{tag}: POLICY_ISOLATED should be s4_eligible=True")

    # D. NodeType casing
    for n in ep["got_graph"]["nodes"]:
        if n["node_type"] not in NODE_TYPES:
            I(f"{tag}: node {n['node_id']} node_type={n['node_type']!r}")

    # E. memory_id uniqueness
    ids = [m["memory_id"] for m in ep["memory_entries"]]
    if len(ids) != len(set(ids)):
        dups = [x for x, c in Counter(ids).items() if c > 1]
        I(f"{tag}: duplicate memory_ids within episode: {dups[:3]}")
    for mid in ids:
        if mid in seen_ids:
            I(f"{tag}: memory_id {mid} collides cross-shard")
        seen_ids.add(mid)
    known = set(ids)

    # F. parent_artifact_ids resolvable
    for m in ep["memory_entries"]:
        md = m.get("metadata") or {}
        for pid in md.get("parent_artifact_ids") or []:
            if pid and pid not in known:
                I(f"{tag}: {m['memory_id']} dangling parent_artifact_id={pid}")

    # G. task_shared_* visibility should not be 'owner'
    for m in ep["memory_entries"]:
        md = m.get("metadata") or {}
        layer = md.get("scope_layer", "")
        vis = m.get("visibility")
        if layer.startswith("task_shared_") and vis == "owner":
            I(f"{tag}: task_shared entry {m['memory_id']} has owner visibility")

    # H. Agent policy
    for a in ep["agents"]:
        pol = a.get("policy") or {}
        vis_list = pol.get("visibility", [])
        if not isinstance(vis_list, list):
            I(f"{tag}: agent {a.get('agent_id')} policy.visibility not list")
        else:
            for v in vis_list:
                if v not in VIS:
                    I(f"{tag}: agent {a.get('agent_id')} policy visibility {v!r}")
        if "max_clearance" not in pol:
            I(f"{tag}: agent {a.get('agent_id')} policy missing max_clearance")
        if a.get("role") not in ROLES:
            W(f"{tag}: agent role={a.get('role')!r}")

    # I. RetrievalRequest scope shape
    for rr in ep["retrieval_requests"]:
        if not rr.get("query"):
            W(f"{tag}: retrieval_request {rr.get('request_id')} empty query")
        scope = rr.get("scope") or {}
        for k in ("private_scopes", "shared_scopes", "workspace_scopes", "governed_scopes"):
            if k not in scope:
                I(f"{tag}: retrieval_request scope missing {k}")

    # J. per-node slot completeness
    slot_by_node: dict = defaultdict(set)
    for m in ep["memory_entries"]:
        md = m.get("metadata") or {}
        s, nid = md.get("slot"), md.get("source_node_id")
        if s and nid:
            slot_by_node[nid].add(s)
    for n in ep["got_graph"]["nodes"]:
        expected = EXPECTED_SLOTS.get(n["node_type"], set())
        got = slot_by_node.get(n["node_id"], set())
        if got != expected:
            I(f"{tag}: node {n['node_id']} ({n['node_type']}) slots={got} expected={expected}")

    # K. artifact metadata completeness
    for m in ep["memory_entries"]:
        md = m.get("metadata") or {}
        if md.get("slot"):
            miss = META_ARTIFACT_REQ - set(md.keys())
            if miss:
                I(f"{tag}: {m['memory_id']} slot={md['slot']} missing metadata {miss}")
                break

    # L. schema_version
    ver = ep["meta"].get("schema_version")
    if ver != "1.0.0":
        I(f"{tag}: schema_version={ver!r} expected '1.0.0'")

    # M. produced_by_slot / produced_artifact_ids
    for n in ep["got_graph"]["nodes"]:
        pbs = n.get("produced_by_slot") or {}
        pia = set(n.get("produced_artifact_ids") or [])
        if set(pbs.values()) != pia:
            I(f"{tag}: node {n['node_id']} produced_by_slot disagrees with produced_artifact_ids")
        for mid in pia:
            if mid not in known:
                I(f"{tag}: node {n['node_id']} produced id {mid} not in episode")

    # N. topo_index int
    for m in ep["memory_entries"]:
        md = m.get("metadata") or {}
        if "topo_index" in md and not isinstance(md["topo_index"], int):
            I(f"{tag}: {m['memory_id']} topo_index {md['topo_index']!r} not int")
            break

    # O. required_clearance 0..3
    for m in ep["memory_entries"]:
        md = m.get("metadata") or {}
        rc = md.get("required_clearance")
        if rc is not None and rc not in (0, 1, 2, 3):
            I(f"{tag}: {m['memory_id']} required_clearance={rc}")

    # P. AUDIT_REPORT ↔ POLICY_ISOLATED
    has_audit = any((m.get("metadata") or {}).get("slot") == "audit_report"
                    for m in ep["memory_entries"])
    if has_audit != iso:
        I(f"{tag}: AUDIT_REPORT presence={has_audit} but graph={ep['graph_type']}")

    # Q. ground_truth.memory_id resolvable
    for g in ep.get("ground_truth", []):
        if g.get("memory_id") not in known:
            I(f"{tag}: ground_truth memory_id {g.get('memory_id')} not in episode")

    # R. created_at ISO-8601
    ts = ep["meta"].get("created_at", "")
    if ts and "T" not in ts:
        I(f"{tag}: created_at not ISO-8601: {ts!r}")

    # S. meta.rollout.llm
    if not (ep["meta"].get("rollout") or {}).get("llm"):
        I(f"{tag}: meta.rollout.llm missing")

    return issues, warnings


def audit_directory(root: Path) -> int:
    shards = sorted(root.rglob("*.jsonl"))
    if not shards:
        print(f"No .jsonl files under {root}")
        return 1

    all_issues: List[str] = []
    all_warnings: List[str] = []
    seen_ids: set = set()
    qa_by_oid: dict = defaultdict(set)
    n_episodes = 0

    for shard in shards:
        for line_no, line in enumerate(shard.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            ep = json.loads(line)
            tag = f"{shard.name}:L{line_no}"
            issues, warnings = audit_episode(ep, tag, seen_ids)
            all_issues.extend(issues)
            all_warnings.extend(warnings)
            qa_by_oid[ep["original_id"]].add((ep["question"], ep["answer"]))
            n_episodes += 1

    for oid, pairs in qa_by_oid.items():
        if len(pairs) > 1:
            all_issues.append(
                f"cross-shard: original_id={oid} has divergent (question,answer): {len(pairs)} variants"
            )

    print("=" * 75)
    print(f"{len(all_issues):4d} issues   {len(all_warnings):4d} warnings   "
          f"{n_episodes} episodes   {len(shards)} shards")
    print("=" * 75)
    for x in all_issues:
        print(f"  [X] {x}")
    for x in all_warnings:
        print(f"  [!] {x}")
    if not all_issues and not all_warnings:
        print("  ALL CLEAN")
    return 0 if not all_issues else 2


def main():
    p = argparse.ArgumentParser(description="Audit CoScope JSONL shards against schema v1.0.0")
    p.add_argument("root", help="Directory containing .jsonl shards (recursive).")
    args = p.parse_args()
    return audit_directory(Path(args.root))


if __name__ == "__main__":
    sys.exit(main() or 0)
