# CoScope 机制串讲（A4 主路径 + S1–S4 划分）

> 背景：A5–A8 因实验效果不佳已决定删去。本文聚焦 **A4** 这条「分桶 + mean query + 个体重排 + private fallback」主路径的代码层面机制，并补齐 dataset → S1/S2/S3/S4 的归类逻辑。文中所有引用都指向 `CoScope_unified/` 仓库。

---

## 1. Batch query：n 个 agent 的 query 是不是 mean 后再去 public？

### 1.1 结论

在 **A4** 里，所谓 "batch query" 就是：bucket 内 `n` 个 agent 的 query embedding **直接做算术平均（无加权）+ L2 归一化**，得到 **1 条** 平均向量，这条平均向量再去 memory store 做 **1 次** first-stage 检索。检索到的 candidate pool 由 bucket 内所有 agent **共享**（之后才进入个体重排）。

> 注意：A4 没有 query matrix `Q ∈ R^(d×n)` 和 SVD —— 那是被删除的 A5。A4 的 "batch" 体现在「N agent 共享一次检索」，而 query 端的处理就是简单 mean。

### 1.2 代码定位

入口在 [retrieval/pipeline.py:205-211](retrieval/pipeline.py#L205-L211)，进入 `_retrieve_mean_bucket`：

```python
elif variant == "a4":
    bucket_results = self._retrieve_mean_bucket(
        bucket,
        enable_rerank=self.config.enable_rerank,
        enable_fallback=self.config.enable_private_fallback,
        metadata_mode="a4_shared_mean",
    )
```

**mean 的计算位置** —— [retrieval/pipeline.py:415-417](retrieval/pipeline.py#L415-L417)：

```python
embeddings = [self.embedding_provider.embed_query(r.query) for r in bucket.requests]
query_embedding = np.mean(np.vstack(embeddings), axis=0)   # ← 这里就是「n 条取均值」
query_embedding = self._l2_normalize(query_embedding)
```

`bucket.requests` 就是经 `_route_scope_only` 划进同一个 scope bucket 的所有 agent 请求。每个 request 走 `embed_query` 拿到 1 条向量，`np.vstack` 成 `n × d` 矩阵后沿 `axis=0` 取均值，得到一条 `d` 维向量；再 L2 归一化。**没有加权、没有 SVD、没有方向选择**，就是朴素均值。

### 1.3 平均向量去哪个 store 做检索？

不是单纯的 "public"，而是 bucket 可访问的 **shared scope ∪ workspace scope 交集**。代码在 [retrieval/pipeline.py:419-426](retrieval/pipeline.py#L419-L426)：

```python
accessible = self._bucket_accessible_scopes(bucket)
candidates = self.memory_store.search(
    query_embedding=query_embedding,
    scope_filter=accessible or None,       # ← 限制在「桶内所有 agent 都有权访问」的 scope
    memory_type_filter=bucket.memory_types or None,
    policy_filter=bucket.merged_policy,
    top_k=self.config.shared_top_k,
)
self._stats["first_stage_retrievals"] += 1
```

`_bucket_accessible_scopes` 的实现在 [retrieval/pipeline.py:639-660](retrieval/pipeline.py#L639-L660)，关键逻辑：

```python
scopes = list(bucket.shared_scopes)                         # task/{ep}/shared 等
workspace_sets = [set(r.scope.workspace_scopes) for r in bucket.requests]
shared_ws = set.intersection(*workspace_sets)               # ← 工作区取交集
for ws in sorted(shared_ws):
    if ws not in scopes:
        scopes.append(ws)
return scopes
```

也就是说：

- **`bucket.shared_scopes`**（如 `task/{ep}/shared`）—— bucket 路由时声明的共享 scope；
- **桶内每个 request 的 `workspace_scopes` 取 set 交集** —— 这部分是「桶内每个 agent 都有权读」的工作区 / 知识库 scope，所以可以安全地纳入共享检索。

`private_scopes` 不在这里，**完全不参与 shared 阶段**——这是 A4 私有保护的第一层。

### 1.4 一图概括（A4 batch query 段）

```
n 条 agent query
    │ embed_query (×n)
    ▼
Q = [q1; q2; ...; qn]  ∈ R^(n×d)   ← 仅在内存里短暂存在
    │ np.mean(axis=0) + L2 normalize
    ▼
q_mean ∈ R^d   ← 单条向量
    │ memory_store.search(q_mean,
    │     scope_filter = shared ∪ ⋂ workspace,
    │     policy_filter = bucket.merged_policy,
    │     top_k = shared_top_k)
    ▼
1 个共享 candidate pool（pool_size ≤ shared_top_k）
    │ 所有 bucket 内 agent 共用这个 pool
    ▼
进入 §2 描述的个体重排 + private fallback
```

---

## 2. A4 完整机制：batch query 之后发生了什么 + private/public 的判定

A4 入口仍然是 `RetrievalPipeline.retrieve(requests, variant="a4")`，整体顺序如下，全程对照 [retrieval/pipeline.py](retrieval/pipeline.py)：

### Stage 0 — Variant 归一化
[pipeline.py:127, 902-930](retrieval/pipeline.py#L902-L930) 把任意别名（如 `shared_mean`、`a4_no_fallback`）规范成 `"a4"`。

### Stage 1 — Scope-only 路由
[pipeline.py:187-191](retrieval/pipeline.py#L187-L191)：

```python
scope_only_routing = variant in {
    "a3", "a4", "a4_nofb", "a4_norerank", ...
}

routing = (
    self._route_scope_only(requests)
    if scope_only_routing
    else self.router.route(requests)
)
```

进入 [pipeline.py:676-722](retrieval/pipeline.py#L676-L722) `_route_scope_only`：按每个 request 的 `request.scope.primary_scope`（典型值 `task/{ep}/shared`）分组。

- 同一 primary_scope 且 `len(group) >= 2` → 形成 `RetrievalBucket`，标记 `is_shareable=True`。
- 单元素分组 → 进 `independent_requests`，后续走 `_retrieve_independent`（仍开 fallback）。

A4 **不做 overlap-aware / policy-merge** 路由（那是 default `HierarchicalRouter` 的事），策略元数据 `{"strategy": "scope_only"}`。

### Stage 2 — Bucket 内 mean query → 共享 first-stage（已在 §1 详述）

调用 `_retrieve_mean_bucket(bucket, enable_rerank=True, enable_fallback=True)`。

### Stage 3 — 个体重排（per-agent personalization）

[pipeline.py:432-439](retrieval/pipeline.py#L432-L439)：

```python
for request in bucket.requests:
    shared = self._personalize(
        request=request,
        memories=memories,
        score_by_id=score_by_id,
        source="shared",
        enable_rerank=enable_rerank,   # A4=True
    )
```

`_personalize` 在 [pipeline.py:566-616](retrieval/pipeline.py#L566-L616)，`enable_rerank=True` 时调用 `RoleAwareReranker.rerank`：以 `request.role`（planner/solver/verifier）和 `request.state` 为条件，对**同一个 shared pool** 重新打分，再与 first-stage 得分 50/50 融合：

```python
score = 0.5 * float(base_score) + 0.5 * float(item.reranked_score)
```

排序后取 `rerank_top_k`。这一步是 A4 区别于 A3 的两个开关之一 —— 让每个 agent 从同一池子里挑出**自己最需要**的 top-k。

### Stage 4 — Private fallback（私域兜底）

[pipeline.py:440-444](retrieval/pipeline.py#L440-L444)：

```python
fallback = self._fallback(request, shared) if enable_fallback else self._no_fallback(...)
if fallback.triggered:
    self._stats["fallback_triggers"] += 1
```

`_fallback` 在 [pipeline.py:618-632](retrieval/pipeline.py#L618-L632) 调 `PrivateFallbackRetriever.retrieve`。**触发条件 + 检索逻辑** 都在 [retrieval/fallback/strategy.py](retrieval/fallback/strategy.py)：

- **触发器** = `CombinedTrigger(min_count=fallback_threshold, min_score=0.3, top_k=5)`（[fallback/strategy.py:56-77](retrieval/fallback/strategy.py#L56-L77)）。两个子条件 OR：
  - `len(shared_candidates) < threshold`（默认 5）；
  - `top-5 平均分 < 0.3`。
- **触发后** 走 `_fetch_private_candidates`（[fallback/strategy.py:119-151](retrieval/fallback/strategy.py#L119-L151)）：用 **原始 query**（不是 mean！）embed，遍历 `request.scope.private_scopes` × `memory_types`，逐个 scope 调 `memory_store.search`，结果打 `source="private"` 后去重。

### Stage 5 — RRF 融合

[pipeline.py:445](retrieval/pipeline.py#L445)：

```python
fusion = self.fusion.fuse(shared, fallback.candidates, top_k=self.config.rerank_top_k)
```

`ReciprocalRankFusion` 把 shared 列表（rerank 后）和 fallback 列表（如有）按 RRF 公式合并，最终 top-k 写入 `RetrievalResult.candidates`。

### Stage 6 — Independent 请求兜底

不在 bucket 里的请求进 [pipeline.py:238-247](retrieval/pipeline.py#L238-L247) 的循环：

```python
for request in routing.independent_requests:
    result = self._retrieve_independent(
        request,
        use_fallback=variant not in {"a3", "a4_nofb"},   # A4 True
        metadata_mode=f"{variant}_independent",
        enable_rerank=(False if variant in {"a4_norerank", "a5_norerank"} else None),
    )
```

走的是 [pipeline.py:466-509](retrieval/pipeline.py#L466-L509) `_retrieve_independent`：单 agent 自己 embed → 自己的 `scope.all_scopes` 全集 → `_personalize` → 同样可触发 private fallback → fusion。

### 2.1 「private vs public」是怎么判定的？

**没有名为 `is_private` 的 flag**，private/public 是通过 **scope 三段式 + 调用路径** 隐式区分的，关键在 [core/types.py:191-228](core/types.py#L191-L228) 的 `ScopeSpec`：

| 字段 | 语义 | 在 A4 中怎么被消费 |
|---|---|---|
| `private_scopes` | `agent/{id}/private`、`task/{ep}/{node}/private` 等 | **绝不进入 Stage 2 的 mean / shared search**；只在 Stage 4 fallback 命中后被 `_fetch_private_candidates` 单独遍历 |
| `shared_scopes` | `task/{ep}/shared` 等 | 是 `bucket.shared_scopes` 的来源；Stage 2 默认带入 |
| `workspace_scopes` | 知识库 / 共同语料 | 在 `_bucket_accessible_scopes` 里 **取桶内交集** 后加入 Stage 2 |
| `governed_scopes` | `task/{ep}/restricted` 等受控区 | 由 `policy_filter`（`bucket.merged_policy`）控制；S4 / Verifier 才会有 clearance |

判定的具体证据：

1. **共享段过滤器**（[pipeline.py:419-426](retrieval/pipeline.py#L419-L426)）：`scope_filter=accessible`，而 `accessible = bucket.shared_scopes ∪ ⋂ workspace_scopes`。`private_scopes` 不在这里。
2. **私有段单独检索**（[fallback/strategy.py:131-140](retrieval/fallback/strategy.py#L131-L140)）：

```python
for scope_id in request.scope.private_scopes:    # ← 显式只用 private 段
    for mem_type in request.memory_types:
        candidates = self.memory_store.search(
            query_embedding=query_embedding,
            scope_filter=[scope_id],
            memory_type_filter=[mem_type],
            policy_filter=request.policy,
            top_k=self.max_candidates,
        )
```

3. **Source 标签**（[fallback/strategy.py:148](retrieval/fallback/strategy.py#L148)）：
   - shared 阶段产出的 candidate `source="shared"`；
   - fallback 私域命中的 `source="private"`；
   - independent 阶段（独立请求） `source="independent"`；
   - A2 强合并 `source="a2_force_merge"`（A4 不会出现）。

这些 `source` 字段会落到 `RetrievalResult.shared_candidates` / `private_candidates`，方便在评测层（FMR、Evidence Hit Rate）按来源分别统计。

### 2.2 A4 一图流

```
requests (n agents)
    │
    ├── _route_scope_only ──────────────────┐
    │       ↓                               │
    │   shareable_buckets[]            independent_requests[]
    │       │                               │
    │       ▼ for each bucket               ▼
    │   ┌──────────────────────┐        ┌─────────────────────────┐
    │   │ Stage 2  mean query  │        │ _retrieve_independent   │
    │   │   q_mean = mean(qi)  │        │  • single embed         │
    │   │   shared_pool = M.   │        │  • scope.all_scopes     │
    │   │     search(q_mean,   │        │  • rerank + fallback    │
    │   │       shared∪∩ws,    │        │  • RRF fusion           │
    │   │       policy)        │        └─────────────────────────┘
    │   └─────────┬────────────┘
    │             ▼ for each request in bucket
    │   ┌──────────────────────────────┐
    │   │ Stage 3  RoleAwareReranker   │
    │   │   score = 0.5*base+0.5*rr    │
    │   │   → rerank_top_k             │
    │   └─────────┬────────────────────┘
    │             ▼
    │   ┌──────────────────────────────┐
    │   │ Stage 4  CombinedTrigger     │
    │   │   if count<5 or score<0.3 → │
    │   │     for s in private_scopes: │
    │   │       M.search(s, query, …)  │
    │   └─────────┬────────────────────┘
    │             ▼
    │   ┌──────────────────────────────┐
    │   │ Stage 5  ReciprocalRankFusion│
    │   │   fuse(shared, fallback)     │
    │   └─────────┬────────────────────┘
    ▼             ▼
   _order_results(...)  → List[RetrievalResult]
```

---

## 3. dataset → S1 / S2 / S3 / S4 是怎么打标的？

S1/S2/S3 与 S4 是 **两个正交的标签维度**：

- **S1/S2/S3**：互斥，由 ρ（Solver 间记忆访问 Jaccard 重叠率）阈值决定。
- **S4**：与 S1/S2/S3 正交，凡 `graph_type == POLICY_ISOLATED` 的 episode 都同时是 `s4_eligible=True`，**仅在 test split 出现**。

### 3.1 流水线总图

```
raw 数据集（MuSiQue / 2Wiki / HotpotQA / GSM8K / MATH）
    │ dataio/loaders/*.py
    ▼
raw_item 字典（统一 schema）
    │ construction/episode_builder.py
    │   1. graph_builder.build → GoTGraph (graph_type ∈ {LINEAR, FORK, MERGE,
    │                                       FORK_MERGE, INDEPENDENT, POLICY_ISOLATED})
    │   2. 多个 builder 构造 workspace / task_shared / private / restricted memory
    │   3. RhoCalculator.compute → ρ ∈ [0, 1]
    │   4. SubsetAssigner.assign(ρ, got_graph)
    ▼
Episode (rho, rho_subset ∈ {S1,S2,S3}, policy_conflict, s4_eligible)
    │ construction/dataset_pipeline.py:_write_shards
    ▼
{processed}/{path_type}/{dataset}/{split}/{subset}_{graph}.jsonl
    例：musique/test/s2_fork.jsonl, hotpotqa/test/s4_policy_isolated.jsonl
```

### 3.2 ρ → S1 / S2 / S3 的阈值

[evaluation/split/subset_assigner.py:21-58](evaluation/split/subset_assigner.py#L21-L58)，`SubsetThresholds` 默认值：

```python
s1_min: float = 0.6
s2_min: float = 0.4
```

判定逻辑在 `SubsetAssigner._rho_label`（[subset_assigner.py:105-111](evaluation/split/subset_assigner.py#L105-L111)）：

```python
if rho > t.s1_min:        # rho > 0.6
    return SubsetLabel.S1
if rho > t.s2_min:        # 0.4 < rho <= 0.6
    return SubsetLabel.S2
return SubsetLabel.S3     # rho <= 0.4
```

阈值可通过 `config/yaml/subset_thresholds.yaml` 覆盖（[subset_assigner.py:36-55](evaluation/split/subset_assigner.py#L36-L55)），约束 `0 <= s2_min <= s1_min <= 1`。

### 3.3 ρ 是怎么算的？

[graph/got/rho_calculator.py:64-99](graph/got/rho_calculator.py#L64-L99)。**只取 Solver 节点参与**（Planner / Verifier 排除）：

1. 对每个 Solver-k 节点，构造 ancestor-or-self 集合 `A_k`；
2. 计算其可访问 memory id 集合（[rho_calculator.py:156+](graph/got/rho_calculator.py#L156)）：
   - `workspace_semantic_hop` —— 仅当 `hop_index == k` 才计入；
   - `task_shared_episodic` —— 仅当 `source_node_id ∈ A_k` 才计入；
   - `workspace_semantic_global` / `agent_private` / `restricted` 不计入；
3. 对所有 Solver 集合**两两 IoU 平均**：

```python
ious = []
for a, b in combinations(accessible_sets, 2):
    union = a | b
    if not union:
        continue
    ious.append(len(a & b) / len(union))
return round(sum(ious) / len(ious), 4)
```

退化情况：单 Solver → ρ = 1.0（非空集）/ 0.0（空集）；无 Solver → ρ = 0.0。

> 之所以用 pairwise mean IoU 而不是严格全集 Jaccard：在 FORK / FORK_MERGE / INDEPENDENT 图里，任意两个兄弟 Solver 的 ancestors 互不相交会让全集 Jaccard 直接坍缩到 0，分布只剩 {0, 0.5}，无法横跨三个 ρ 桶（[rho_calculator.py:85-90](graph/got/rho_calculator.py#L85-L90) 注释）。

### 3.4 S4 的判定 —— 与 ρ 无关

[evaluation/split/subset_assigner.py:94-118](evaluation/split/subset_assigner.py#L94-L118)：

```python
def assign(self, rho, got_graph) -> SubsetAssignment:
    rho_subset      = self._rho_label(rho)              # 仍给 S1/S2/S3 标签
    policy_conflict = self._is_policy_isolated(got_graph)
    return SubsetAssignment(
        rho_subset=rho_subset,
        policy_conflict=policy_conflict,
        s4_eligible=policy_conflict,                    # ← S4 = POLICY_ISOLATED
    )
```

`_is_policy_isolated` 直接看 `got_graph.graph_type == GraphType.POLICY_ISOLATED`（[core/types.py:760-768](core/types.py#L760-L768) 列出了 6 种 graph_type）。

### 3.5 怎么落到 JSONL：S4 的 test-only 写保护

[construction/dataset_pipeline.py:199-218](construction/dataset_pipeline.py#L199-L218) `_write_shards`：

```python
for ep in episodes:
    if ep.s4_eligible and split != "test":
        rejected += 1
        logger.error("Dropping s4_eligible episode %s placed in split=%s (must be test)", ...)
        continue
    subset_key = "s4" if ep.s4_eligible else ep.rho_subset.value.lower()
    graph_key  = self._graph_shard_key(ep.graph_type)
    shard_name = f"{subset_key}_{graph_key}.jsonl"
    shard_episodes[shard_name].append(ep)
```

要点：

1. `s4_eligible` 的 episode 只允许进 `test` split；其它 split 直接丢并打 error。
2. **shard 命名优先用 S4**：一旦 `s4_eligible=True`，文件名就是 `s4_*.jsonl`，原本的 `rho_subset`（S1/S2/S3）只保留在字段里、不进文件名。所以 S4 episode **不会同时出现在 `s2_*.jsonl` 等分片里**。
3. 之前在 [dataset_pipeline.py:104-113](construction/dataset_pipeline.py#L104-L113) 还有 `kept` 过滤 + `enforce_coverage` 检查：默认 test split 必须至少有一条 S4，其它 split 严禁出现 S4。

### 3.6 一张表对照

| 标签 | 决定者 | 阈值 / 条件 | 互斥关系 | 出现位置 |
|---|---|---|---|---|
| **S1** | ρ | ρ > 0.6 | S1/S2/S3 互斥 | train / dev / test 都有 |
| **S2** | ρ | 0.4 < ρ ≤ 0.6 | 同上 | 同上 |
| **S3** | ρ | ρ ≤ 0.4 | 同上 | 同上 |
| **S4** | graph_type | `POLICY_ISOLATED` | 与 S1/S2/S3 **正交**（独占 shard 名） | **仅 test** |

> 为什么 S4 要 test-only：S4 被设计来量化「无差别共享的安全风险」（FMR），训练 / dev 阶段模型不应见到 verifier-only 的受限内容，否则数据本身就泄露。这条约束由 `_write_shards` 强制保证。

---

## 4. 接下来删 A5–A8 的影响面（提示）

如果现在要清掉 A5–A8，至少以下几处需要同步处理（按依赖优先级）：

| 位置 | 处理建议 |
|---|---|
| [retrieval/pipeline.py:127-249](retrieval/pipeline.py#L127-L249) | 删 `a5/a5_*/a6/a7/a8` 分支；`PipelineConfig.variant` 默认改为 `"a4"` |
| [retrieval/pipeline.py:902-930](retrieval/pipeline.py#L902-L930) `_normalize_variant` | 删 `query_matrix_*` / `svd` / `doc_random_*` 等别名 |
| [retrieval/pipeline.py:275-406](retrieval/pipeline.py#L275-L406) `_retrieve_svd_bucket` | 整段删除；`_svd_projection` / `_dump_svd_bucket` 同删 |
| [retrieval/matrix/](retrieval/matrix/) + [retrieval/projection/](retrieval/projection/) | 整个子模块可移除（仅 SVD 路径使用）|
| [retrieval/router/hierarchical.py](retrieval/router/hierarchical.py) | A6 的 block routing 入口 —— 若不再用可删，但 default router 链路要切回 `OverlapAwareRouter` 或 `_route_scope_only` |
| `PipelineConfig` 字段 `query_embedding_dim` / `projection_dim` / `svd_rank` / `use_sparse_mask` / `dump_svd_artifacts` | 删 |
| `__init__` 中 `self.matrix_builder` / `self.projection` 实例化 | 删 |
| [introduction.md §14.4.1](introduction.md) 与 [coscope流程.md](coscope流程.md) 的变体表 | 同步精简 |
| `examples/evaluate_variants.py`、`evaluation/runner.py` | 检查是否硬编码了 a5/a6/a7/a8 |
| `data/processed/**/s4_*.jsonl` | S4 与 variant 解耦，不受影响；保留 |

S1/S2/S3/S4 的划分逻辑全在 `construction/` + `evaluation/split/`，**与 A1–A4 的检索分发完全解耦**，删 A5–A8 不会动这条数据线。
