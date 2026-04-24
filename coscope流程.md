

# MAS CoScope 协同检索流水线（结构化描述）

## 顶层入口

```
INPUT: 用户任务输入
```

---

## Phase 0：推理结构初始化（消融变量）

```python
reasoning_mode = one_of("GoT", "CoT", "ToT")
# reasoning_mode 决定：
#   - agent 数量
#   - scope 大小
#   - 影响 ρ（overlap rate）

agents = instantiate([
    Planner,   # 负责任务方向
    Solver_k,  # 负责当前步骤所需事实（可多个）
    Verifier,  # 负责结论校验
])

# 执行顺序：按拓扑序，每轮执行一批 agent
```

---

## Phase 1：每轮迭代（Round t）

> 以下 Step 1–6 在每轮 t 中对每个 agent 执行

---

### Step 1：硬依赖注入

```python
# 父节点结论直接作为 context 喂给当前 agent
agent.context = parent_node.conclusion
```

---

### Step 2：检索前推理 ⭐（新增模块）

```python
# 每个 agent 独立推理："我这一步需要查什么？"
if agent == Planner:
    pre_retrieval_thought = infer_task_direction()
elif agent == Solver_k:
    pre_retrieval_thought = infer_facts_needed_for_current_step()

agent.retrieval_intent = pre_retrieval_thought
# 目的：使后续 query 更精准，query matrix 方向差异更真实
```

---

### Step 3：生成 Retrieval Request

```python
retrieval_request = agent.generate({
    "query": agent.retrieval_intent,
    "scope": agent.scope,     # 检索范围（哪些区域）
    "policy": agent.policy,   # 检索策略
    "state": agent.state,     # 当前状态
})
```

---

### Step 4：CoScope 协同检索（核心验证对象）

```python
# 两大数据源
workspace_semantic = KnowledgeBase(embedding="Qwen")  # 知识库
memory_store = MemoryStore(layers=3)                   # 三层 Memory，按权限检索各区域

# 检索流水线
results = pipeline(
    scope_bucket(retrieval_request),          # scope 分桶
    → build_query_matrix(agents),             # 构造 query matrix
    → shared_rerank(cross_agent=True),        # 共享重排
    → private_fallback(per_agent=True),       # 私有回退
    → C_i_final                               # 最终候选集
)
```

**ρ 说明：**
```
ρ = 不同 agent 在 Step 4 中需要检索的内容重叠程度
    检索前推理（Step 2）使 query 更精准 → query matrix 方向差异更真实
```

---

### Step 5：LLM 推理产出

```python
output = LLM(
    model="Qwen-32B",
    input={
        "context": agent.context,          # Step 1 注入
        "retrieval_candidates": C_i_final, # Step 4 产出
    },
    output={
        "scratch": agent.scratch,          # 思维链内容（中间推理）
        "conclusion": agent.conclusion,    # 最终结论
    }
)
```

---

### Step 6：写回 Memory Store

```python
memory_store.write({
    "zone_1_scratch": {
        # 思维链内容
        "shared":  Planner.scratch,        # Planner 计划，team 可见
        "private": Solver_k.scratch,       # Solver 草稿，仅自己可见
    },
    "zone_2_conclusion": {
        # 结论，task 范围内共享
        "task_shared": agent.conclusion,
    },
    "zone_3_verifier": {
        # Verifier 记录
        "log": Verifier.record,
    }
})

# 下一轮 agent 按权限检索以上内容 → 回到 Step 2
```

---

## 终止条件 & 输出

```python
while not termination_condition():
    run_round(t)
    t += 1

OUTPUT: MAS 最终答案输出
```

---

## 变量速查表

| 变量 | 含义 |
|------|------|
| `ρ` | 不同 agent 在 Step 4 中检索内容的重叠程度（overlap rate） |
| `reasoning_mode` | GoT / CoT / ToT，决定 agent 数量与 scope 大小 |
| `scope` | 每个 agent 的检索范围 |
| `C_i_final` | agent i 的最终检索候选集 |
| `scratch` | agent 内部思维链，不直接对外暴露 |
| `conclusion` | agent 当前轮的输出结论，可被后续 agent 消费 |
| `memory_store` | 三层结构：Zone1 思维链 / Zone2 结论 / Zone3 Verifier 记录 |

---

## 数据流总览（单行伪代码）

```
用户输入
  → [GoT/CoT/ToT 决定结构]
  → instantiate(agents)
  → for t in rounds:
      for agent in topological_order(agents):
          context = parent.conclusion                        # Step 1
          intent  = agent.pre_retrieval_reason()            # Step 2 ⭐
          req     = agent.build_retrieval_request(intent)   # Step 3
          C_final = coscope_retrieve(req, workspace, memory)# Step 4
          out     = LLM(context, C_final) → scratch+concl  # Step 5
          memory_store.write(out)                           # Step 6
  → return MAS最终答案
```