1. 文档目标
本文档用于描述一种面向多智能体系统（Multi-Agent System, MAS）的协同记忆检索机制。该机制的核心目标不是把所有 agent 的检索请求强行合并，也不是让每个 agent 永远独立检索，而是在 memory scope 存在交叉 的前提下，对共享部分执行协同的 first-stage retrieval，对私有部分保留 agent-specific 的补检索能力。
该机制尤其面向以下需求：
1. 多个 agent 围绕同一任务协作时，经常会访问相同或相近的共享记忆区域。
2. 多个 agent 独立检索会导致重复扫描、重复召回和重复排序。
3. 系统既希望降低 first-stage retrieval 成本，又不能牺牲各 agent 的个体需求和权限边界。
4. 系统需要一个可以自然接入 query matrix、batch retrieval、局部 block routing 的统一检索框架。
本文档将重点说明：
- 如何组织 multi-agent retrieval request
- 如何基于 scope overlap 执行协同检索
- 如何把 query matrix 作为共享检索层的核心表示
- query matrix 在系统中的完整数据流、维度变化与变换过程
- 如何结合共享候选池、个体重排和 private fallback 完成最终检索、

2. 设计目标
2.1 降低重复检索成本
多个 agent 往往会对同一 task-shared memory、同一 session memory 或同一批 shared artifact 发起相近检索。若每个 agent 都独立执行 retrieval，会出现大量重复工作。本机制希望通过共享 first-stage retrieval 降低这种重复成本。
2.2 提升共享证据复用率
同一任务下，不同 agent 往往需要参考同一批证据，只是关注角度不同。通过共享候选池，可以提升候选证据的复用率，使多个 agent 建立在更一致的证据基础上进行后续推理。
2.3 保留 agent 的个体差异
虽然多个 agent 可以共享 first-stage retrieval，但最终结果不能被完全合并。不同 agent 的角色、状态、权限和目标不同，因此系统必须支持 agent-specific rerank 和 private fallback。
2.4 为 MAS-RAG 提供统一检索中间层
该机制不仅适用于 memory retrieval，也可以作为 MAS-RAG 中的共享检索中间层。也就是说，query matrix 不仅可以表示多个 agent 对 memory 的请求，也可以表示多个 agent 对外部知识库、工具输出或 artifact 的联合检索请求。

3. 核心思想
3.1 从按 agent 检索转向按共享作用域检索
传统做法通常以 agent 为检索单位：每个 agent 单独编码 query，单独在自己的可访问记忆区域中检索，单独生成候选，再单独排序。这种方式实现简单，但在多智能体协作场景下会造成明显冗余。
本设计将检索单位从“agent”下沉到“请求访问的记忆作用域”。也就是说，系统不先问“这个请求属于哪个 agent”，而是先问：
- 这个请求要访问哪些 scope
- 它检索哪种 memory type
- 它受什么 policy 约束
- 它是否和其他 agent 的请求访问同一片共享记忆区域
如果多个请求访问同一主要 shared scope，检索相同或兼容的 memory type，且 policy 不冲突，则它们可以先共享一次 first-stage retrieval。
3.2 共享候选，不共享最终答案
本机制共享的是候选池，而不是最终答案。
对于多个具备 scope overlap 的 agent：
1. 系统先对其共享部分执行一次 batch retrieval。
2. 生成一个 shared candidate pool。
3. 每个 agent 再基于自己的 role、state、query 对共享候选进行个体化重排。
4. 若共享候选不足，再对私有 scope 执行 fallback retrieval。
这种方式兼顾了效率与精度：
- 共享 first-stage retrieval 节省成本
- 个体重排保留差异
- private fallback 保护 recall
4. 检索请求的统一表示
为了支持协同检索，系统首先需要将每个 agent 发出的请求表示为统一结构。
定义一个 retrieval request：
r_i = (q_i, role_i, scope_i, type_i, policy_i, state_i)
其中：
- q_i：第 i 个 agent 的原始 query
- role_i：agent 角色，例如 planner、solver、verifier、critic、memory manager
- scope_i：该 agent 本次检索允许访问的 scope 集合
- type_i：本次检索面向的 memory type
- policy_i：访问控制、可见性、隔离规则等约束
- state_i：agent 当前状态，例如任务节点、已知证据、工具上下文、推理分支等
进一步地，可以把决定共享检索可行性的部分抽象成：
Ω_i = (scope_i, type_i, policy_i)
其中 Ω_i 不直接决定最终排序，但决定：
- 该请求属于哪个共享检索桶
- 它是否可以与其他请求一起 batch
- 它应当访问哪一类 memory slice
- 
5. 记忆作用域的组织方式
5.1 memory scope 的基本层级
为了让 scope overlap 能够被直接判断，记忆系统需要显式维护 scope 层级。建议至少包括以下几类：
1. agent-private scope
仅当前 agent 自己可访问的记忆，例如：
- 私有中间推理
- 私有草稿
- 局部缓存
- 未共享的观察结果
2. task-shared scope
同一任务下多个 agent 都可访问的共享记忆，例如：
- 共享任务黑板
- 共享计划树
- 共享证据池
- 共享中间结论
3. session scope
当前 session 中产生的上下文，例如：
- 当前轮消息
- 当前轮工具调用
- 当前轮状态更新
- 当前会话事件流
4. workspace/project scope
跨 session 的长期共享知识，例如：
- 项目文档
- 长期共享 artifact
- 公共知识库
- 历史经验
5. governed/restricted scope
受更强治理和权限控制的区域，例如：
- 高权限记忆
- 审计区
- 隔离区
- 待验证记忆区
5.2 memory slice 的含义
从检索角度看，系统并不是在“全记忆空间”中盲目搜索，而是在一个或多个具体的 memory slice 上检索。
可以将一个 memory slice 表示为：
Slice = (scope, type, policy)
这表示：
- 检索发生在哪个作用域中
- 检索面对哪种 memory type
- 检索受到哪些策略约束
- 
6. Scope Overlap 的判定机制
6.1 判定原则
多个请求可以进入同一次协同检索，需要满足以下三项：
1. scope 是否重叠
如果多个 agent 的请求都访问同一个主要 shared scope，例如：
- 同一 task-shared memory
- 同一 session memory
- 同一 shared blackboard
- 同一 workspace/project 下的共享 artifact 区域
那么这些请求可以优先进入同一个共享检索流程。
2. memory type 是否一致或兼容
如果多个请求都在检索相同或兼容的 memory type，例如都在查：
- episodic memory
- artifact memory
- shared memory
- semantic memory
那么它们适合共享 first-stage retrieval。
3. policy 是否兼容
只有在访问控制和可见性规则一致时，多个请求才允许共享候选检索。例如：
- 都能访问 team-visible 记忆
- 都排除 quarantine 区域
- 都满足相同 clearance 级别
如果 policy 冲突，则不能放进同一个共享检索桶。
6.2 简化分桶规则
在工程实现中，系统可以按以下顺序处理：
1. 先按 scope 分桶
2. 在每个 scope bucket 内按 memory type 分开
3. 再按 policy 检查是否可以继续共享
因此，不需要复杂评分，只需使用清晰规则：
- 共享 scope 才可能 batch
- type 不兼容则拆开
- policy 冲突则拆开

7. 协同检索的总体流程
7.1 阶段一：请求收集与标准化
系统在一个调度周期内收集所有 agent 的 retrieval request，形成请求集合：
R = {r_1, r_2, ..., r_n}
每个请求都被标准化为统一结构，确保后续模块可以一致处理。
7.2 阶段二：基于 scope/type/policy 的规则分桶
系统对 R 中的请求执行规则分桶，识别哪些请求可以共享检索。例如：
- task-shared + artifact
- task-shared + episodic
- session + shared
- workspace + semantic
- private + artifact
这一阶段的目标不是求“最优分组”，而是识别：
- 哪些请求可以共享 first-stage retrieval
- 哪些请求必须独立处理
7.3 阶段三：共享候选召回
对于每一个可共享的 bucket，系统执行一次共享候选召回。这里是 query matrix 发挥作用的核心阶段。
7.4 阶段四：agent-specific rerank
共享候选池生成后，每个 agent 用自己的 role、state 和 query 对候选进行个体化重排。
7.5 阶段五：private fallback retrieval
若共享候选不足以满足 agent 自身需求，则再在 agent 的 private scope 或专属 scope 上进行补检索。

8. Query Matrix 机制详解
这一部分是本设计的核心。query matrix 是共享检索层的主要表示形式，用于将多个 agent 的 query 组织为一个统一的批处理对象，并在共享 memory slice 上执行联合检索。
8.1 为什么需要 query matrix
如果每个 agent 的 query 都单独处理，通常会产生如下流程：
1. 单独编码 query
2. 单独进入 retrieval index
3. 单独生成候选
4. 单独做 top-k
5. 单独 rerank
在多智能体协作场景下，这会带来大量重复操作。引入 query matrix 后，可以把同一共享 scope bucket 中的 query 一起组织、一起变换、一起进入共享 first-stage retrieval，从而降低重复检索成本。
8.2 query matrix 的语义
在本设计中，query matrix 不再表示普通样本 batch，而表示：
同一共享检索桶中多个 agent 的联合检索请求。
假设某个共享 bucket 中有 n_b 个请求，每个请求的 query embedding 维度为 k，则可以构造：
Q^(b) = [q_1, q_2, ..., q_(n_b)]
其中：
- q_i ∈ R^k 表示第 i 个 agent 的 query 向量
- Q^(b) 表示整个 bucket 的 query matrix
为了和原始表述一致，可以采用“列堆叠”形式：
Q ∈ R^(k × n)
这里：
- k 是 query embedding dimension
- n 是同一个共享 bucket 中的 query 数量
- 第 i 列 Q[:, i] 表示第 i 个 query 向量
这种写法与原始方案中的 Q 定义保持一致，并且后续便于和投影矩阵、低维表示和 block 构造对接。
8.3 输入到 query matrix 的数据流
从系统角度看，query matrix 的构造分为以下几步：
Step 1：原始请求输入
每个 agent 提供一个原始 query，可以是：
- 文本 query
- 结构化 query
- 带 role/state 约束的复合 query
例如：
- planner: 当前任务的关键约束是什么
- verifier: 当前证据的来源是什么
- solver: 当前步骤需要哪些事实支持
Step 2：query normalization
系统对原始 query 做标准化处理，例如：
- 清洗格式
- 补充任务上下文
- 引入 role prompt
- 附加 state 中的必要信息
得到标准化查询文本或结构化表达。
Step 3：query encoding
系统将标准化后的 query 编码为向量（这个地方的向量化follow传统的embedding方法）：
q_i ∈ R^k
这里的 k 是统一的 embedding 维度。
Step 4：按共享 bucket 收集 query
系统只把属于同一个共享 bucket 的 query 拼成同一个 query matrix。也就是说，query matrix 不是全局唯一的，而是按 bucket 动态生成。
Step 5：构造 query matrix
若某个 bucket 有 n 个 query，则把它们按列拼接得到：
Q = [q_1, q_2, ..., q_n] ∈ R^(k × n)
8.4 query matrix 的投影变换
你原始设计中引入了一个可学习投影矩阵 W，用于把原始 query 表示映射到更适合后续检索组织的低维空间。这里我们保留这一思想，并将其作为共享 first-stage retrieval 的核心变换。
设：
W ∈ R^(l × k)
其中：
- k 是原始 query embedding 维度
- l 是投影后的中间维度
原始设计中进一步引入了结构化稀疏 mask，对 W 的每一行进行尾部截断，得到有效投影矩阵。这里可以保留该思想，但在系统设计文档中，建议将其解释为：
- W 负责把共享 bucket 内多个 query 映射到一个可组织、可压缩、可路由的共享表示空间
- 稀疏化或截断结构用于控制检索空间的有效自由度
- 后续的 SVD/降秩操作用于提取共享检索主方向
8.5 带 mask 的投影矩阵
如果保留原始设计中的结构化稀疏形式，则可以写为：
W' = W ⊙ M
其中：
- M 是 mask matrix
- ⊙ 是逐元素乘法
- W' 是有效投影矩阵
mask 的作用是：
- 限制某些行的有效长度
- 形成结构化稀疏
- 让不同维度的投影具有不同有效支撑区域
8.6 SVD 与共享子空间提取
为了进一步降低表示冗余并得到更紧凑的共享检索方向，可以对 W' 做 SVD：
W' = U Σ V^T
再根据有效秩 r 提取共享子空间：
W_final = V[:, 1:r] ∈ R^(k × r)
这里的含义是：
- 原始 query 空间是 k 维
- 经过共享子空间提取后，只保留最主要的 r 个方向
- W_final 用于把多个 agent 的 query 投影到一个更紧凑的共享检索子空间中
8.7 query matrix 的核心变换
有了 Q 和 W_final 后，就可以得到共享检索表示：
Z = Q^T W_final
维度变化如下：
Q^T ∈ R^(n × k)
W_final ∈ R^(k × r)
Z ∈ R^(n × r)
这一步非常关键，其含义是：
- 原始每个 query 是 k 维
- 投影后每个 query 变为 r 维
- Z 的每一行表示一个 agent query 在共享检索子空间中的表示
- 整个 Z 则表示同一共享 bucket 中所有 query 的联合低维表示
8.8 Z 的系统意义
矩阵 Z 不是最终相似度矩阵，而是共享 first-stage retrieval 的中间表示层。它有三个作用：
1. 统一表示：把同一个 bucket 中的多 agent query 投影到同一共享子空间
2. 便于路由：后续可以按 `Z` 的结构进行局部 block 组织
3. 降低冗余：减少在原始高维 query 空间中的重复比较
8.9 从 Z 到局部 block
原始设计中希望根据 Z 的结构把 query 组织成局部块，再在块内进行匹配。对于 MAS memory retrieval，这一步可以重解释为：
- 把同一共享 bucket 中的 query，按照其在共享子空间中的分布组织成若干局部检索区域
- 每个 block 对应共享 scope 内的一个局部候选区域或局部语义区域
- block 的意义不再只是数学打包，而是共享检索空间中的局部路由结构
在实现上，可以采用较轻量的方式：
- 不一定要求严格求解复杂的 packing
- 可以仅把 Z 作为共享候选构造时的局部组织依据
- 更重要的是把 block 作为“局部候选路由”而非硬性隔离边界
8.10 query-key 匹配与共享候选池生成
在共享 bucket 中，系统会从对应的 memory slice 中取出 candidate keys。设：
K ∈ R^(m × r)
其中：
- m 是共享 memory slice 中候选 key 数量
- r 是共享子空间维度
此时可以计算相似度矩阵：
A = Z K^T
维度变化如下：
Z ∈ R^(n × r)
K^T ∈ R^(r × m)
A ∈ R^(n × m)
这里：
- A[i, j] 表示第 i 个 agent query 与第 j 个候选 key 的匹配分数
- 第 i 行表示第 i 个 agent 对共享候选集的全部匹配结果
- 整个 A 表示同一共享 bucket 中所有 query 对共享候选区域的联合打分结果
8.11 softmax 与候选选择
随后可以对 A 的每一行做 softmax：
P = Softmax(A, dim=1)
得到：
P ∈ R^(n × m)
其中：
- 第 i 行 P[i, :] 表示第 i 个 query 对共享候选 key 的概率分布
- 再对每一行做 top-k，即可得到每个 query 在共享候选池中的初始候选集合
但是在系统实现上，更推荐把这一步解释为“共享候选打分”而不是严格概率化。也就是说：
- A 用于构造共享候选池
- P 或 top-k 用于构造 agent-specific 的初始候选列表
- 不建议过早对 block 外候选做硬性归零
8.12 query matrix 的完整数据流总结
下面给出 query matrix 相关的数据流：
原始 agent requests
    ↓
request normalization
    ↓
query encoding
    ↓
per-agent query vectors q_i ∈ R^k
    ↓
按 scope/type/policy 进入共享 bucket
    ↓
构造 query matrix Q ∈ R^(k × n)
    ↓
使用投影矩阵 W / W' / W_final 做共享子空间映射
    ↓
Z = Q^T W_final ∈ R^(n × r)
    ↓
与共享 memory slice 的 keys 做匹配
    ↓
A = Z K^T ∈ R^(n × m)
    ↓
shared candidate scoring / top-k selection
    ↓
共享候选池 C_shared
    ↓
agent-specific rerank
    ↓
private fallback retrieval
    ↓
最终证据集合 C_i_final
这条数据流表明，query matrix 在本设计中并不是附属模块，而是共享检索层的核心组织形式。


9. 共享候选池与个体重排
9.1 共享候选池的含义
经过 query matrix 变换和共享匹配后，系统得到一个 shared candidate pool：
C_shared
这批候选可能包括：
- 共享记忆片段
- 共享 artifact
- 共享事件记录
- 共享知识条目
- 共享历史推理证据
9.2 为什么还需要个体重排
即使多个 agent 共享了相同候选来源，它们最终需要的结果仍然不同。例如：
- planner 更偏 summary、约束、全局状态
- solver 更偏当前子任务直接相关的事实
- verifier 更偏 provenance、timestamp、source reliability
- critic 更偏冲突、边界条件、失败案例
因此，系统需要对共享候选执行 agent-specific rerank：
C_i = Rerank(C_shared, q_i, role_i, state_i)
其中 C_i 是第 i 个 agent 的个体化结果。

10. Private Fallback Retrieval
共享候选池不能保证满足所有 agent 的需求，因此必须保留 private fallback。
如果某个 agent 发现共享候选中没有足够覆盖自己的需求，则系统在该 agent 的私有 scope 或专属 scope 上执行补检索：
C_i_final = Merge(C_i, C_i_private)
其中：
- C_i 是共享候选重排后的结果
- C_i_private 是私有补检索结果
- C_i_final 是最终返回给 agent 的证据集合
private fallback 的意义在于：
1. 保护 recall
2. 避免共享检索过于粗糙造成漏检
3. 保留 agent 的个体差异与私有知识依赖

11. 系统模块设计
11.1 Retrieval Request Encoder
负责把原始 agent 请求转成标准化结构。
输入：
- raw query
- agent id
- role
- allowed scopes
- memory type
- policy constraints
- current state
输出：
- 标准化 retrieval request r_i
11.2 Scope Router
负责根据 scope、type 和 policy 对请求做规则分桶。
输出：
- 可共享的 bucket
- 必须独立处理的 request
11.3 Query Matrix Builder
负责对每个共享 bucket 构造 query matrix。
输出：
- Q ∈ R^(k × n)
11.4 Shared Projection Module
负责执行 W / W' / W_final 的变换，把 query matrix 映射到共享子空间。
输出：
- Z ∈ R^(n × r)
11.5 Shared Candidate Retriever
负责使用 Z 和 memory slice keys 构造共享候选池。
输出：

- 共享候选池 C_shared
11.6 Personalized Reranker
负责每个 agent 对共享候选做个体化排序。
输出：
- C_i
11.7 Private Fallback Retriever
负责在 private scope 上做补检索。
输出：
- C_i_private
11.8 Evidence Fusion Layer
负责将共享结果与补检索结果融合成最终输出。
输出：
- C_i_final

12. 数据结构建议
12.1 检索请求对象
{
  "request_id": "r_001",
  "agent_id": "planner_1",
  "role": "planner",
  "query": "当前任务的关键约束是什么",
  "scope": {
    "private_scopes": ["agent/planner_1"],
    "shared_scopes": ["task/alpha/shared", "session/current"],
    "workspace_scopes": ["workspace/project_x"]
  },
  "memory_types": ["episodic", "artifact", "shared"],
  "policy": {
    "visibility": ["team"],
    "max_clearance": 2,
    "exclude_quarantined": true
  },
  "state": {
    "plan_node": "constraint_analysis",
    "known_evidence_ids": ["mem_12", "mem_30"]
  }
}
12.2 候选记忆对象
{
  "memory_id": "mem_203",
  "scope_id": "task/alpha/shared",
  "memory_type": "artifact",
  "confidence": 0.91,
  "salience": 0.82,
  "ttl": 7200,
  "provenance": {
    "source": "tool_output",
    "parent_event": "ev_882"
  },
  "content": "...",
  "embedding": "...",
  "visibility": ["team"]
}


13. 伪代码
Input:
  R = {r1, r2, ..., rn}

Step 1: Normalize requests
  For each ri in R:
      encode ri into structured request object

Step 2: Route by scope/type/policy
  place each ri into scope buckets
  split bucket by memory type
  split again if policy is incompatible

Step 3: Build query matrix for each shareable bucket
  For each shareable bucket Bb:
      collect queries {q1, q2, ..., qn}
      build Q in R^(k × n)

Step 4: Shared projection
  compute W' and W_final if needed
  compute Z = Q^T W_final

Step 5: Shared candidate retrieval
  fetch keys K from shared memory slice
  compute A = Z K^T
  select shared candidates C_shared

Step 6: Personalized rerank
  For each request ri in Bb:
      Ci = Rerank(C_shared, ri.query, ri.role, ri.state)

Step 7: Private fallback
  For each request ri:
      if Ci is insufficient:
          Ci_private = PrivateFallbackRetrieve(ri)
          Ci_final = Merge(Ci, Ci_private)
      else:
          Ci_final = Ci

Output:
  {C1_final, C2_final, ..., Cn_final}

14. 评测建议
Baseline 1：Independent Retrieval
每个 agent 独立检索。
Baseline 2：Naive Shared Retrieval
多个 agent 直接共享一次检索结果，不做个体重排。
Baseline 3：Shared Candidate + Personalized Rerank
共享候选池后，各自重排，但不做 fallback。
Proposed：Scope-Overlap-Aware Collaborative Retrieval
先按 scope/type/policy 分桶，构造 query matrix，共享 first-stage retrieval，再个体重排，最后 private fallback。

关注以下指标：
- Recall@k
- MRR@k
- evidence hit rate
- answer support rate
- first-stage retrieval 次数
- 重复候选率
- 候选复用率
- fallback 触发率
- 平均延迟
- 对应任务的指标（acc、bleu等等）

# 第14章 实验设计

## 14.1 数据集

### 14.1.1 数据集概览

本文使用五个数据集，分为多跳问答和数学推理两类。所有数据集使用统一的 schema 和 agent 配置规则，memory store 结构、agent 分配方式、ground-truth 标注方式完全一致。

| 数据集 | 类型 | 评测 split | episode 数 | 跳数/步数 |
|---|---|---|---|---|
| MuSiQue | 多跳问答 | dev | ~2400 | 2/3/4跳 |
| 2WikiMultiHopQA | 多跳问答 | dev 前3000条 | ~3000 | 2跳（多类型） |
| HotpotQA | 多跳问答 | dev distractor | ~7400 | 2跳 |
| GSM8K | 数学推理 | test | ~1300 | 4–6步 |
| MATH | 数学推理 | test（步数≥3） | ~2500 | 3–9步 |

### 14.1.2 Episode 结构

每条 episode（JSONL 一行）包含以下内容：

**固定结构（来自数据集）：**
- `question` / `answer`：题目和答案
- `got_graph`：agent 拓扑结构，描述谁依赖谁
- `agents`：每个 agent 的配置和 scope 权限声明
- `retrieval_requests`：每个 agent 的检索请求，含实时生成的 query
- `ground_truth`：每个 agent 应该检索到哪些 memory_id，用于评测打分
- `rho` / `rho_subset`：scope overlap 值和子集标签（S1/S2/S3/S4）

**Memory Store（系统运行时产出，写入后存入 JSONL）：**

| 层 | slot | 内容 | 可见性 |
|---|---|---|---|
| workspace_semantic | — | 公共知识库（corpus 段落） | PUBLIC，所有 agent |
| task_shared_plan | plan | Planner 的任务分解计划 | TEAM，所有 agent |
| task_shared_artifact | conclusion | 每个 Solver 产出的结论 | TEAM，所有 agent |
| agent_private_intent | query_intent | 每个 agent 的检索前推理意图 | OWNER，仅自己 |
| agent_private | scratch | 每个 agent 的推理草稿 | OWNER，仅自己 |
| restricted | audit_report | Verifier 评判记录（仅 S4） | RESTRICTED，仅 Verifier |

**关键设计：**
- workspace_semantic 扁平存储，所有 agent 共享，不按 hop/agent 分区
- agent_private 的内容对其他 agent 完全不可见，Solver_1 的草稿 Solver_2 看不到
- ρ 在 Step 6 写回 memory 时决定：写入 task_shared 越多则重叠越大，ρ 越高

### 14.1.3 各数据集差异点

| 数据集 | task_shared 写入方式 | 特殊说明 |
|---|---|---|
| MuSiQue | online（每轮实时写入） | 主力数据集，完整覆盖 S1–S4 |
| 2WikiMultiHopQA | online | comparison 类适合构造 S2 场景 |
| HotpotQA | online | 候选池固定10条（2 gold + 8 distractor） |
| GSM8K | oracle（步骤预写入） | gold 计算结论预写入 task_shared |
| MATH | oracle | workspace 额外写入公式和定理 |

---

## 14.2 子集划分

### 14.2.1 ρ 的定义与产生时机

ρ 衡量不同 Solver 在运行时需要检索的内容重叠程度。**ρ 在 Step 6 写回 Memory Store 时产生**，不是事后计算的：

```
对每个 Solver_i：
    A_i = {plan.memory_id}
        ∪ {conclusion.memory_id | 产生者 ∈ ancestors_or_self(i)}

ρ(episode) = mean_{i<j} |A_i ∩ A_j| / |A_i ∪ A_j|
```

排除范围：workspace_semantic（所有 solver 等价可见）、query_intent + scratch（各 solver 独有）。

评测时直接读 JSONL 里的 `rho_subset` 字段，不需要重新计算。

### 14.2.2 子集定义

阈值：high = 0.6，low = 0.4。

| 子集 | ρ 范围 | 含义 | 实测对应图类型 |
|---|---|---|---|
| S1 | ρ > 0.6 | 高重叠，协同检索收益最大 | LINEAR、POLICY_ISOLATED |
| S2 | 0.4 < ρ ≤ 0.6 | 中等重叠 | FORK_MERGE |
| S3 | ρ ≤ 0.4 | 低重叠，接近独立检索 | FORK、INDEPENDENT |
| S4 | policy 冲突 | 安全性验证，与 S1/S2/S3 互斥 | 全量按3%–5%分层抽取 |

实测 ρ 分布（MuSiQue dev）：

| 图类型 | ρ 均值 | 子集 |
|---|---|---|
| FORK | 0.333 | S3 |
| INDEPENDENT | 0.394 | S3 |
| FORK_MERGE | 0.444 | S2 |
| LINEAR 2-hop | 0.667 | S1 |
| POLICY_ISOLATED | 0.667 | S1 |

### 14.2.3 推理结构对 ρ 的影响

GoT / CoT / ToT 三种推理结构**分别构造三套 episode JSONL**，各自独立计算 ρ，独立划分子集标签。

| 结构 | agent_private 规模 | ρ 趋势 | S1 占比 |
|---|---|---|---|
| GoT（主实验） | 最大 | 最低 | 最低（~25%） |
| ToT | 居中 | 居中 | 居中 |
| CoT | 最小 | 最高 | 最高（~40%） |

GoT 节点最多，各节点有独立的私有推理路径，写入 agent_private 的内容最多，写入 task_shared 的相对少，ρ 系统性偏低。CoT 线性链上每步结论全部写入 task_shared，ρ 最高。

### 14.2.4 S4 构造

从每个数据集全量 episode 中按 seed=42 分层随机抽取（S1/S2/S3 三档各占 1/3），抽取后从原子集中移除，与 S1/S2/S3 完全互斥。

各数据集 restricted scope 内容：

| 数据集 | restricted 内容 | 生成方式 |
|---|---|---|
| MuSiQue / 2WikiMultiHopQA / HotpotQA | supporting facts 句子级来源可信度标注 | 确定性规则 |
| GSM8K / MATH | 解题步骤数值一致性校验记录 | 确定性规则 |

S4 仅报告 False Merge Rate（FMR），不参与任务完成指标评测。

---

## 14.3 CoScope 检索机制

### 14.3.1 每轮 t 的实时执行流程

系统完全实时运行，每一轮 t 执行当前批次的 agent，全部 6 步均为实时：

```
Step 1：硬依赖注入
  父节点的 conclusion 直接作为 context 喂给当前 agent prompt
  （强制依赖，不经过检索）

Step 2：检索前推理（★ 核心新增）
  每个 agent 独立推理："我这一步需要查什么？"
  → 产出 query_intent，写入 agent_private
  Planner → 整个任务需要哪些大方向信息
  Solver_k → 当前步骤需要哪些具体事实

Step 3：生成 retrieval request
  基于 query_intent 生成精准的检索请求
  带 scope / policy / state

Step 4：CoScope 协同检索（★ 核心验证对象）
  检索来源 A：workspace_semantic（公共知识库）
  检索来源 B：Memory Store 三层（其他 agent 已产出的内容）
  pipeline：scope 分桶 → batch query matrix → shared rerank
           → private fallback → C_i_final

Step 5：LLM 推理产出（Qwen 32B）
  输入：context（硬依赖）+ C_i_final（检索结果）
  产出：scratch（私有草稿）+ conclusion（对外结论）

Step 6：写回 Memory Store  ★ ρ 在此产生
  scratch    → agent_private（仅自己可见）
  conclusion → task_shared（所有 agent 可见）
  audit_report → restricted（仅 Verifier，S4 场景）

下一轮 agent 通过 CoScope 检索以上内容，循环直到出最终答案
```

### 14.3.2 Batch Query Matrix（核心机制）

Step 4 的核心创新：将同一 scope bucket 内多个 agent 的 query 组织为矩阵，联合处理，而非逐个独立检索。

**完整数据流：**

```
同一 bucket 内 n 个 agent 各自的 query 文本
    ↓
text-embedding-v3（通义千问）实时向量化
    ↓
q_i ∈ R^k，按列拼成 Q ∈ R^(k × n)
    ↓
Truncated SVD：Q = U Σ V^T，取前 r 个左奇异向量
Z = U[:, 1:r]  ∈ R^(k × r)
    ↓
与 memory keys 做匹配：A = Z^T K^T  ∈ R^(n × m)
    ↓
按行 top-k → 共享候选池 C_shared
    ↓
agent-specific rerank
    ↓
private fallback（候选不足时补充私有检索）
    ↓
C_i_final（每个 agent 的最终候选集）
```

**SVD 的作用：**
不训练任何参数，直接对当前 bucket 的 query matrix 做 Truncated SVD，提取这批 query 的主方向，用主方向联合检索 memory。每个 bucket 独立算一次，算完即用，无需存储。适用条件：bucket 内 agent 数 n ≥ 3，n < 3 时退化为 query 均值。

**与独立检索的区别：**

| | 独立检索（A1） | Batch Query Matrix（CoScope） |
|---|---|---|
| 处理方式 | 每个 query 单独检索 | 同 bucket 内 query 拼成矩阵联合处理 |
| query 方向差异 | 丢失 | 保留 |
| first-stage 次数 | n 次 | 1 次 |
| 跨 agent 收益 | 无 | ρ 越高收益越大 |

---

## 14.4 对比方法

### 14.4.1 内部消融（A1–A8）

GoT / CoT / ToT 三种推理结构各自独立构造 JSONL，各自独立跑 A1–A8 全部变体，独立出结果表。

变体定义如下（三种推理结构下完全相同）：

| 变体 | 描述 |
|---|---|
| A1 | 每个 agent 完全独立检索，无任何共享（下界基线） |
| A2 | 强制合并所有请求，不做 scope/policy 校验，无重排无 fallback |
| A3 | 仅按 scope 分桶，query 直接平均，无重排无 fallback |
| A4 | scope 分桶 + 个体重排 + private fallback，无 query matrix |
| A5 | scope 分桶 + batch query matrix + SVD + 重排 + fallback，无 block routing |
| A6 | A5 + block routing |
| A7 | 去掉 Step 2（直接用 raw question 做 query），其余与 A6 相同 |
| A8 | 完整方法：全组件开启，含 Step 2 检索前推理 |

执行顺序：A1 → A2 → A3 → A4 → A5 → A6 → A7 → A8（全部无需训练）。

**消融轴对照（组件边际贡献，在每种推理结构下各自成立）：**

| 对比 | 验证内容 |
|---|---|
| A1 vs A3 | scope 分桶的效率收益 |
| A3 vs A4 | 个体 rerank + fallback 的质量收益 |
| A4 vs A5 | batch query matrix + SVD 的收益（核心） |
| A5 vs A6 | block routing 的效率与精度收益 |
| A7 vs A8 | Step 2 检索前推理的收益 |
| A2 on S4 | 无差别共享的安全风险（FMR） |

**推理结构对比（跨 JSONL 水平对比）：**

| 对比 | 验证内容 |
|---|---|
| GoT-A8 vs CoT-A8 vs ToT-A8 | 三套推理结构下完整方法的横向对比，量化结构对 fallback、任务完成与检索效率的影响 |

### 14.4.2 外部对比方法

| 方法 | 类型 | 适用数据集 | 说明 |
|---|---|---|---|
| BM25 | 稀疏检索 | 全部五个 | 词频检索，无跨 agent 共享，检索下界 |
| DPR | 密集检索 | 全部五个 | 双编码器，每个 agent 独立运行 |
| MemMA | 多 agent 记忆管理 | MuSiQue、2Wiki、HotpotQA | 原生多 agent 记忆架构，代码开源，定位最接近 |
| Collaborative Memory | 多 agent 协同记忆 | 多跳问答 | agent 间记忆协作机制，直接对标本文 "scope overlap-aware" 思想 |
| AMA (Adaptive Memory Agent) | 多 agent 自适应记忆 | 多跳问答 | 动态记忆分配，验证是否依赖 scope/policy 显式建模 |
| CoMAM (Cooperative Multi-Agent Memory) | 多 agent 合作记忆 | 多跳问答 | 合作式共享策略，验证显式 bucket 划分的效率收益 |
| LegalMALR（裁剪版） | 多 agent 法律检索 | 多跳问答（裁剪通用化） | 原法律领域多 agent 检索系统，去除领域特化后跑通用 QA |
| L-MARS（裁剪版） | 多 agent 检索 RL | 多跳问答（裁剪通用化） | 强化学习驱动的多 agent 检索，裁剪为无 RL 版本作 retrieval 对照 |

注：
- GSM8K / MATH 数学推理类仅运行 BM25 和 DPR，其他多 agent 系统不适配数学推理场景。
- LegalMALR / L-MARS 的裁剪版仅保留检索机制，去除原系统中的领域特化模块。

---

## 14.5 评测方式

### 14.5.1 两种评测模式

**模式 B（主实验）：算法评测**

```
读 JSONL：
  retrieval_requests（query + scope + policy）
  memory_entries（workspace + 已有 artifact）
      ↓
Step 4 CoScope 实时跑（唯一变量）
      ↓
C_i_pred vs ground_truth → Recall@k / FMR
      ↓
按 rho_subset 分组报告 S1/S2/S3/S4
```

Step 4 是唯一变量，排除 LLM 随机性干扰，结果可复现。ground_truth 直接用 JSONL 里的标注，memory_id 精确对应。

**模式 A（附录）：端到端评测**

```
Step 1–6 全部实时跑，多轮迭代
      ↓
最终答案 vs gold answer → EM / F1
```

ground_truth 只用 workspace_semantic 层（corpus 级，对所有 run 都稳定）。

### 14.5.2 结果汇报

五个数据集各自独立汇报，主表格式（以 MuSiQue 为例）：

| 方法 | S1 EM/F1 | S2 EM/F1 | S3 EM/F1 | S4 FMR | Recall@10 | Evidence Hit Rate |
|---|---|---|---|---|---|---|
| A1 | | | | — | | |
| A2 | | | | ~1.0 | | |
| A3–A6 | | | | — | | |
| A8 | | | | ~0.0 | | |
| BM25 / DPR / … | | | | — | | |

S4 列仅报告 FMR，不报告任务完成指标。

**跨数据集 S4 FMR 汇总（附表）：**

| 数据集 | A2 FMR（预期~1.0） | A8 FMR（预期~0.0） |
|---|---|---|
| MuSiQue | | |
| 2WikiMultiHopQA | | |
| HotpotQA | | |
| GSM8K | | |
| MATH | | |

---

## 14.6 评测指标

评测指标分三层：**检索质量**（模式 B 核心，所有数据集通用）、**任务完成质量**（模式 A 端到端，分数据集特化）、**效率与安全性**（所有数据集通用）。

### 14.6.1 检索质量（模式 B 通用）

模式 B 的核心验证对象。ground_truth 来自 JSONL 的 `ground_truth` 字段，memory_id 精确对应，不依赖 LLM 随机性。

| 指标 | 定义 | 报告粒度 |
|---|---|---|
| **Recall@k** | 每个 retrieval request 的 top-k 中命中 gold memory 的比例，按 request 平均 | k=5, 10；按 S1/S2/S3 分层 |
| **MRR@k** | Mean Reciprocal Rank，首个命中 gold 的倒数排名，按 request 平均 | k=10；按 S1/S2/S3 分层 |
| **Evidence Hit Rate** | 每个 episode 的所有 agent 的 top-k 的并集是否完整覆盖 supporting facts | 按 episode 平均 |
| **Answer Support Rate** | top-k 检索结果中是否包含足以推出最终答案的 memory（需要 gold chain 标注） | 按 episode 平均 |

### 14.6.2 任务完成质量（模式 A 端到端，分数据集特化）

模式 A 才有的指标。每个数据集的 gold 形式不同，metric 也不同。

**多跳问答类（MuSiQue / 2WikiMultiHopQA / HotpotQA）：**

| 指标 | 定义 | 说明 |
|---|---|---|
| **EM (Exact Match)** | 最终答案字符串归一化后严格匹配 gold answer | 主指标 |
| **F1** | token-level F1，最终答案 tokens 与 gold answer tokens 的 F1 | 主指标 |
| **Supporting Facts EM/F1** | 模型引用的 supporting facts 与 gold 的严格/F1 匹配 | 仅 HotpotQA（原数据集自带标注） |
| **Joint EM/F1** | 答案与 supporting facts 同时匹配 | 仅 HotpotQA |

**数学推理类（GSM8K / MATH）：**

| 指标 | 定义 | 说明 |
|---|---|---|
| **Accuracy（GSM8K）** | 最终数值答案字符串匹配（归一化数字格式） | 主指标 |
| **Accuracy（MATH）** | 最终表达式通过 sympy 等价性判定匹配 gold | 主指标 |
| **Step Accuracy** | 每个中间推理步骤的结论是否与 oracle 步骤一致 | 可选，GSM8K/MATH 均适用 |
| **Formula Hit Rate（MATH）** | workspace_semantic 中预写入的公式/定理是否被模型正确引用 | MATH 专用 |

### 14.6.3 效率（所有数据集通用）

| 指标 | 定义 | 适用模式 |
|---|---|---|
| **Shared-first-stage Savings** | `1 - 实际 first-stage 检索次数 / n`（n = 请求数）；A1 为 0，其他变体越高越好 | 模式 B |
| **Fallback Necessity Rate** | 个体 rerank 后触发 private fallback 的比例 | 模式 B |
| **平均端到端延迟（ms）** | 从 request 集合到返回所有 retrieval 结果的总耗时 | 模式 B |
| **任务平均轮数** | 完整任务从 Step 1 到最终答案经历的轮数 | 模式 A |
| **任务平均端到端延迟** | 完整任务的总耗时（含 LLM 推理） | 模式 A |

### 14.6.4 分桶安全性（S4 专用，所有数据集通用）

| 指标 | 定义 | 说明 |
|---|---|---|
| **Routing FMR** | 有 policy 冲突的请求被错误放进同一共享桶的比例 | 路由层检测，A2 预期≈1.0，A8 预期≈0.0 |
| **Content FMR** | 非 Verifier 的 top-k 实际包含 restricted memory 的比例 | 内容层检测，最严格的安全指标 |
| **Shareability Precision** | 被系统判定为可共享的请求对中，实际 scope/policy 确实兼容的比例 | 越高越好 |
| **Shareability Recall** | 实际应该共享的请求对中，被系统正确识别并合并的比例 | 越高越好 |

### 14.6.5 各数据集指标套装汇总

| 数据集 | 模式 B 检索指标 | 模式 A 任务指标 | S4 安全性指标 |
|---|---|---|---|
| MuSiQue | Recall@k, MRR@k, Evidence Hit Rate | EM, F1 | Routing FMR, Content FMR |
| 2WikiMultiHopQA | Recall@k, MRR@k, Evidence Hit Rate | EM, F1 | Routing FMR, Content FMR |
| HotpotQA | Recall@k, MRR@k, Evidence Hit Rate | EM, F1, Sup-EM/F1, Joint EM/F1 | Routing FMR, Content FMR |
| GSM8K | Recall@k, MRR@k | Accuracy, Step Accuracy | Routing FMR, Content FMR |
| MATH | Recall@k, MRR@k | Accuracy（sympy 等价）, Formula Hit Rate | Routing FMR, Content FMR |

---

## 14.7 消融分析

### 14.7.1 各组件边际贡献

每个变体仅控制一个变量，逐步开启组件，量化各模块的边际贡献：

- **Scope 分桶（A1 vs A3）**：效率收益为主，关注 first-stage savings 和延迟变化，预期质量提升有限
- **个体 rerank + fallback（A3 vs A4）**：关注 Recall@k 和 Evidence Hit Rate 提升，预期在 S2 部分重叠子集最显著
- **Batch query matrix + SVD（A4 vs A5）**：**共享候选池的 first-stage 批量化机制**，核心价值在于使同一个 shared bucket 内的多个 agent query 共用一次 first-stage retrieval，是计算效率的共享化；在搭配个体级全维 rerank 的前提下 Recall@k 与 A4 持平，并不直接带来质量提升。SVD 的价值是（i）为多 agent 提供统一的候选池接口，（ii）为后续 block routing 与 policy 隔离提供结构基础
- **Block routing（A5 vs A6）**：关注 S1 高重叠场景下 memory slice 规模最大时的延迟收益
- **Step 2 检索前推理（A7 vs A8）**：query 更精准，query matrix 方向差异更真实，预期 Recall@k 提升，在 S2 效果最显著
- **Policy 安全性（A2 on S4）**：FMR 接近 1，证明无差别共享的安全风险；A8 on S4 FMR 接近 0，证明分权隔离的有效性

### 14.7.2 SVD 有效秩 r 的敏感性分析

**重要声明：**在现代 dense encoder（k 大约 384–1024）与典型 multi-agent bucket size（n = 3–5）下，有效秩 r = min(r_config, n, k) 被 n 硬约束，往往减少到 3–5。因此 r_config 上界对 **天花板表现**并无防御性影响。

重点实验：

- 在 r_config ∈ {8, 16, 32, 64} 上网格搜索，测量实际有效秩改变对 Recall@10 、latency、first-stage savings 的影响
- 预期 r_config ≥ n 时 Recall@10 达到上限并饱和，仅 latency 随 r_config 增长
- 低维 embedding (k ≤ n 时或 k 高但 r_config 接近 k 时) 下 SVD 截断的信号损失才会引起质量下降

另外分析 bucket 内 agent 数 n 对 SVD 效率价值的影响：n = 1 时 SVD 退化为单 query 检索，共享化无收益；n ≥ 3 时 first-stage 批量化收益随 n 线性增长。

### 14.7.3 Scope 重叠程度的影响

在 S1/S2/S3 三档分别报告 A8 vs A1 的提升幅度，绘制提升幅度随 ρ 变化的折线图。

预期正相关：S1 提升最大，S3 接近 0，证明协同检索的自适应性——在不该共享时不强行共享，ρ 低时自动退化接近独立检索。

### 14.7.4 Agent 数量的影响

将 bucket 内 agent 数 n 从 2 变化至 6，分析对以下指标的影响：

- **Recall@10**：预期 n=3–4 时最优，n 过大时 block routing 的局部路由价值更突出
- **平均端到端延迟**：n 增大时 query matrix 构造和 SVD 计算开销增加
- **SVD 有效性**：n < 3 时退化，n ≥ 3 时开始稳定

### 14.7.5 跨数据集泛化性分析

SVD 是无监督方法，不依赖任何训练数据，天然具备跨数据集泛化性。在附表中报告：

- A5（batch query matrix + SVD）在五个数据集上的 Recall@10 提升方向是否一致
- 若在 MuSiQue 上确定的最优 r，在 2WikiMultiHopQA、HotpotQA、GSM8K、MATH 上的表现
- 两类数据集（多跳问答 vs 数学推理）的提升幅度差异及原因分析

### 14.7.6 推理结构的影响

GoT / CoT / ToT 三套 JSONL 各自独立构造、独立跑 A1–A8。主表报告各推理结构下 A8 vs A1 的提升幅度对比，量化完整方法相对独立检索的增益是否随结构变化。组件边际贡献（A2–A7）作为附表在每种结构下选择性报告。

**推理结构横向对比（GoT-A8 vs CoT-A8 vs ToT-A8）：**

| 指标 | GoT 预期 | ToT 预期 | CoT 预期 | 原因 |
|---|---|---|---|---|
| Fallback Necessity Rate | 最高 | 居中 | 最低 | GoT 节点多，共享候选难以全覆盖 |
| Evidence Hit Rate | 最高 | 居中 | 较低 | 节点级 fallback 弥补漏洞更充分 |
| S2 任务完成 | 最优 | 居中 | 较差 | 部分重叠场景节点级 fallback 更精准 |
| 平均端到端延迟 | 最高 | 居中 | 最低 | 图遍历和多节点 fallback 开销 |

附表中报告三种结构下 S1/S2/S3 的实际 ρ 分布，验证 GoT ρ 系统性低于 CoT 的预期，并分析 ρ 偏移对各结构任务完成指标的影响。

---

## 14.8 效率分析

效率结果按 S1/S2/S3 分层报告：

| 指标 | A1 | A5 | A8 |
|---|---|---|---|
| Shared-first-stage Savings | 0% | — | — |
| 平均端到端延迟（ms） | — | — | — |
| Fallback Necessity Rate | — | — | — |

---

## 14.9 数据验收标准

### 14.9.1 Episode 级别验收

| 检查项 | 规则 | 失败处理 |
|---|---|---|
| workspace_semantic 非空 | item 数 ≥ 1 | 过滤 |
| restricted 仅 Verifier 可访问 | visibility = ["verifier"] | 过滤 |
| memory_id 全局唯一 | 同一 episode 内无重复 | 报错去重 |
| agent_private 内容各自独立 | Solver_i 的 private scope 不在 Solver_j 的 allowed_scopes 里 | 报错 |
| ρ ∈ [0, 1] | 0 ≤ ρ ≤ 1 | 重新计算 |
| 子集标签与 ρ 一致 | S1: ρ>0.6；S2: 0.4<ρ≤0.6；S3: ρ≤0.4 | 重新分配 |
| ground_truth 非空 | 每个 agent 至少一条 | 过滤 |
| ground_truth 的 memory_id 在 memory_entries 中存在 | 逐条检查 | 报错 |

### 14.9.2 数据集级别验收

| 检查项 | 预期 |
|---|---|
| S1 占比 | MuSiQue ~30%–40%；GSM8K/MATH ~60%–80% |
| S3 占比 | 各数据集均 > 5% |
| S4 内部三档均衡 | S1/S2/S3 各占 S4 约 1/3（±5%） |
| GoT S1 占比 < CoT S1 占比 | 同数据集内方向性验证 |
| 三套 JSONL 题目集合完全一致 | 固定题目，只改推理结构 |
| S4 与 S1/S2/S3 完全互斥 | 同一 episode_id 不出现在多个子集 |
| S4 中 Verifier 均持有 restricted item | 100% |
