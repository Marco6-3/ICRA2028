# I002 — Asynchronous Multi-Rate Semantic–Tactile Policy Communication

**Status:** Candidate research direction  
**Theme:** tactile manipulation / asynchronous control / multi-rate policy / multimodal communication / shared latent representation  
**Goal:** 研究低频 semantic policy 与高频 tactile policy 在异步运行时应如何交换信息，以及通信量、通信方向、信息陈旧程度与控制性能之间的关系。

---

## 1. Core question

接触丰富操作中，semantic policy 与 tactile policy 天然处在不同时间尺度：

- semantic / visuomotor reasoning 往往更慢、更贵；
- tactile reaction 需要更快、更局部、更低延迟。

因此一个核心问题不是简单地“触觉是否应该更高频”，而是：

> How should asynchronous slow semantic and fast tactile policies communicate under different control rates?

更具体地，需要回答：

1. **异步运行是否优于强制同步运行？**
2. **两个 policy 应共享多少信息？**
3. **共享的是 coarse action、少量 learned tokens、完整 semantic latent，还是更接近全量 multimodal context？**
4. **共享信息可以陈旧多久，fast policy 仍然能够稳定工作？**
5. **通信应该是 semantic -> tactile 单向，还是 semantic <-> tactile 双向？**
6. **是否应该固定频率更新 semantic policy，还是由 tactile / uncertainty / contact event 触发重新规划？**

这些都作为待验证问题，不预设答案。

---

## 2. 与 I001 的区别

[I001](I001_HIGH_FREQUENCY_TACTILE_CONTEXT.md) 主要研究：

- 高频 tactile policy 是否需要短期触觉 / action / proprioception history；
- 更长 interaction history 是否可以形成 in-context adaptation / implicit system identification。

I002 主要研究：

- slow semantic policy 与 fast tactile policy 如何异步执行；
- 两个 policy 之间交换什么信息、交换多少、多久更新一次；
- 通信机制本身如何影响闭环性能、延迟与泛化。

因此：

[
	ext{I001: fast tactile policy 内部需要什么时间上下文}
]

[
	ext{I002: fast tactile policy 与 slow semantic policy 之间如何通信}
]

两者可以共享实验基础设施，但不是同一个科学问题。

---

## 3. 为什么异步可能是必要的

一个简单同步多模态策略可以写成：

[
a_t = pi(V_t, T_t, q_t, L)
]

其中视觉、触觉、语言和机器人状态在同一时刻一起进入同一个 policy。

这种设计的问题可能包括：

- 所有模态被迫以同一 policy rate 运行；
- semantic backbone 的推理成本限制 tactile reaction rate；
- 高频 tactile observation 可能被低频视觉 / 大模型推理拖慢；
- 每次 tactile update 都重新计算大量不需要快速变化的 semantic context。

异步设计则允许：

[
z_s^{(k)} = f_s(V,L,q), qquad f_s approx 2	ext{–}10 mathrm{Hz}
]

[
a_t = f_t(T_{t-H:t}, q_t, M_t), qquad f_t approx 20	ext{–}200+ mathrm{Hz}
]

其中 (M_t) 是 semantic 与 tactile 模块之间的共享信息。

重点不是先规定具体频率，而是研究 **multi-rate execution + communication**。

---

## 4. “共享 token”本身是实验变量

不能预设“少量 token 一定最好”。

设 semantic network 输出 hidden representation：

[
H_s in mathbb{R}^{N 	imes d}
]

可以构造不同通信容量：

### A. 只共享 coarse action

[
M = a^{semantic}
]

这是最强信息瓶颈之一。

### B. 共享少量 learned tokens

[
H_s
ightarrow
	ext{Token Compressor}
ightarrow
M_K in mathbb{R}^{K	imes d}
]

测试：

[
K in {1,4,16,64,ldots}
]

### C. 共享完整 semantic latent

[
M = H_s
]

不做显式压缩。

### D. 接近 full multimodal cross-attention

fast tactile policy 可以访问大量 visual / language / semantic tokens。

这时系统虽然仍可在软件上保持两个异步模块，但功能耦合会越来越强。

这里所谓“趋向 monolithic”不是指性能一定下降，而是：

- fast loop 对完整 semantic context 的依赖增强；
- communication bandwidth 增加；
- high-rate inference 成本可能上升；
- 对 stale semantic context 的脆弱性可能增加；
- 两个时间尺度的功能独立性可能降低。

这些都需要实验验证，而不是理论上直接断言。

---

## 5. Proposed architecture

第一版可以采用显式双流：

```text
RGB / language / task state
            │
            ▼
      Semantic policy
          2–10 Hz
            │
            ├──── coarse action
            │
            └──── semantic tokens / latent
                         │
                         ▼
                Shared policy memory
                         ▲
                         │
tactile history ──> Fast tactile policy
proprioception       20–200+ Hz
                         │
                         ▼
                  residual / refined action
                         │
                         ▼
                 robot controller
```

一种基础动作形式：

[
a_t = a_t^{semantic} + Delta a_t^{tactile}
]

但 residual 形式也只是 baseline，不排除：

- tactile policy 直接预测 refined action；
- tactile policy 输出 action gate / gain；
- tactile policy 只在 contact event 后接管；
- hierarchical option / skill switching。

---

## 6. Communication dimensions to study

### 6.1 Communication capacity

研究 shared representation 大小：

[
K = 0,1,4,16,64,	ext{full}
]

观察是否存在：

- saturation；
- under-capacity；
- over-coupling；
- bandwidth–performance trade-off。

### 6.2 Communication content

比较：

- coarse action only；
- learned latent tokens；
- raw semantic hidden states；
- hand-designed state summary；
- tactile summary token；
- multimodal cross-attention。

不提前规定 token 必须具有“task phase”“contact state”等人类语义。

可以在训练后通过 probing / visualization 分析，但这些诊断不能单独证明因果。

### 6.3 Staleness

semantic token 在一次更新后可能被 fast tactile policy 重复使用：

[
M_{t_0}, M_{t_0}, ldots, M_{t_0}
]

直到下一次 semantic update。

测试：

[
Delta t = 0, 50, 100, 200, 500 mathrm{ms}, ldots
]

核心问题：

> How stale can semantic context become before fast tactile control degrades?

同时应区分：

- deterministic delay；
- random jitter；
- dropped updates；
- delayed observation；
- delayed communication。

### 6.4 Communication direction

比较：

#### Uni-directional

[
semantic ightarrow tactile
]

#### Bi-directional

[
semantic leftrightarrow tactile
]

tactile 可以向 semantic 返回：

- learned tactile token；
- contact event；
- uncertainty；
- anomaly / failure signal；
- interaction summary。

但这些信号形式本身也不应过早固定。

### 6.5 Update schedule

比较：

- fixed-rate semantic update；
- periodic + contact-triggered update；
- uncertainty-triggered update；
- tactile-event-triggered replanning；
- adaptive semantic rate。

例如 tactile policy 检测到重大接触变化后触发：

[
e_t = 	ext{replan}
]

使 semantic policy 不必始终以固定高频运行。

---

## 7. First simulation platform: Franka

第一阶段计划优先在仿真中的 Franka 平台验证。

Franka 只是实验平台，不希望科学问题依赖 Franka 特定关节定义。

### 上层动作空间优先保持 embodiment-agnostic

例如使用：

[
a =
[Delta x,Delta y,Delta z,
Delta r_x,Delta r_y,Delta r_z,
Delta g]
]

即：

- end-effector delta pose；
- gripper command。

下面再由 robot-specific controller 转换：

[
a_{EE}
ightarrow
	ext{IK / operational-space controller}
ightarrow
q_{target}
]

这样后期即使课题组真机不是 Franka，也可以保留上层 semantic–tactile architecture，只替换：

- robot model；
- controller；
- kinematics / dynamics adapter；
- tactile hardware interface。

### 真机阶段不绑定 Franka

后期可能使用课题组能够提供的其他机械臂。

真机验证目标应优先是：

> 异步通信机制是否能够跨 controller / embodiment 保持作用。

而不是要求仿真和真机必须是同型号机器人。

---

## 8. Tactile simulation strategy

不要在第一阶段同时把“异步通信问题”和“高保真 tactile sensor simulation”绑在一起。

### Stage A — low-dimensional contact sensing

先使用可控的物理量：

[
T_t =
[F_x,F_y,F_z,	au_x,	au_y,	au_z]
]

或：

- fingertip contact forces；
- shear / normal components；
- contact points；
- penetration / deformation proxy；
- slip proxy。

目标是快速验证 multi-rate communication hypothesis。

### Stage B — richer tactile representation

如果 Stage A 有明确结果，再升级：

- tactile image；
- deformation field；
- force field；
- learned tactile encoder；
- sensor-specific representation。

这样可以区分：

[
	ext{architecture contribution}
]

和

[
	ext{sensor realism contribution}
]

---

## 9. Minimal tasks

任务需要真的依赖快速接触反馈，而不是纯视觉即可完成。

优先候选：

### Task A — disturbed grasp

- robot 抓住物体；
- 改变外部扰动 / friction / load；
- 产生 slip / contact instability；
- fast tactile policy 尝试恢复稳定。

适合研究 reaction latency 与 stale semantic context。

### Task B — force-sensitive insertion / contact

- peg / connector / constrained insertion；
- semantic policy 给出 nominal motion；
- tactile policy 根据接触修正局部动作。

适合研究 slow intent 与 fast contact correction 的分工。

### Task C — contact transition task

- push / slide / rotate / regrasp；
- task 中存在明确 contact-mode transition。

适合研究 tactile -> semantic event triggering。

第一阶段不需要同时做很多任务；优先找一个能明确区分 baseline 的任务。

---

## 10. Baselines

至少比较：

1. **Synchronous monolithic multimodal policy**
2. **Slow semantic only**
3. **Fast tactile only**
4. **Async + coarse action only**
5. **Async + compressed shared tokens**
6. **Async + full semantic latent**
7. **Async + bidirectional communication**
8. **Async + event-triggered semantic update**

如果算力有限，可以分阶段增加，而不是一次训练全部。

---

## 11. Metrics

不能只报告 success rate。

### Task performance

- episode success rate；
- completion time；
- recovery success；
- slip / failure count；
- contact force overshoot。

### Timing

- semantic policy rate；
- tactile policy rate；
- controller / actuation rate；
- sensor rate；
- communication latency；
- end-to-end tactile reaction latency；
- semantic replanning latency。

### Compute

- GPU memory；
- forward latency；
- average semantic calls per episode；
- fast-policy compute；
- total energy / compute proxy（如果可测）。

### Communication

- shared token count；
- bytes / step；
- bytes / second；
- update frequency；
- token age / staleness。

### Robustness / generalization

- unseen friction；
- unseen object mass；
- unseen stiffness / compliance；
- unseen disturbance；
- timing jitter；
- communication drop；
- different controller；
- later: different robot embodiment。

---

## 12. Key ablations

### Token-count ablation

[
K = 0,1,4,16,64,	ext{full}
]

### Rate ablation

例如：

[
f_s in {2,5,10,20} mathrm{Hz}
]

[
f_t in {20,50,100,200} mathrm{Hz}
]

具体频率根据仿真和硬件可实现范围决定，不提前把数字写成结论。

### Delay / staleness ablation

系统性加入：

- observation delay；
- semantic delay；
- tactile delay；
- communication jitter。

### Direction ablation

[
Sightarrow T
quad 	ext{vs}quad
Sleftrightarrow T
]

### Representation ablation

- learned compressed tokens；
- full hidden states；
- hand-crafted summary；
- action-only communication。

---

## 13. Local compute plan

本方向第一阶段应尽量能在 RTX 5060 Laptop 上推进。

### Local 5060

适合：

- Franka simulator；
- asynchronous scheduler；
- small semantic encoder / frozen vision encoder；
- compact tactile policy；
- shared-token compressor；
- latency / jitter experiments；
- small-scale training；
- ablations；
- profiling。

不要求一开始训练大型 VLA。

### Larger lab compute

如果小模型和仿真实验得到明确现象，再使用课题组更大 GPU：

- 替换更强 semantic backbone；
- 扩大数据量；
- joint finetuning；
- larger multimodal baselines；
- reproduction / comparison with large VLA-style models。

算力扩展应该发生在假设已经有初步证据之后，而不是用大模型替代问题定义。

---

## 14. Relation to nearest work

T-Rex 等工作已经说明 tactile-reactive manipulation 可以采用不同时间尺度的 semantic / action 与 tactile processing。

因此本方向不能把“slow + fast 双流”本身直接宣称为 novelty。

需要进一步检索最近工作，重点确认：

- 是否已有系统研究 shared-token capacity；
- 是否已有 semantic-token staleness analysis；
- 是否已有双向 semantic–tactile communication；
- 是否已有 tactile-triggered semantic replanning；
- 是否已有 communication bandwidth / compute–performance trade-off；
- 是否已有跨 embodiment 的 multi-rate communication study。

潜在贡献应来自这些更具体的问题，而不是“做一个双流架构”。

参考入口：

- T-Rex project: https://tactile-reactive-dexterous.github.io/
- T-Rex paper: https://arxiv.org/abs/2606.17055

---

## 15. Falsification / stop conditions

下面结果都应该被允许出现：

### H1 被推翻

同步 multimodal policy 在相同 compute budget 下始终更好，异步没有明显收益。

### H2 被推翻

shared token 数量几乎不影响性能，communication bottleneck 不是关键变量。

### H3 被推翻

full latent sharing 同时获得最好性能与可接受延迟，没有观察到明显 coupling / bandwidth 问题。

### H4 被推翻

semantic context 对 fast tactile policy 很快失效，无法在低 semantic rate 下维持稳定控制。

### H5 被推翻

双向 tactile -> semantic communication 不提升任何任务性能，也不减少失败恢复时间。

如果主要假设被连续推翻，应停止为这个结构继续堆模型，而不是通过增加网络复杂度强行制造贡献。

---

## 16. Initial implementation milestones

### M0 — Infrastructure

- Franka simulation；
- contact-rich task；
- separate semantic / tactile clocks；
- timestamped shared-memory interface；
- latency logger；
- deterministic replay。

### M1 — Basic asynchronous baseline

实现：

[
semanticightarrow coarse action
]

[
tactileightarrow residual
]

先验证不同 rate 下系统能够稳定运行。

### M2 — Shared-token sweep

实现：

[
K = 0,1,4,16,64,	ext{full}
]

同时记录 success / latency / bandwidth。

### M3 — Staleness study

人为注入：

- delay；
- jitter；
- drop；
- stale context。

### M4 — Bidirectional / event-triggered communication

让 tactile branch 可以向 semantic branch 返回 learned token 或 replan event。

### M5 — Rich tactile + stronger semantic backbone

只有前面出现明确现象后再增加模型和传感器复杂度。

### M6 — Real-robot transfer

根据课题组实际可获得机械臂与 tactile hardware 进行迁移，不要求必须是真机 Franka。

---

## 17. One-sentence research version

> Study how asynchronous slow semantic and fast tactile policies should exchange information in contact-rich manipulation, treating communication capacity, staleness, direction, and update schedule as experimentally testable variables rather than fixed design assumptions.
