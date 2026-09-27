# I001 — High-Frequency Tactile Feedback with Temporal Context and In-Context Adaptation

**Status:** Candidate research direction  
**Theme:** tactile manipulation / reactive control / temporal representation / in-context adaptation  
**Goal:** 研究高频触觉策略到底需要什么“历史”，以及这种历史能否从单纯状态估计进一步变成对未知接触动力学的在线适应。

---

## 1. Core question

高频触觉反馈是否应该同时利用两种时间尺度的信息：

1. **短期 history**：估计“现在发生了什么”，例如是否正在滑移、剪切是否持续增长、接触是否正在失稳；
2. **更长的 interaction context**：推断“当前物体 / 接触系统是什么样的”，例如摩擦、顺应性、局部接触动力学，从而改变之后的反馈策略。

最终希望检验：

> Can a high-frequency tactile policy infer contact state from recent history and adapt to contact dynamics in-context from its own interaction, without online gradient updates?

这里的 in-context adaptation 是待验证现象，不预设一定会出现。

---

## 2. 为什么单帧触觉可能不够

一个瞬时观测 (T_t) 往往存在歧义。

例如当前剪切相关信号处于同一个数值，但过去几十毫秒可能分别是：

```text
A: low -> medium -> high      (正在接近失稳)
B: high -> medium -> stable   (正在恢复)
C: action changed -> signal changed
                               (机器人自己的动作造成的正常响应)
```

因此更合理的 fast policy 输入不是：

[
a_t = pi(T_t)
]

而是：

[
delta a_t =
pi_{fast}
left(
T_{t-H:t},
q_{t-H:t},
dot q_{t-H:t},
a_{t-H:t-1},
a_t^{nominal}
ight)
]

这里的关键不是 Transformer 本身，而是 **history-conditioned decision**。

---

## 3. 三个需要严格区分的层次

### Level 1 — Instantaneous reactive control

[
T_t ightarrow delta a_t
]

只根据当前触觉修正动作。

适合作为最弱 baseline。

### Level 2 — Temporal reactive control

[
(T,a,q,dot q)_{t-H:t}
ightarrow
z_t^{contact}
ightarrow
delta a_t
]

利用最近几十到几百毫秒的历史估计当前 contact state。

这属于 temporal memory / history-conditioned policy，**不能因为用了历史就直接称为 ICL**。

### Level 3 — In-context adaptation

模型利用更长的当前 episode 交互：

[
mathcal C_t =
{T_	au,q_	au,dot q_	au,a_	au}_{	au=0}^{t}
]

形成某种隐式环境表征：

[
z_t^{env} = f_{context}(mathcal C_t)
]

并在**不更新网络权重**的情况下改变后续控制：

[
delta a_t =
g(
z_t^{contact},
z_t^{env},
a_t^{nominal}
)
]

只有当过去 interaction 改变了之后面对相似触觉状态时的动作选择，才更接近这里所说的 in-context adaptation / in-context system identification。

---

## 4. Proposed multi-timescale architecture

最初不做 monolithic multimodal model，而是显式分离任务层和接触层：

```text
RGB / language / task state
          │
          ▼
  Slow semantic policy
          │
     nominal action
          │
          ▼
┌───────────────────────────────┐
│       Fast tactile loop       │
│                               │
│ tactile history ───────────┐  │
│ action history  ───────────┼─>│ short-context encoder
│ proprioception  ───────────┘  │        │
│                               │        ▼
│ episode interaction ──────────> long-context encoder
│                                        │
│        z_contact + z_env + nominal     │
│                    │                   │
│                    ▼                   │
│             residual policy            │
└────────────────────┬───────────────────┘
                     ▼
          a = a_nominal + delta_a
```

概念上：

[
z_t^{contact}
=
f_{short}(C_{t-H:t})
]

回答“现在接触状态是什么”。

[
z_t^{env}
=
f_{long}(C_{0:t})
]

回答“这个接触系统过去表现出什么规律”。

[
a_t
=
a_t^{nominal}
+
pi_{residual}
(
z_t^{contact},
z_t^{env},
a_t^{nominal}
)
]

### 不提前固定模型家族

short / long encoder 可以分别测试：

- MLP + explicit finite differences
- GRU / LSTM
- causal Transformer
- Mamba / state-space model
- event-based / SNN variant

研究问题优先于网络名字。模型只在它能区分假设时加入。

---

## 5. 为什么 context 中应该包含 action history

只有 tactile history：

[
T_{t-H:t}
]

最多告诉模型“触觉发生了变化”。

加入 action history：

[
(a,T)_{t-H:t}
]

模型才有机会学习：

[
	ext{action}
ightarrow
	ext{contact consequence}
]

例如同样的 shear increase：

- 如果刚刚主动横移，这是预期响应；
- 如果动作基本不变却 shear 突升，可能代表外部扰动或 slip onset。

因此一个核心消融必须是：

[
T history
quad vs quad
T + action
quad vs quad
T + action + proprioception
]

---

## 6. Mechanics-informed data design

这个方向**不要求先建立完整机器人动力学模型**。更有价值的是用接触力学指导“采什么、怎么标、如何设计变化”。

### 候选接触变量

[
z^{physics}
=
[
F_n,
F_t,
	au,
slip,
contact patch,
deformation,
vibration,
dot F,
dot T,
dots
]
]

其中优先考虑：

| Variable | Why it may matter |
| --- | --- |
| Normal force (F_n) | grip / contact loading |
| Tangential / shear force (F_t) | lateral interaction and slip tendency |
| (F_t/F_n) or related ratio | friction-margin proxy under suitable assumptions |
| slip / incipient slip | directly relevant to reactive correction |
| contact location / patch | distinguishes where contact is occurring |
| torque | useful in twisting / insertion / rotational slip |
| deformation history | important for compliant contacts |
| high-frequency vibration | potential cue for fast contact events |
| temporal derivatives | distinguishes static state from rapidly changing state |

这些量不一定全部作为 policy 输入。它们可以是：

- privileged ground truth；
- auxiliary prediction target；
- representation probe target；
- experimental control variable；
- analysis variable。

真正重要性必须通过 intervention / ablation / closed-loop behavior 验证，而不是由力学直觉直接宣布。

---

## 7. Recommended data schema

如果硬件允许，采集时尽量保存原始同步数据：

```text
timestamp
raw tactile
robot q
robot dq
commanded action
executed action / controller state
RGB (optional for fast loop, useful for slow policy)
wrist F/T (optional privileged measurement)
contact-event labels or derived annotations
object / material / trial metadata
```

关键要求：

- 保存原始时间戳，不只保存重采样后的张量；
- 区分 sensor sampling rate、message arrival rate、policy rate 和 actuator update rate；
- 保存 failure / slip / unstable contact，不只保存成功轨迹；
- 若存在 F/T 传感器，优先把它作为监督和分析工具，而不是默认部署依赖；
- 记录传感器重装、标定、漂移与不同采集批次。

---

## 8. Hypotheses

### H1 — Short temporal context is necessary

在 contact-rich task 中，固定当前观测质量后，短期触觉历史相对单帧输入能改善：

- slip / instability detection；
- correction timing；
- closed-loop success；
- false correction rate。

若没有稳定提升，则“高频触觉必须依赖 temporal memory”这一强说法不成立。

### H2 — Action-conditioned history is more informative than tactile-only history

[
(T,a,q)_{t-H:t}
]

应当比

[
T_{t-H:t}
]

更好地区分“机器人主动造成的触觉变化”和“环境 / 接触状态变化”。

### H3 — Longer interaction context can support adaptation

面对未见摩擦、顺应性、重量或 contact response，模型经历短暂交互后，如果保留 episode context，后续控制表现相对 reset-context baseline 改善。

这才是 ICL 部分最关键的证据。

### H4 — Two-timescale decomposition is useful

slow semantic policy + fast tactile residual policy 在保持任务语义能力的同时，可以比统一低频策略更快地处理局部 contact event。

这不是预设结论，需要和统一 policy / matched-compute baseline 比较。

---

## 9. Minimal experimental ladder

不要一开始就做完整 VLA + 真机系统。

### Stage 0 — Signal and latency characterization

先回答：

- tactile 实际有效采样率是多少；
- signal-to-noise / drift / hysteresis 情况如何；
- action command 到真实执行的延迟是多少；
- sensor -> policy -> action 的 end-to-end latency 是多少；
- 高频信息是否真的存在且能被执行器利用。

如果这一层不成立，后面“high-frequency policy”的结论没有基础。

### Stage 1 — Offline temporal prediction / state estimation

最小任务可以是：

- slip onset detection；
- contact-state classification；
- short-horizon tactile/contact prediction；
- privileged (F_n/F_t) regression 或 probing。

比较：

```text
single frame
short tactile history
short tactile + action history
short tactile + action + proprio history
```

目的不是把 offline accuracy 当论文结果，而是确认历史中是否存在可利用信息。

### Stage 2 — Closed-loop fast residual control

选择一个简单、容易反复执行的 contact-rich task，例如：

- grasp + controlled slip disturbance；
- object lift with variable friction / load；
- simple insertion / alignment with lateral contact。

先只测试：

[
a = a_{nominal} + delta a_{tactile}
]

### Stage 3 — Context-length study

系统改变：

[
H = 0, 20ms, 50ms, 100ms, 200ms,dots
]

研究性能是否存在明显的时间尺度，而不是默认“上下文越长越好”。

### Stage 4 — In-context adaptation test

训练时覆盖一组 contact dynamics，测试时使用 held-out 条件。

关键比较：

1. context 每步保留；
2. context 周期性 reset；
3. 只保留短期 history；
4. 保留完整 episode context；
5. oracle physical-property input；
6. 若可行，显式 system-identification baseline。

真正的问题是：

> 同样的当前触觉状态，在之前经历不同 interaction 后，policy 是否会作出系统性不同且更合适的动作？

### Stage 5 — Multi-rate semantic + tactile policy

在 fast loop 本身成立后，再加入 slow semantic policy。

不要反过来先做大模型，再尝试解释它有没有利用触觉。

---

## 10. Simulation and real-robot strategy

### Simulation

可以先在仿真机械臂上研究因果关系和可控参数。Franka 是一个自然的候选平台，但不是强制限定。

仿真尤其适合系统扫描：

- friction；
- object mass；
- compliance；
- controller delay；
- tactile noise；
- sensor latency；
- disturbance timing。

但仿真触觉不能自动替代真机结论。高频振动、橡胶迟滞、传感器漂移、结构共振等真实效应可能很难可信模拟。

### Real robot

后续真机不要求必须是 Franka。核心接口是：

- 可获取高频 tactile；
- 能读 proprioception；
- 能记录 action / controller state；
- 有足够快且安全的局部修正接口。

算法层尽量避免绑定特定机器人本体。

---

## 11. Baselines and ablations

至少考虑：

| Baseline / ablation | What it tests |
| --- | --- |
| current tactile only | 是否真的需要 history |
| tactile history only | tactile temporal information 本身 |
| tactile + action history | action consequence 是否关键 |
| tactile + action + proprio | 本体状态是否补充可观测性 |
| short context only | 状态估计能力 |
| short + episode context | 是否存在 adaptation |
| episode context reset | 排除“只是模型更大 / 输入更多” |
| oracle physical variables | 学到的 context 距离理想物理状态还有多少差距 |
| low-rate unified policy | fast loop 是否真正带来反应收益 |
| matched-compute model | 排除纯参数量 / 算力优势 |

如果比较 Transformer、Mamba、GRU、SNN，尽量匹配参数量、输入窗口和部署硬件。

---

## 12. Evaluation

不要只看 success rate。

### Task metrics

- episode success rate；
- slip recovery rate；
- insertion / alignment success；
- disturbance robustness；
- held-out object / material performance。

### Reaction metrics

- event -> detection latency；
- event -> policy output latency；
- event -> physical correction onset；
- total end-to-end reaction time；
- missed events；
- unnecessary / false corrections。

### Physical metrics

- slip distance；
- peak normal force；
- peak shear / torque；
- object displacement；
- contact stability；
- unnecessary squeezing / force overshoot。

### Learning metrics

- context length sensitivity；
- held-out dynamics generalization；
- adaptation curve within one episode；
- latent probe / auxiliary prediction only as supporting diagnostics。

---

## 13. What would count as evidence for in-context adaptation?

仅仅把 1 秒历史送进 Transformer 不算。

至少希望观察到：

1. 当前 observation 近似相同；
2. 前面 interaction history 不同；
3. policy 输出出现系统性差异；
4. 这种差异与当前隐藏 contact dynamics 相匹配；
5. 权重没有在线更新；
6. reset / shuffled context 会破坏这种收益。

可以进一步构造 paired trials：

```text
same object pose
similar current tactile state
different preceding probe interaction
-> different residual action
-> different closed-loop outcome
```

这比只画 attention map 更接近因果证据。

---

## 14. Falsification / stop conditions

出现下面情况时应缩小或放弃相应 claim：

- 单帧与短历史在多个任务上无稳定差异；
- 加 action history 没有可复现收益；
-所谓“高频收益”实际来自更高 actuator rate，而不是 tactile information；
- episode context 只在训练分布内有效，对 held-out contact dynamics 无适应；
- reset context 不影响性能，说明长期 context 可能没被使用；
- 真实系统 latency 已经超过目标 contact event 时间尺度；
- 仿真结果依赖不可实现的 tactile / actuator assumptions；
- 一个简单 explicit estimator + controller 已经同样好，则需要重新定义学习方法的价值。

---

## 15. Possible contribution paths

下面只是可能形成论文贡献的路线，不是新颖性声明。

### Path A — What history does tactile control actually need?

核心贡献是系统回答 short temporal context、action history、context length 与闭环触觉控制之间的关系。

### Path B — In-context contact adaptation

核心贡献是：

[
	ext{interaction history}
ightarrow
	ext{implicit contact dynamics}
ightarrow
	ext{adaptive control}
]

并证明不需要 online finetuning。

### Path C — Multi-rate embodied control

核心贡献是 slow semantic policy 与 fast tactile residual policy 的接口、异步状态共享以及 end-to-end reaction benefit。

这三个方向可以共享基础设施，但未必应该写成同一篇工作。

---

## 16. Immediate next questions

在实现前优先回答：

- [ ] 选择一个最简单的 contact event：slip、impact、insertion contact 还是其他？
- [ ] 当前可获得的 tactile sensor 原始频率和有效 bandwidth 是多少？
- [ ] actuator / controller 的真实可修改频率是多少？
- [ ] 哪些物理量可以作为 privileged ground truth？
- [ ] 仿真阶段使用什么 tactile observation，而不制造过度理想化结论？
- [ ] short history 的最小时间尺度应该从多少开始扫描？
- [ ] episode-level context 如何在训练时构造，才能避免模型仅记住 object identity？
- [ ] 如何设计 paired / reset / shuffled-context 实验验证真正的 context dependence？
- [ ] 相关工作是否已经直接研究“tactile in-context system identification / adaptation”？需要单独做最新文献核查。

---

## 17. Current research stance

目前最值得先验证的不是“是不是应该做 ICL 模型”，而是更基础的两步：

[
oxed{
	ext{Does short tactile-action history improve contact-state estimation and control?}
}
]

如果成立，再研究：

[
oxed{
	ext{Can longer interaction context adapt the same policy to unseen contact dynamics?}
}
]

这样可以把 **触觉、接触力学、高频反馈、时序建模、ICL** 连成一条可逐层证伪的研究路线，而不是一开始就把它们堆进一个大模型。
