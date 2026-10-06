# Core Baselines and Positioning

状态：研究定位与实验基线设计，未形成项目结果。更新：2026-10-06。

## 核心问题

当前候选系统试图回答：

> **What is the right interface between dense high-rate touch and a slower manipulation policy?**

候选方法链：

$$
X_t^{\text{tactile}}
\xrightarrow{\phi_{\text{physics}}}
s_t^{\text{contact}}
\xrightarrow{\text{streaming memory}}
h_t
\rightarrow
\text{policy conditioning / action}.
$$

当前不把“FlexiTac + Mamba”本身视为 novelty。新颖性必须来自完整问题组合及实验证据：**dense spatial touch、compact physical state、persistent streaming memory、low-cost policy interface、closed-loop usefulness**。

## Core Triad

### 1. TacMamba — Long tactile memory baseline

来源：<https://arxiv.org/abs/2603.01700>

角色：**long-horizon streaming tactile memory**。

TacMamba 使用单点 1D force stream，通过 Mamba / SSM 维护递推 tactile history，并将压缩状态提供给低频 VLA。它是当前最直接的 memory baseline。

本项目与它的关键差异不是“用了更大的 Mamba”，而是 observation space：

$$
\text{TacMamba: }F_t^{1D}
\rightarrow
\text{streaming memory}
$$

vs.

$$
\text{Ours: }s_t^{\text{spatial contact}}
\rightarrow
\text{matched streaming memory}.
$$

正式实验应优先构造 **same temporal model, different tactile state** 的公平比较，隔离 spatial contact geometry 的价值。

不能写：

- “TacMamba 把高维触觉压成 1D”；
- “TacMamba 在所有 spatial task 上必然失败”。

更准确的表述是：其真实系统从传感端使用 1D force observation，因此不直接观测二维接触拓扑。

### 2. LeFlexiTac — Dense tactile-to-policy baseline

项目：*LeFlexiTac: Giving Robots a Sense of Touch*，Columbia RoboPIL / open-source project，2026。

项目页：<https://tna001-ai.github.io/LeFlexiTac/>

角色：**dense FlexiTac map → modern policy / VLA**。

LeFlexiTac 将 FlexiTac 集成进 LeRobot，并支持 ACT、Diffusion Policy、SmolVLA 和 π0.5。其 π0.5 路径使用 tactile tokens，将 dense tactile map 直接作为 learned policy input。

它为本项目提供最自然的直接问题：

> 为什么不直接把 FlexiTac tactile map token 化后送进 π0.5？

因此 Stage 1 / system-level experiments 应尽可能保留一个 **LeFlexiTac-style dense tactile token baseline**。

目前可防守的差异是：

- LeFlexiTac 直接保留 dense tactile map / token representation；
- 本项目研究显式低维 contact-state bottleneck；
- LeFlexiTac 没有提出独立的 high-rate persistent tactile-memory module；
- LeFlexiTac 在其 π0.5 实验设置中报告 action-expert-only / LoRA 微调效果不足，最终采用 full fine-tuning。

不能写：

- “π0.5 加 tactile 必然需要 full fine-tuning”；
- “LeFlexiTac 一定只运行在 10–30 Hz”；
- “32 tokens 一定冗余”。

这些都必须在本项目的真实实现和 latency / training-cost 测量后再判断。

### 3. Reactive Diffusion Policy (RDP) — Fast tactile control baseline

项目：<https://reactive-diffusion-policy.github.io/>

论文：<https://arxiv.org/abs/2503.02881>

角色：**slow-fast tactile-reactive control**。

RDP 将慢速 latent diffusion planning 与快速 tactile / force-conditioned action decoding结合，代表“如何把快速触觉反馈送入动作执行”的强基线。

本项目与 RDP 的预期差异：

$$
\text{RDP:}
\quad
z^{action}+f_t^{tactile}
\rightarrow
a_t
$$

强调快速 action reaction；

$$
\text{Ours:}
\quad
h_t=f(h_{t-1},s_t^{contact})
$$

强调 persistent physical interaction state / memory，然后再供策略使用。

可用的定位是：

> **RDP is primarily action-reactive; the proposed route is primarily state-memory-driven.**

但正式论文必须继续检查：RDP / ImplicitRDP 是否已经包含足以覆盖目标 claim 的历史状态。不能把“RDP 没有长时记忆”写成未经实验和源码核查的绝对事实。

## Supporting work

### FlexiTac / VTAP

- FlexiTac 提供二维法向接触阵列硬件基础；
- VTAP 证明 FlexiTac 类指尖触觉能进入真实机器人操作；
- 它们不是“learned baseline”的替代品，而是 hardware / geometry processing reference。

### T-Rex / TacForcing

- T-Rex：temporal force + spatial deformation、多速率 tactile refinement；
- TacForcing：execution-time tactile conditioning 与 action-generation timing；
- 二者用于约束本项目不能把“temporal tactile + async”或“execution-time touch”本身当 novelty。

### π0 / π0.5

π0 / π0.5 主要作为 **backbone / system-level visual-only ablation**，而不是 tactile representation baseline。

推荐系统级比较：

$$
\pi_{0.5}
$$

vs.

$$
\pi_{0.5}+\text{LeFlexiTac-style dense tactile}
$$

vs.

$$
\pi_{0.5}+\text{compact streaming contact memory}.
$$

## Baseline matrix

| 维度 | 代表工作 | 主要能力 | 本项目要回答的增量问题 |
| --- | --- | --- | --- |
| Long tactile memory | TacMamba | 1D force + streaming SSM memory | spatially structured contact state 是否在相同 temporal model 下提供额外价值？ |
| Dense tactile → VLA | LeFlexiTac | FlexiTac tactile map / tokens → LeRobot policies / π0.5 | compact physical bottleneck 能否以更低 training / inference cost 保留足够控制信息？ |
| Fast tactile control | RDP | slow planning + fast tactile-conditioned action decoding | persistent contact memory 是否能补充即时 reactive control？ |
| Supporting timing | TacForcing / T-Rex | execution-time tactile / multi-rate tactile expert | memory interface 是否在执行时真正产生闭环增量？ |

## 三个诊断任务

任务不是为了“刷 SOTA”，而是逐层检验 scientific claim。

### Task 1 — Blind bistable / click counting

问题：

> Physics contact state 是否保留 scalar-force baseline 擅长的时间事件与长时计数能力？

关键比较：

$$
\text{scalar force + matched memory}
\quad vs \quad
\text{physics state + matched memory}.
$$

期望判断改为预注册 non-inferiority，而不是模糊的“≈”：

$
Perf_{\text{physics}}
\ge
Perf_{\text{scalar}}-\delta,
\qquad \delta=3\%\ \text{relative margin}.
$

如果主指标是“越小越好”的误差，则按等价的相对退化定义转换。3% 是当前项目决策 margin，不宣称为领域通用阈值；只能在看 test 前基于任务量纲前瞻性修改。

同时报告 click detection latency、count error、long-horizon degradation。

visual-only policy 只作为 observability negative control。

### Task 2 — Contact redistribution under eccentric loading

问题：

> 当总接触强度近似不变，但空间接触分布发生改变时，spatial contact state 是否提供 scalar force 看不到的控制信息？

尽量设计：

$$
P_t \approx \text{constant},
$$

但：

$$
c_t,\Sigma_t
$$

显著变化。

关键比较应使用 **same temporal model**：

$$
P_{1:t}
\xrightarrow{\text{same memory}}
a_t
$$

vs.

$$
s_{1:t}^{phys}
\xrightarrow{\text{same memory}}
a_t.
$$

任务名称优先使用 **Contact Redistribution Recovery under Eccentric Loading** 或 **Spatial Contact Rebalancing under External Perturbation**。

不要把 CoP movement 未经验证地称为 incipient slip。

### Task 3 — History-dependent tactile search / insertion

问题：

> 当前 contact state 不足以确定动作时，spatial contact history 是否提供额外信息？

任务必须构造部分可观测性：

$$
s_t^A \approx s_t^B,
\qquad
s_{t-H:t}^A \neq s_{t-H:t}^B.
$$

关键消融：

$$
\text{physics state, no memory}
$$

vs.

$$
\text{physics state + memory}
$$

并与 learned compact latent / dense tactile baseline 比较。

不要预设某个 CNN 会因为异形孔“必然过拟合”。

## Stage-2P matched-memory probe

Stage 1 尚未正式通过时，允许一次低成本 probe 检验 **representation × memory** interaction。首轮固定比较：

$
\{P_t,\ [P_t,c_x,c_y],\ s_t^{phys11D},\ z_t^{learned11D}\}
\xrightarrow{\text{same streaming memory}}
h_t.
$

这一步不接 π0.5/VLA，也不做大规模 backbone search。它只回答“长历史是否暴露单帧读出没有显示的表示价值”，不能用更强网络掩盖当前表示负结果。

### Coupled spatiotemporal diagnostic target

最终最有价值的诊断任务不是单独“偏心检测”或单独“Mamba 压缩”，而是构造两个互补的可观测性缺口：

1. **Why Spatial**：总载荷近似匹配，但接触位置/形状/偏心状态不同，使 scalar history 缺少关键信息；
2. **Why Memory**：当前 compact spatial state 近似匹配，但此前的接触演化历史不同，使 memoryless predictor 无法唯一判断后续状态或所需动作。

候选物理过程包括 eccentric loading、rolling / redistribution、stick–slip-like transition 或 history-dependent insertion；具体任务必须由传感器可观测性和独立真值决定。FlexiTac 为法向阵列时，不预先把 CoP/shape 变化等同于 shear、slip 或 instability precursor。

关键消融至少包括：

- Spatial state + matched streaming memory；
- Spatial state + memoryless readout；
- Scalar + matched streaming memory；
- Learned compact state + matched streaming memory；
- 必要时 rich/raw spatial encoder + matched temporal backbone。

Raw/rich baseline 不能简单把 384D 直接塞进同尺寸 Mamba 后据此宣称高维输入“太慢”；应给它合理的空间 encoder，并分别报告 representation cost 与 matched temporal cost。

## Go / No-Go logic

理想但尚未成立的三层证据：

$$
\begin{aligned}
\text{Task 1: }&
Perf_{\text{physics}}
\approx
Perf_{\text{scalar}}
\\
\text{Task 2: }&
Perf_{\text{physics}}
>
Perf_{\text{scalar}}
\\
\text{Task 3: }&
Perf_{\text{physics+memory}}
>
Perf_{\text{physics-no-memory}}.
\end{aligned}
$$

如果成立，论文可以支持：

1. compact physical state 没有牺牲主要 temporal-event capability；
2. spatial contact geometry 在 scalar-force ambiguous conditions 下提供必要信息；
3. spatial contact history 在部分可观测接触任务中进一步产生闭环价值。

决策不采用机械的“任一层失败就整体 No-Go”：

- Task 1 只要求在预注册 margin 内 non-inferior；若轻微劣于 Scalar 但 Task 2 在冻结主指标上给出稳定的 spatial advantage，可以保留“空间几何有用”主线，但缩小 temporal-event claim；
- Task 2 失败而 Task 3 成立时，只能主张 history-dependent value，不能主张 current spatial state 的普遍优势；
- Task 3 失败时，不再扩大 persistent-memory claim；
- 若 Physics11D 与 Scalar+centroid + memory 近似，则缩小 descriptor，放弃“完整 11D 必要”的主张；
- 若 Learned11D 明显更优，则转向 learned/hybrid bottleneck。

另外预写 dense-path 分支：若 LeFlexiTac-style dense tactile path 在目标任务上达到或超过 compact path，且训练/推理/延迟成本在系统预算内可接受，则 compact-only 路线降级为 efficiency/hybrid 问题；优先考虑 low-rate dense token + high-rate compact state 的 hybrid，而不是强行证明 bottleneck 必须存在。

## 关于 novelty 的纪律

截至 2026-10-05，没有把“未检索到 FlexiTac + Mamba”写成 novelty 证明。

邻近工作已经覆盖：

- tactile + SSM / Mamba；
- dense tactile + policy / VLA；
- slow-fast tactile reactive control；
- tactile history modeling。

因此不使用：

> “first FlexiTac + Mamba”

作为主要贡献。

更强、更可防守的问题是：

> **Can dense high-rate tactile arrays be compressed into a physically structured streaming state that preserves spatial contact information, supports long-horizon memory, and interfaces efficiently with a slower manipulation policy?**
