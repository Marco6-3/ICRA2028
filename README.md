# ICRA2028 · 触觉接触状态与长时记忆

更新：2026-10-05。状态：研究路线与首个表示实验设计；尚未运行本路线的实验。

## 一条长期主线

**研究机器人应当保留哪些接触信息，以及如何把这些信息压缩成能够支持长时间、低延迟闭环控制的内部状态。**

当前最高优先级候选问题：

> **What is the minimal tactile state that preserves control-relevant spatial contact information over long horizons?**

当前候选切入点是 **Physics-Guided Contact-State Compression**：先用 FlexiTac 一类二维触觉阵列研究“高维接触场 → 低维、物理可解释 contact state”的信息保真与失败边界；只有这个 bottleneck 经验证成立后，再研究 TacMamba / Mamba 一类流式长历史模型。这样把“表示是否足够”与“历史如何记忆”分开验证。

T-Rex 提供 temporal / spatial tactile representation 与快慢触觉控制参照；TacMamba 提供低维流式长历史压缩参照；TacForcing 提供 execution-time tactile conditioning 参照。它们是方法来源和强基线，不作为模块拼装清单。

## 当前候选证据链

Raw tactile field → Physics contact state → Representation evidence → Temporal memory → Closed-loop control

对应三个阶段：

1. **Stage 0 — Physics sanity check**：不训练网络，检查 descriptor 的噪声、漂移、重复性、事件响应和真实延迟。
2. **Stage 1 — Representation bottleneck**：比较 Raw / Physics 11D / Learned 11D 的数据效率、OOD、延迟和信息丢失。
3. **Stage 2 — Temporal memory**：只有前两阶段成立后，才比较 physics-state history 与 learned/raw temporal representation；Mamba 是强候选但不预设为最终答案。

详细定义见 [Physics-Guided Contact-State Compression](research/PHYSICS_BOTTLENECK.md)。

## 目标年份

以 **2027 年完成研究并投稿 ICRA 2028** 为目标，而非承诺录用。ICRA 2028 的正式截止、页数与投稿细则以最终 CFP / PaperPlaza 为准；内部暂按 **2027-07-31 完整初稿** 留出缓冲。来源与日期冲突见 [路线与时间表](ROADMAP.md)。

## 导航

| 文档 | 用途 |
| --- | --- |
| [研究主线](research/CORE.md) | 核心问题、候选假设、方法边界与可能的论文贡献 |
| [Physics Bottleneck](research/PHYSICS_BOTTLENECK.md) | 当前最高优先级候选：FlexiTac contact state、预处理、Stage 0/1/2 与停止条件 |
| [实验方案](research/EXPERIMENTS.md) | 公平比较、因果时间约束、指标和投稿前证据门槛 |
| [研究路线](ROADMAP.md) | 从当前小实验到 2027 年投稿准备 |
| [论文地图](papers/README.md) | 核心与邻近文献、来源、已读范围和与本路线的关系 |
| [T-Rex](papers/T_REX.md) | 快慢专家、时序触觉编码与去噪过程 |
| [TacMamba](papers/TACMAMBA.md) | 流式历史压缩、1D force sensing 与证据边界 |
| [TacForcing](papers/TACFORCING.md) | 执行时触觉、EATA，以及对既往理解的纠正 |
| [协作规则](AGENTS.md) | 维护假设、证据和实验记录约定 |

## 现在先做什么

1. 固定 FlexiTac 硬件 / 固件版本与真实数据格式，实测采样率、传输和 preprocessing latency；不要沿用文献数字代替本机测量。
2. 实现 research/PHYSICS_BOTTLENECK.md 中的 11D contact descriptor，并完成 Stage 0：contact → load → redistribution / roll → release 的重复数据。
3. 只有 descriptor 数值稳定、事件响应可重复后，进入 Stage 1：Raw vs Physics 11D vs Learned 11D。
4. Stage 1 若显示 physics bottleneck 在 compactness / data efficiency / OOD / latency 中至少具有清晰价值，再进入 temporal memory；否则停止或修改 bottleneck，而不是直接堆 Mamba。

**不把好看的相平面图当成论文证据，不把 Mamba、FlexiTac 或“physics”本身当 novelty。** 最终贡献必须来自可重复的增量证据和明确失败边界。

本仓库管理研究问题与证据，不重建通用 infra。
