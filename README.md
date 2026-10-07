# ICRA2028 · 触觉接触状态与时序记忆

更新：2026-10-07。

**当前新增实验：离线 Mamba / GRU 对照。** 按用户要求，沿用 TaF 当前状态重构流程，比较 Scalar / MCF、T=50/100、三种子与约48k参数，并测量 batch1 流式延迟。详见 [Mamba 架构实验](research/STAGE2P_MAMBA.md)。

**最新决策：Stage 1 条件性通过并封存，当前推进 Stage 2P Now-casting。** 按用户纠偏，将无动作条件的未来力矩目标改为利用 1、20、50 帧历史重构当前剪切力和侧向力矩 `[Fx,Fy,Mx,My]`，比较相同参数量两层 GRU 的 Scalar 与 MCF。详见 [当前重构实验](research/STAGE2P_NOWCAST.md)。[首轮未来预测实验](research/STAGE2P.md) 和下文 Stage 1 方案保留作为历史记录。

## 已封存 Stage 1

之前的 Stage 1 主要尝试验证完整静态 11D contact descriptor，包括二阶矩等空间统计量。已有离线实验没有得到理想结果，因此当前不继续在原有静态 11D 上死磕，而是进行一次特征和验证任务的调整。

新的候选表示是 **Minimalist Contact Flow**：

```text
s_t = [f_N, CoP_x, CoP_y, A]
Δs_t = s_t - s_{t-1}
MCF_t = [s_t, Δs_t]
```

四个状态量及其一阶差分全部保留时为 8D。

当前待验证假设：

> **CoP + 差分是否能够在 Contact-active Subsets 上显著优于纯 Scalar，并与 Dense Raw Tokens / Raw tactile representation 表现相当？**

这只是实验假设，结论等待实验结果。

详见 [Minimalist Contact Flow](research/MINIMALIST_CONTACT_FLOW.md)。

## 为什么更换 Stage 1 指标

此前使用平滑轨迹中的开环 Action MSE，触觉信息可能被大量普通、稳定运动稀释。

因此下一轮不需要先采新数据，而是直接复用已有 LeFlexiTac / TaF 数据进行切片过滤：

- 剔除完全未接触（`f_N ≈ 0`）的空载段，以及静态死锁且无外力扰动的平淡段。
- 保留接触 / 脱开瞬间（`abs(Δf_N) > ε_f`）、受力中心移动 / 倾覆阶段（`norm(ΔCoP, 2) > ε_p`）、面积收缩 / 膨胀（`abs(ΔA) > ε_A`）；任一条件满足即保留。
- 脱开事件和仅面积变化的片段优先保留，避免被空载或静态判据误删。

两项核心评估任务：

1. **任务 A（TaF）：预测外力矩变化量 `Δτ_ext`（Torque Delta）**。比较 Scalar vs MCF vs Dense Raw Tokens。
2. **任务 B（LeFlexiTac）：接触状态跳变检测**。检验 MCF 是否能准确识别突然切换（微滑移或失稳前兆）。

任务 A 的预期判定：Scalar 由于丢失空间偏心信息，预测力矩变化误差必然显著偏高；MCF 凭借 CoP 及其差分，以 8 维捕捉力矩偏置，误差显著低于 Scalar，且逼近 Dense Tokens。这是保留的实验假设，实验验证后再修正。

## Stage 1 核心比较

| Representation | 作用 |
| --- | --- |
| Scalar | 纯标量基线 |
| **Minimalist Contact Flow** | 当前主要候选 |
| Dense Raw Tokens / Raw tactile representation | 高维触觉参照 |

Static CoP、Physics11D 等可以继续作为消融和历史结果保留。

## Stage 1 希望回答的问题

如果实验得到：

**MCF > Scalar，并且 MCF ≈ Dense Raw**

那么可以进一步形成候选观点：

> **高阶空间矩可能是冗余的，核心是 CoP 及其动态演化。**

如果没有得到这一结果，则根据实际实验结果重新分析，不提前规定后续结论。

## 后续

只有完成这一轮 Stage 1 后，再根据结果决定是否以及如何进入 Mamba / streaming temporal memory。

当前顺序：

**Static 11D 收尾 → Minimalist Contact Flow → Contact-active validation → 根据结果决定下一步。**

## 导航

| 文档 | 用途 |
| --- | --- |
| [Minimalist Contact Flow](research/MINIMALIST_CONTACT_FLOW.md) | 当前 Stage 1 实验方案 |
| [离线成果与决策](research/OFFLINE_RESULTS.md) | 已完成 11D 离线实验和结果 |
| [研究主线](research/CORE.md) | 长期研究问题 |
| [实验方案](research/EXPERIMENTS.md) | 实验设计与长期验证 |
| [Core Baselines](research/BASELINES.md) | TacMamba / LeFlexiTac / RDP 等参考 |
| [研究路线](ROADMAP.md) | 后续推进顺序 |
| [协作规则](AGENTS.md) | Agent 执行原则 |

本仓库中的候选假设可以被实验支持、修改或推翻；不在实验前把候选结论写成既定事实。
