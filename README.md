# ICRA2028 · 触觉流形接触场、多模态防坍缩自监督与 3-DoF 空间反射控制

> **最新突破与研究范围升级（2026-11-03，Stage 3S–V2.1）**：根据前期实验暴露的三个物理瓶颈（滑动窗口 `MCF` 抹除慢变剪切、1D 夹爪收紧 $\Delta g$ 在 `0.20 N +Fx` 下受限、高预紧静态阈值易零载误触发），项目研究范围正式升级为三大核心机制：
> 1. **参考帧锁定的双指预紧不变流形接触场（`BPI-MCF`，32D）**：锁定初始接触参考帧（$k_{\text{base}}=5$）+ 单指预紧归一化（$f_{i,0}$）+ 双指对称/反对称正交分解与偏心力矩偶极。8 阶段消融证实切向方向辨识率在开发集达 **`97.5%`**（Scalar `25.0%`，Standard MCF `46.7%`），在 12 组独立盲测集 `Fresh12` 上从 `33.9%` 跃升至 **`65.7%`**（`b0+b1` 达 **`68.1%`**），在 36 工况 OOD `pilot_tangent_02` 上达 **`47.2%`**。
> 2. **跨模态接触力螺旋与剪切方向自监督（防止多模态表征坍缩）**：全模型 26 组权重/偏置张量梯度审计 100% 严格大于零（`all_positive = True`）；在公平的冻结隐空间线性探针（`FrozenProbeDirAcc`，$\lambda=1.0$）下将方向可分性从朴素融合 `B1` 的 `85.8%` 提升至 **`94.4%`**（`B5`），遮挡反转跟踪误差 `RevNeg(Occ)` 从 `0.9034`（纯视觉）和 `0.3490`（`B1`）降至 **`0.2202`**。
> 3. **因果偶极新息能量门控 3-DoF 空间反射策略（$[\Delta x_{\text{wrist}}, \Delta y_{\text{wrist}}, \Delta g_{\text{grip}}]$）**：在 **0% `Stage 3S` 混杂** 的 Isaac Lab 7 控制器闭环对比中，纯 V2.1 神经策略 `6_v2_1_pure_bpi_3dof_no_stage3s` 受载四向平均滑移降至 **`0.7795 mm`**（较开环 `8.2011 mm` 降低 **`90.5%`**，较原 `Stage 3S` 1D 基线 `1.2303 mm` 降低 **`36.6%`**，`+Fx` 最差滑移由 `3.7210 mm` 降至 **`2.1754 mm`**），且预紧静默段抖动为 **`0.0000 mm`**（无门控 `B3` 为 `1.4605 mm`）。在 3 阶段长时序操作基准（预紧 $\to$ 视觉遮挡受扰搬运 $\to$ 接触反转插孔对齐）中末端最终对齐误差降至 **`0.9627 mm`**。
>
> 详见 **[V2.1 完整实验归档](experiments/v2_1_bpi_mcf_3dof_reflex/README.md)**、**[2026-11-03 进展记录](research/STAGE3S_V2_1_BPI_MCF_3DOF_REFLEX_20261103.md)** 与 **[更新后的研究范围修订稿](research/ICRA2028_MINIMAL_PAPER_FREEZE.md)**。

> **前序仿真成果（2026-10-15 / 2026-10-13）**：
> - **2026-10-15 等预紧握持与因果脉冲触发**：完成 1190 个物理试次，拉平 Scalar/MCF 静息握力并检验早期脉冲触发，见 **[等预紧归档](experiments/stage3s_alignment/README.md)**。
> - **2026-10-13 Stage 3S 1D 闭环**：归一化修复、无支撑受控加载及触觉 GRU→Δg 闭环已完成。0.18 N / 15 mm 偏心的四方向，MCF Reflex 最坏相对滑移 **1.013 mm**，相比 High-Grip 平均法向力积分降低 **33.9%**；0.20 N +Fx 的未达标结果、Scalar 对照及全部记录一并保留。见 **[Stage 3S 实验归档](experiments/stage3s_isaac/README.md)** 与 [进展记录](research/STAGE3S_SIM_REFLEX_20261013.md)。

> **历史研究范围冻结记录（2026-10-10，已于 2026-11-03 修订升级）**：第一篇论文早期只研究“法向阵列的紧凑空间接触状态能否改善偏心扰动下的闭环控制”，入口为 **[研究方向修订与范围冻结稿](research/ICRA2028_MINIMAL_PAPER_FREEZE.md)**。下方历史记录保留项目演进脉络。


> **当前最高优先级（2026-10-07）：进入真机 Stage 3A Controlled Tactile Reflex。** 先不接 VLA。优先做 Anti-Tilt / eccentric disturbance，再做 Anti-Slip；主比较为 Fixed、Scalar Reflex、MCF Reflex。现有 Stage 2P checkpoint 是状态重构模型，不直接视为控制器。详见 [Stage 3A 真机方案](research/STAGE3A_REAL_ROBOT.md) 与 [真机执行协议](experiments/real_robot_reflex/PROTOCOL.md)。只有受控闭环实验出现可信增益后，才进入 Stage 3B 的 scripted / learned policy / VLA plug-and-play 展示。


更新：2026-10-07。

**已完成离线实验：Mamba / GRU 对照。** 按用户要求，沿用 TaF 当前状态重构流程，比较 Scalar / MCF、T=50/100、三种子与约48k参数，并测量 batch1 流式延迟。详见 [Mamba 架构实验](research/STAGE2P_MAMBA.md)。

**历史推进记录：Stage 1 条件性通过并封存，随后完成 Stage 2P Now-casting。** 按用户纠偏，将无动作条件的未来力矩目标改为利用 1、20、50 帧历史重构当前剪切力和侧向力矩 `[Fx,Fy,Mx,My]`，比较相同参数量两层 GRU 的 Scalar 与 MCF。详见 [当前重构实验](research/STAGE2P_NOWCAST.md)。[首轮未来预测实验](research/STAGE2P.md) 和下文 Stage 1 方案保留作为历史记录。

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
| [Stage 3S–V2.1 实验归档](experiments/v2_1_bpi_mcf_3dof_reflex/README.md) | **最新（2026-11-03）**：`BPI-MCF` 8 阶段消融、多模态防坍缩自监督、Isaac Lab 7 控制器 3-DoF 闭环与 3 阶段长时序基准 |
| [2026-11-03 V2.1 进展记录](research/STAGE3S_V2_1_BPI_MCF_3DOF_REFLEX_20261103.md) | 三大创新点数学定义、审计问题清零与 2×A100/DGX Spark/真机推进方案 |
| [研究方向修订与冻结稿](research/ICRA2028_MINIMAL_PAPER_FREEZE.md) | 2026-11-03 范围升级说明与 2026-10-10 历史冻结稿 |
| [MCF 物理可观测性边界辨析](research/MCF_OBSERVABILITY_LIMITS_DISCUSSION.md) | 单指零阶/一阶矩的旋转不可辨识反例与双指偶极扩展的理论背景 |
| [Stage 3S 等预紧与触发归档](experiments/stage3s_alignment/README.md) | 2026-10-15 等预紧握持、成本—安全关系与弱触觉脉冲触发归档 |
| [Stage 3S 1D 闭环归档](experiments/stage3s_isaac/README.md) | 2026-10-13 无支撑加载与 1D $\Delta g$ 闭环实验归档 |
| [Minimalist Contact Flow](research/MINIMALIST_CONTACT_FLOW.md) | 早期 Stage 1 实验方案 |
| [离线成果与决策](research/OFFLINE_RESULTS.md) | 已完成 11D 离线实验和结果 |
| [研究主线](research/CORE.md) | 长期研究问题 |
| [实验方案](research/EXPERIMENTS.md) | 实验设计与长期验证 |
| [Core Baselines](research/BASELINES.md) | TacMamba / LeFlexiTac / RDP 等参考 |
| [研究路线](ROADMAP.md) | 后续推进顺序 |
| [协作规则](AGENTS.md) | Agent 执行原则 |

本仓库中的候选假设可以被实验支持、修改或推翻；不在实验前把候选结论写成既定事实。
