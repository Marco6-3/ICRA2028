# ICRA2028 研究路线

> **2026-11-21 最新研究路线（Stage 6 / `V4 PAEC` — 路线 1：物理锚定接触状态压缩）**：
> 项目已完成研究转型（不再购买 FlexiTac，与 `NewGripper` 硬件/无本体采集论文形成双论文联动；本仓库聚焦算法主论文 Paper B）。
> - **已验证进展（离线真机公开数据，严格 FP32 + episode 划分 + 权重冻结审计）**：
>   1. **Round-1 音频抽象 JEPA 证伪归档**（`ManiWAV/flip` 283 eps，[ROUND1_AUDIT_SUMMARY.md](experiments/v3_caec_jepa_mvp/outputs/maniwav_flip_v3/ROUND1_AUDIT_SUMMARY.md)）：实证抽象隐空间 JEPA + VICReg 在几百条轨迹上发生高维噪声膨胀（有效秩 `33.35`）导致下游少样本 BC 劣于低秩 `MAE`（`5.71`）。
>   2. **V4 `PAEC` 双任务域内复现**（RSS-2025 RDP `peel_v3` 60 eps + `wipe_v1.5` 80 eps，37D = 6D Wrench + 1D 夹爪力 + 30D 双 GelSight，[peel 报告](experiments/v4_paec_contact_compression/outputs/rdp_peel_v4/RDP_PEEL_V4_RESULTS.md) / [wipe 报告](experiments/v4_paec_contact_compression/outputs/rdp_wipe_v4/RDP_WIPE_V4_RESULTS.md)）：`B1_RawFT_End2End`、`B2_MAE` 与 `B3_LatentJEPA`（有效秩 `40.21 / 41.79`）全部劣化下游 BC；唯有 **`PAEC`**（$d_z=16$，有效秩 `9.73–10.98`）稳定反超纯本体，并在前向 6D 力预测（Clean `0.4620 / 0.6343`、零漂 OOD `0.3759 / 0.5043`）与反转段夺冠。
>   3. **双向 100% 冻结零样本跨任务迁移（`wipe ↔ peel`）**（[跨任务迁移报告](experiments/v4_paec_contact_compression/outputs/RDP_CROSS_TASK_V4_RESULTS.md)）：`wipe → peel` 纯零样本（`src_norm`）30% BC 达 **`0.58805`**（超越 `B0 0.59301` 与域内 `0.59032`），零样本反转段 6D 力预测 $R^2 = +0.2626$ 追平域内；反向不对称迁移 `peel → wipe` 中 `Xfer_PAEC_z16` 在 `src_norm` 10%（`0.56327`）/ 100%（`0.39521`，零漂 `0.39711` 仍超 `B0 0.40057`）及 `tgt_norm` 100%（`0.39557`，唯一战胜 `B0`）全面胜出，消融证实 `b+g` 与 `DriftAug` 缺一不可、且跨任务应直接迁移 16D 状态 $z_t$。
> - **当前下一阶段待执行计划（未运行，严禁提前当成结论）**：
>   1. **Step 1 — 仿真闭环与独立外扰/相变惊奇度真值验证**：在物理闭环仿真（如 MuJoCo / Isaac Lab 的剥离/滑移/卡扣/偏心抗扰任务）中，将离线 BC MSE 升级为**闭环成功率、滑移距离与脆性过载率**，并引入独立外扰/相变事件真值替代原同义反复的 `||d²w||` 标签。
>   2. **Step 2 — 跨模态缺省迁移（推理期去六维力）**：检验训练期使用 6D Wrench 作为物理锚定目标、推理期仅输入 GelSight / 应变时 `PAEC` 还能保留多少压缩增益。
>   3. **Step 3 — 接入 `NewGripper` 真机数据与闭环**：基于 `NewGripper`（500 Hz 6D F/T + 1–2 kHz 工具根部应变 + GelSight Mini + 6-DOF 本体）的无标签交互数据预训练通用 `PAEC` 接触编码器，冻结接入少样本真机策略（BC / Diffusion Policy / VLA）。
>
> 详见 **[V4 PAEC 算法蓝图](research/V4_PHYSICS_GROUNDED_CONTACT_COMPRESSION.md)** 与 **[v4_paec_contact_compression 实验索引](experiments/v4_paec_contact_compression/README.md)**。下文完整保留 Stage 1–4（`MCF / BPI-MCF / CMQ-BPI-MCF V2.2`）历史路线记录，不改写。

---

> **2026-11-03 历史研究范围升级与路线（Stage 3S–V2.1）**：
> 基于 [Stage 3S–V2.1 实验归档](experiments/v2_1_bpi_mcf_3dof_reflex/README.md) 与 [2026-11-03 进展记录](research/STAGE3S_V2_1_BPI_MCF_3DOF_REFLEX_20261103.md)，底层三项核心机制（**32D 参考帧锁定 `BPI-MCF` 算子 + 跨模态力螺旋防坍缩自监督 + 因果偶极新息门控 3-DoF 空间反射控制**）已在离线 8 阶段消融与 Isaac Lab 无混杂闭环/3 阶段长时序基准中完成验证。
>
> **当前最新三步推进顺序**：
> 1. **Step 1（Isaac Lab 长时序残差强化学习 Residual PPO/SAC）**：冻结 32D `BPI-MCF` 与预训练 `B5` 编码器作为 Warm-Start，训练受限残差策略并挂载在线物理自监督损失，进一步压低 `+Fx` 阶段 3 最终插孔对齐误差（`< 2.0 mm`）。
> 2. **Step 2（2×A100 慢频 VLM 微调 + DGX Spark 128GB 快慢双频推理部署）**：在 2×A100 上微调 `Qwen2.5-VL-7B-Instruct` 生成阶段切换与顺应模式原语（2–5 Hz 慢环），与 100–200 Hz 的 18K 参数 V2.1 3-DoF 触觉反射网络（快环）协同。
> 3. **Step 3（真机 Sim-to-Real 轻量标定与受控闭环验证）**：利用 `BPI-MCF` 的初始接触参考帧锁定（$k_{\text{base}}=5$）与单指预紧归一化（$f_{i,0}$），在真实双指触觉机械臂上完成空间极性对齐、预紧窗口设定与闭环抗扰/插孔对齐测试。
>
> 详见更新后的 **[研究方向修订与冻结稿](research/ICRA2028_MINIMAL_PAPER_FREEZE.md)**。下文保留 2026-10-10 及更早期的阶段规划作为历史记录。

> **2026-10-10 历史范围冻结记录（已于 2026-11-03 升级）**：早期以 [最小论文方向冻结稿](research/ICRA2028_MINIMAL_PAPER_FREEZE.md) 为准则，优先 **E1 仿真物理可观测性检查 / 正负切向与偏心加载 → FlexiTac 实测标定 → E2 Scalar/MCF/Dense 与历史消融 → E3 仿真 pilot 和 Stage 3A 真机抗倾覆闭环**。下文 2026-10-07 的“直接进入 Stage 3A”作为先前规划保留。


## 当前最高优先级：Stage 3A 真机闭环

离线 Stage 2P 已完成当前 GRU / Mamba 对照。下一步不并行联调 VLA，而是进入受控真机闭环：

**Stage 3A-0 bring-up → Stage 3A-1 受控扰动采集 → Stage 3A-2 matched Scalar / MCF residual controller → Stage 3A-3 冻结闭环评测。**

优先任务为 **Anti-Tilt / eccentric disturbance**，其次为 **Anti-Slip / pull disturbance**。主 baseline 为 Fixed、Scalar Reflex、MCF Reflex。第一轮 residual 只控制夹爪，不同时控制末端 XYZ / 姿态。

现有 Stage 2P 模型用于时序诊断，不能直接当作已训练好的控制器。只有 Stage 3A 给出可信闭环增量后，才进入 Stage 3B，将同一 tactile reflex 接到 scripted / BC / Diffusion / VLA nominal policy 上做 plug-and-play 展示。

详见 [research/STAGE3A_REAL_ROBOT.md](research/STAGE3A_REAL_ROBOT.md)。


更新：2026-10-07。

**已完成的 Stage 2P 记录：Mamba 离线验证。** 在当前 Now-casting 管线上进行两层 Mamba 与 GRU 的 T=50/100 对照，保留 Scalar / MCF、FP32和种子协议，验证侧向力矩RMSE与batch1延迟。见 [Mamba实验](research/STAGE2P_MAMBA.md)。

**历史推进记录：Stage 1 Conditional Pass → 封存 → Stage 2P Now-casting。** 首轮未来预测实验保留；当前按用户纠偏，采用同参数量两层 GRU，以 T=1/20/50 触觉历史重构当前 ATI 剪切力和侧向力矩，比较 Scalar / MCF，并比较各自长历史与 T=1。详见 [当前重构实验](research/STAGE2P_NOWCAST.md)。不回调 Stage 1。下文保留原始阶段安排，以上最新决策优先。

## 已完成 / 归档阶段

长期关注触觉表示、时序信息与接触丰富操作。此前阶段顺序为：

**Static 11D 收尾 → Minimalist Contact Flow → Contact-active validation → Stage 2P temporal probe**

这些阶段保留作为当前 Stage 3A 的离线依据与历史记录；不再作为当前执行入口。

## Stage 1：Minimalist Contact Flow

候选表示：

```text
s_t = [f_N, CoP_x, CoP_y, A]
delta_s_t = s_t - s_(t-1)
MCF_t = [s_t, delta_s_t]
```

第一轮直接复用已有 LeFlexiTac / TaF 数据。

不再把平滑全轨迹 Action MSE 作为主要验证目标，而是筛选 Contact-active Subsets，研究：

- 任务 A（TaF）：预测外力矩变化量 `Δτ_ext`（Torque Delta）；
- 任务 B（LeFlexiTac）：接触状态跳变检测，检验 MCF 对突然切换（微滑移或失稳前兆）的识别能力。

切片过滤剔除完全空载段和静态死锁且无外力扰动的平淡段；保留 `abs(Δf_N) > ε_f`、`norm(ΔCoP, 2) > ε_p`、`abs(ΔA) > ε_A` 中任一成立的片段。保留事件优先，避免误删脱开瞬间或仅面积变化的片段。

任务 A 预期判定：Scalar 丢失空间偏心信息，预测力矩变化误差必然显著偏高；MCF 依靠 CoP 及其差分，以 8 维捕捉力矩偏置，误差显著低于 Scalar，且逼近 Dense Tokens。保留该假设，实验验证后再修正。

核心比较：

**Scalar vs Minimalist Contact Flow vs Dense Raw tactile representation**

Static CoP、Physics11D 等可以作为额外消融。

当前希望通过实验回答：

> CoP + 差分是否显著优于纯 Scalar，并且能否与 Dense Raw Tokens 表现相当？

若实验支持，则进一步讨论“高阶空间矩是否冗余、核心是否主要来自 CoP 及其动态演化”。若不支持，则根据结果重新分析，不预先规定结论。

详细方案见 [research/MINIMALIST_CONTACT_FLOW.md](research/MINIMALIST_CONTACT_FLOW.md)。

## 后续 Stage 2

Stage 1 得到结果后，再决定如何研究 temporal memory。

可能包括把经过验证的 compact tactile representation 接入 Mamba / streaming memory，测试更长历史信息是否有额外作用。

具体架构、实验任务和 claim 等待 Stage 1 结果后确定，不在当前阶段锁死。

## Stage 3

若前面的 representation 与 temporal hypothesis 得到支持，再进入真实闭环 manipulation / recovery 实验。

具体任务同样根据前序实验结果选择。

## 时间目标

仍以 2027 年形成完整研究成果并准备 ICRA 2028 投稿为长期项目目标。内部可继续以 2027-07-31 作为完整初稿的项目管理参考日期；正式投稿要求以未来官方 CFP 为准。

## 原则

路线图描述的是当前计划，不代表结果。每一阶段完成后根据真实实验结果更新下一阶段，而不是提前把整篇论文的结论锁死。
