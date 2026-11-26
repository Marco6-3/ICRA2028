# ICRA2028 · 物理锚定接触状态压缩（V4 PAEC）

> **仓库主线**：本仓库聚焦 **PAEC（Physics-Anchored Action-Effect Contact Compression）** 研究主线——以机器人物理力学增量（6D Wrench / 夹爪力 / 触觉响应）与零动作恒等分解预测器（$\widehat{\Delta \mathbf{y}}_{t,\Delta} = b_\phi(z_t) + g_\phi(z_t, a)$，$g(\cdot, \mathbf{0}) \equiv \mathbf{0}$）将因果接触历史压缩为紧凑低维状态 $z_t$，并评估其在未见任务、未见本体、跨实验室数据集、工业漂移与 MuJoCo 闭环控制中的零梯度冻结迁移能力。
>
> **历史主线归档说明**：早期的 Stage 1–4（`MCF / BPI-MCF / CMQ-BPI-MCF / real_robot_reflex`）与 V3（音频 `CAEC-JEPA`）已从 `main` 分支清理，完整历史快照保存在 Git 标签 **`archive/pre-v4-mainlines`** 中（`git checkout archive/pre-v4-mainlines` 可随时查阅）。

---

## 核心进展摘要（严格 FP32 + Episode 划分 + SHA-256 权重冻结审计）

1. **V4.7 工业三任务视觉-触觉 ACT 缺陷检测基准（2026-11-26，1,620 局闭环）**
   - 配置：`15 demos × 3 seeds × 12 eval eps × 3 regimes`（`Clean` / `+0.5σ` 温漂 / `+1.5σ` 极端零漂），共享冻结 ImageNet `ResNet18 layer3` + `4×4` SpatialSoftmax 与 `O(1)` 递推触觉编码（递推等价最大误差 `1.67e-6 < 1e-5`），`strict_fp32=True`、`strictly_frozen_pass=True`。
   - **Task 1（`battery_scan` 新能源电芯扫描，限力 `≤ 11.5 N`）**：`ACT_RGB_PAEC` 在 Clean 与 `+0.5σ` 下均取得 **`100.00 ± 0.00%`** 成功率与 **`0.0%`** 过载违例（`+1.5σ` 为 `66.67 ± 0.00%` / `0.0%` 违例）；在线缺陷检出 `PAEC_Rel_Surprise`（`max - median`）在 Clean / `+0.5σ` / `+1.5σ` 全工况保持 **`1.0000` AUROC**。
   - **Task 2（`auto_connector` 汽车接插件插接，限力 `≤ 15.0 N`）与 Task 3（`semi_grasp` 半导体载具抓放，限力 `≤ 14.0 N`）诚实边界**：
     - Task 2 中纯视觉 `ACT_RGB_Proprio` 过载违例高达 `58.33%`（峰值 `20.49 N`），`ACT_RGB_RawFT` Clean `58.33%` 但 `+0.5σ` 跌至 `0.00%`，`ACT_RGB_PAEC` Clean `50.00%`（`2.78%` 违例）、`+0.5σ` `11.11%`（`0.0%` 违例）、`+1.5σ` `0.00%`；
     - Task 3 中 `Classical_Baseline` `0.00%` 且 `100.00%` 压碎违例（`15.85 N`），`ACT_RGB_PAEC` 为全场最高 `13.89%`（Clean 与 `+0.5σ` `0.00%` 违例），反映 `15-shot` 纯端到端 ACT 在毫米级硬约束对齐与多阶段离散切换上的边界。
   - **单步端到端延迟（`batch = 1`）**：GPU（`RTX A6000`）`P50 = 5.266 ms / P95 = 5.685 ms`（`175.9 Hz`）；纯 CPU `P50 = 7.550 ms / P95 = 9.669 ms`（`103.4 Hz`）。
   - 详情：[V4_7_ACT_DEFECT_INSPECTION_REPORT.md](experiments/v4_paec_contact_compression/outputs/v4_7_act_defect_suite/V4_7_ACT_DEFECT_INSPECTION_REPORT.md) | [fig7_v4_7_act_defect_suite.png](experiments/v4_paec_contact_compression/outputs/v4_7_act_defect_suite/fig7_v4_7_act_defect_suite.png)

2. **V4.6 工业科学强化：同规模 `B4_NoBG` 消融、经典导纳控制与零抖动虚拟刚度探针（2026-11-26）**
   - **同参数消融（`B4_NoBG_Small_d32` `900,694` vs `PAEC_Small_d32` `902,038`）**：`PAEC_Small_d32` 在跨库 `RH20T → RDP` `5/10/100%` Clean MSE 达 `0.23083 / 0.20375 / 0.16327`（超越 RDP 域内自训 `0.20453`），优于去掉 $b_\phi+g_\phi$ 分解的 `B4_NoBG_Small_d32`（`0.23133 / 0.20571 / 0.16474`）；同时诚实记录在同源 `RH20T cfg_0004` $R^2$（`0.6065` vs `0.5918`）与顺应擦拭闭环成功率上 `B4_NoBG_Small_d32` 同样可达到 `100.0%`。
   - **经典导纳基线对比（`3 seeds × 24 eps = 72` 局）**：`Admittance_FixedFeed` / `Admittance_TunedShear` 在 Clean 均 `100.0%`（RMSE `1.19 / 1.13 N`），但在 `Static +1.5σ Drift` 跌至 `55.6% / 31.9%`；开机清零 `Admittance_InitTare` Clean `81.9%`、`Step +1.5σ Drift (t≥35)` `73.6 ± 5.2%`；冻结 `PAEC_Small_d32` 在全部工况与预算（`5/10/20 demos`）保持 **`100.0%`** 与 `0.0%` 违规。
   - **零抖动虚拟方向刚度探针**：直接验证在未知软硬区刚度辨识上的定量能力。
   - 详情：[V4_6_INDUSTRIAL_FORTIFICATION_REPORT.md](experiments/v4_paec_contact_compression/outputs/v4_6_industrial_fortification/V4_6_INDUSTRIAL_FORTIFICATION_REPORT.md) | [fig6_v4_6_industrial_fortification.png](experiments/v4_paec_contact_compression/outputs/v4_6_industrial_fortification/fig6_v4_6_industrial_fortification.png)

3. **V4.5 MuJoCo 4-DoF 顺应擦拭与阻滞恢复闭环（2026-11-25）**
   - 将 `RH20T`（`6,289` eps）预训练权重 **100% 零梯度冻结（`strictly_frozen_pass = true`）** 迁移至 MuJoCo 闭环仿真：`PAEC_Small_d32` 与 `PAEC_Base_d32` 在 `5/10/20 demos × {Clean, +1.5σ Drift}` 全预算全工况下均保持 **`100.00% ± 0.00%`（各 432/432 局）** 与 `0.00%` 违规（同规模 `B1b_RawWrench6D_E2E` 在 20-demo Clean 因卡滞坍塌至 `0.00%`）。
   - 详情：[V4_5_MUJOCO_CLOSEDLOOP_REPORT.md](experiments/v4_paec_contact_compression/outputs/v4_5_mujoco_closedloop/V4_5_MUJOCO_CLOSEDLOOP_REPORT.md)

4. **V4.4 `A6000` 万条 `RH20T` 规模律与跨库迁移（2026-11-25）**
   - 全量 7 本体 `10,041` 轨迹 / `9,839,771` 帧 / `140` 任务（保留 `10,018` 轨迹，中位接触活跃率 `0.8431`），`323 / 323` 下游评测 `strictly_frozen_pass = True`。
   - `PAEC_Small_d64` 未见任务 / 未见本体 3 步物理力学 $R^2$ 达 **`0.5375 / 0.6078`**；3 种子统计下 `PAEC_Small_d32` 最稳健（$R^2 = \mathbf{0.5359 \pm 0.0016} / \mathbf{0.5951 \pm 0.0030}$，跨库 `RH20T → RDP` `5-shot 0.23075 ± 0.00019` / `10-shot 0.20350 ± 0.00018` 击败域内自训）。
   - 详情：[V4_4_A6000_SCALE_REPORT.md](experiments/v4_paec_contact_compression/outputs/v4_4_a6000_scale/V4_4_A6000_SCALE_REPORT.md)

5. **V4.0–V4.3 `RDP` 域内复现、双向跨任务迁移与 `Colab L4` Scaling Law**
   - 官方 RSS-2025 `Reactive Diffusion Policy`（`peel_v3`、`wipe_v1.5`、`lift_v2`）复现与双向 `wipe ↔ peel` 零样本冻结迁移，详见 [v4_paec_contact_compression README](experiments/v4_paec_contact_compression/README.md)。

---

## 仓库结构与导航

| 目录 / 文档 | 说明 |
| --- | --- |
| [experiments/v4_paec_contact_compression/](experiments/v4_paec_contact_compression/README.md) | `PAEC` 核心实现、MuJoCo 仿真、V4.0–V4.7 实验脚本、图表与审计报告 |
| [research/V4_PHYSICS_GROUNDED_CONTACT_COMPRESSION.md](research/V4_PHYSICS_GROUNDED_CONTACT_COMPRESSION.md) | `PAEC` 算法原理与设计蓝图 |
| [research/V4_SCALING_AND_PROTOCOL_V4_1.md](research/V4_SCALING_AND_PROTOCOL_V4_1.md) | `V4.1–V4.4` 物理零点、审计协议与 `RH20T / RDP` 缩放规范 |
| [research/ARXIV_2026_Q2_ROADMAP_AND_HARDWARE_BOM.md](research/ARXIV_2026_Q2_ROADMAP_AND_HARDWARE_BOM.md) | 2026 Q2 文献调研、工业界痛点综述与真机硬件 BOM |
| [papers/README.md](papers/README.md) | 参考文献整理（`TacForcing`、`TacMamba`、`T-REX` 等） |
| [ROADMAP.md](ROADMAP.md) | V4 阶段演进与后续待选路径 |
| [AGENTS.md](AGENTS.md) | 本仓库科研与工程协作规范 |
