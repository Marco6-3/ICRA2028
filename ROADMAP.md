# 研究路线（V4 PAEC 主线）

> **历史归档说明**：Stage 1–4（`MCF / BPI-MCF / CMQ-BPI-MCF`）与 V3 音频 `CAEC-JEPA` 相关代码与文档已存档于 Git 标签 `archive/pre-v4-mainlines`，当前分支聚焦 **V4 PAEC** 单一主线。

## 已完成阶段（V4.0 – V4.7）

1. **V4.0 – V4.1（`RDP` 物理基础与双向跨任务迁移）**：在 RSS-2025 `Reactive Diffusion Policy`（`peel_v3`、`wipe_v1.5`、`lift_v2`）上验证 `PAEC` 的低秩物理压缩与双向 `wipe ↔ peel` 零梯度冻结迁移能力，并冻结物理零点与多基线公平评测协议（见 [V4_SCALING_AND_PROTOCOL_V4_1.md](research/V4_SCALING_AND_PROTOCOL_V4_1.md)）。
2. **V4.2 – V4.4（7 本体 `RH20T` 规模律与 `RH20T → RDP` 跨库迁移）**：流式抽取并物理审计最高 `10,041` 轨迹 / `9,839,771` 帧 / `140` 任务（`V4.4 A6000`），建立数据规模与瓶颈容量 Scaling Law，并在 `5/10-shot` 跨库 `RH20T → RDP` 迁移上超越域内自训（见 [V4_4_A6000_SCALE_REPORT.md](experiments/v4_paec_contact_compression/outputs/v4_4_a6000_scale/V4_4_A6000_SCALE_REPORT.md)）。
3. **V4.5 – V4.6（MuJoCo 闭环零漂鲁棒性与经典导纳/消融对照）**：验证冻结 `PAEC_Small_d32` 在 Clean、Static `+1.5σ`、Step `+1.5σ` 三工况下全预算 `100.0%` 闭环成功与 `0.0%` 过载违例，补齐同参数 `B4_NoBG_Small_d32` 消融、三组非稻草人经典导纳基线对比与零抖动虚拟刚度探针（见 [V4_6_INDUSTRIAL_FORTIFICATION_REPORT.md](experiments/v4_paec_contact_compression/outputs/v4_6_industrial_fortification/V4_6_INDUSTRIAL_FORTIFICATION_REPORT.md)）。
4. **V4.7（工业三任务视觉-触觉 ACT 缺陷检测基准）**：在 `battery_scan`、`auto_connector`、`semi_grasp` 三个工业场景上完成 `1,620` 局闭环实测与 `batch=1` 延迟评测（GPU `P95=5.685 ms` / CPU `P95=9.669 ms`），并诚实标定出端到端小样本 ACT 在无倒角硬插接与多阶段离散抓放上的边界（见 [V4_7_ACT_DEFECT_INSPECTION_REPORT.md](experiments/v4_paec_contact_compression/outputs/v4_7_act_defect_suite/V4_7_ACT_DEFECT_INSPECTION_REPORT.md)）。

## 后续待决策/待执行方向（未运行，不提前写成结果）

1. **论文叙事收束与实验取舍**：结合 V4.4–V4.7 已验证证据（跨本体预训练 + 抗静态/突跳零漂 + 虚拟刚度/惊奇度在线异常检测 + 视觉-触觉 ACT 边界）确定最终投稿叙事，决定是否补充 Task 2/3 的低成本环境与策略修复消融（V4.7b）。
2. **第 3 阶段 ROS 2 仿真控制验证（硬件到位前）**：在远程 `a6000` Linux/WSL2 环境中基于 `ros-controls/mujoco_ros2_control` 构建混合分层控制闭环（高层阶段状态机 + `PAEC` 柔顺/防滑原语 + 力安全看门狗），验证 ROS 2 实时消息链路与控制安全性。
3. **真机集成与闭环（硬件交付后）**：待机械臂、夹爪与六维力/触觉传感到位后，按照 [ARXIV_2026_Q2_ROADMAP_AND_HARDWARE_BOM.md](research/ARXIV_2026_Q2_ROADMAP_AND_HARDWARE_BOM.md) 完成标定与物理缺陷检测闭环验证。
