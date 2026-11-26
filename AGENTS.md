# ICRA2028 仓库协作规范（V4 PAEC 主线）

> **当前主线（2026-11-26）**：仓库已清理早期的 Stage 1–4（`MCF / BPI-MCF / CMQ-BPI-MCF / real_robot_reflex`）与 V3（音频 `CAEC-JEPA`）分支，当前唯一活跃研究主线为 **`V4 PAEC`（物理锚定动作效应接触压缩）**。完整历史快照保存在 Git 标签 `archive/pre-v4-mainlines`。

## 活跃目录与文档边界

- **实验代码与报告**：[experiments/v4_paec_contact_compression/](experiments/v4_paec_contact_compression/README.md)
- **算法与协议文档**：
  - [research/V4_PHYSICS_GROUNDED_CONTACT_COMPRESSION.md](research/V4_PHYSICS_GROUNDED_CONTACT_COMPRESSION.md)
  - [research/V4_SCALING_AND_PROTOCOL_V4_1.md](research/V4_SCALING_AND_PROTOCOL_V4_1.md)
  - [research/ARXIV_2026_Q2_ROADMAP_AND_HARDWARE_BOM.md](research/ARXIV_2026_Q2_ROADMAP_AND_HARDWARE_BOM.md)
- **参考文献**：[papers/README.md](papers/README.md)
- **局部草稿目录保护**：本地若存在未跟踪的 `paper/` 目录（如论文草稿），**禁止删改或提交至远端**。

## 科学诚信与实验审计约束

- **严格区分事实与计划**：区分已运行验证的实测数值、当前待证假设与未来规划；未运行的实验禁止写成结果。
- **保留所有负结果与能力边界**：如实保留 `V4.7` 中 Task 2（`auto_connector` 在 `+0.5σ / +1.5σ` 下降至 `11.11% / 0.00%`）、Task 3（`semi_grasp` `13.89%`）、`V4.7` Task 1 `+1.5σ` 回落（`66.67%`），以及 `V4.6` 中 `B4_NoBG_Small_d32` / `MAE_Tiny` 在部分同源指标上的接近或反超，不为了美化叙事删除或弱化对照。
- **严格计算与划分规范**：
  - 所有预训练与闭环评测保持严格 `FP32`（`allow_tf32=False`）；
  - 严禁帧级随机切分造成的轨迹泄漏，保持 episode / task / embodiment 级别严格划分与 Train-only 归一化；
  - 下游冻结迁移必须通过 `SHA-256` 权重冻结审计（`strictly_frozen_pass = True`）。
- **仓库卫生**：不向 Git 提交 `.pt` / `.npz` 等大型权重或缓存二进制产物（保持 `.gitignore` 规则），不随意扩展仓库为无关通用基础设施。
- **优先级**：用户最新明确指令始终优先于文档既有规划。
