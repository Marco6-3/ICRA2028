# V4（路线 1）实验：`PAEC` 物理锚定动作效应接触状态压缩

> **方向**：基于 Round-1 `ManiWAV/flip` 证伪后的结构性转型（见 [V4_PHYSICS_GROUNDED_CONTACT_COMPRESSION.md](../../research/V4_PHYSICS_GROUNDED_CONTACT_COMPRESSION.md)）。
> - **跳出抽象隐变量 $z$ 三难困境**：不再做抽象潜空间 EMA + VICReg 噪声膨胀，而是以机器人自动记录的**多步真实物理力学增量**（6D Wrench + 夹爪力 + GelSight 视触觉形变）为免人工标注自监督目标，将因果接触历史压缩为紧凑低维状态 $z_t \in \mathbb{R}^{16}$。
> - **与 `NewGripper` 硬件直接对齐**：主传感聚焦于**高信噪比力觉 / 应变 / 视触觉 + 6-DOF 本体**（不再押注冗余且易受传递函数漂移影响的 20 kHz 接触麦克风）。

## 文件说明
- [fetch_rdp_full_nonimg.py](fetch_rdp_full_nonimg.py)：通过标准 HTTP Range 仅流式抽取官方 RSS-2025 `Reactive Diffusion Policy` 完整版 `.zarr.zip`（`peel_v3`、`wipe_v1.5`、`lift_v2`）中的非图像物理流（6D Wrench + 夹爪力/开度 + GelSight Mini marker 位移与嵌入 + 6-DOF TCP 位姿/速度），将 45 GB 压缩至 ~60 MB 全采率物理数据。
- [inspect_rdp_zarr.py](inspect_rdp_zarr.py)：检查数据完整性并构建严格 episode 级 70/15/15 划分与 Train-only 归一化缓存。
- [models_paec.py](models_paec.py)：实现 `PAEC`（紧凑 $d_z=16$ + 零动作恒等分解预测器 $b_\phi + g_\phi$ + 零抖动虚拟方向刚度探针 + 物理力残差惊奇度）及全部同容量对照（`B0_ProprioOnly`、`B1_RawFT_End2End`、`B2_MAE`、`B3_LatentJEPA_VICReg`、`B4_NoBG`、`B5_SingleFrame`、`B6_ShortWin`）。
- [run_v4_paec_benchmark.py](run_v4_paec_benchmark.py)：端到端单元审计、预训练、力学前向预测精度、Codex 匹配力螺旋/相反历史压测、速度控制物理惊奇度突变检测，以及多预算 × 3 seeds 下游 Action-Chunk BC（含 Clean / Contact / 零点漂移 OOD / 刚度增益 OOD），支持 `--save-ckpt` 落盘预训练权重。
- [run_v4_cross_task_transfer.py](run_v4_cross_task_transfer.py)：严格跨任务零样本冻结迁移评测（SHA-256 权重冻结审计 + `src_norm` 纯零样本物理尺度与 `tgt_norm` 目标训练集标准化双协议）。

## 已完成严格审计报告（Tailscale `nvidia-desktop` Orin CUDA, 严格 FP32）
1. **域内任务 1 — `peel_v3`（60 eps, 25,710 帧）**：[RDP_PEEL_V4_RESULTS.md](outputs/rdp_peel_v4/RDP_PEEL_V4_RESULTS.md) | [v4_paec_results.json](outputs/rdp_peel_v4/v4_paec_results.json)
2. **域内任务 2 — `wipe_v1.5`（80 eps, 37,267 帧）**：[RDP_WIPE_V4_RESULTS.md](outputs/rdp_wipe_v4/RDP_WIPE_V4_RESULTS.md) | [v4_paec_results.json](outputs/rdp_wipe_v4/v4_paec_results.json)
3. **双向零样本冻结跨任务迁移 — `wipe_v1.5 ↔ peel_v3`**：[RDP_CROSS_TASK_V4_RESULTS.md](outputs/RDP_CROSS_TASK_V4_RESULTS.md) | [wipe→peel JSON](outputs/rdp_xfer_wipe2peel_v4/v4_cross_task_results.json) | [peel→wipe JSON](outputs/rdp_xfer_peel2wipe_v4/v4_cross_task_results.json)
