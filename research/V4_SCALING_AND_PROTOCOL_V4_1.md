# V4.1–V4.5 研究协议：PAEC 物理零点冻结、跨数据集 Canonical Schema 与 Scaling Law 验证计划

> **状态（2026-11-24，规划与 V4.1 代码已就绪，尚未在外部大规模数据上运行）**：  
> - **历史保留**：2026-11-20/21 的 V4.0 RDP `peel_v3`（60 eps）、`wipe_v1.5`（80 eps）及 `wipe ↔ peel` 跨任务迁移归档完整保留，不覆写。  
> - **本轮目标**：先通过 **V4.1** 修复已定位的物理零点与基线对齐隐患、冻结评测协议；再通过 **V4.2–V4.5** 系统检验 PAEC 在 `RH20T_lowdim`、`PRISM`、`RDP`（含 `lift_v2`）上的数据规模曲线、跨本体/跨任务泛化能力，以及模型容量 vs 瓶颈维度的解耦规律；并与 `NewGripper` / 真机闭环双线并行。

---

## 1. 为什么要做 Scaling Experiment（三大科学判别）

在 RDP 140 条轨迹（测试集仅 21 条轨迹）上，PAEC 已相对 `Raw / MAE / LatentJEPA` 和纯本体 `B0` 取得稳定排名领先，但 Clean MSE 绝对降幅约 0.8%–1.6%。单靠小规模单本体数据无法区分 PAEC 优势来自以下哪一种机制：

1. **小样本低维正则假说**：仅仅因为 16D 瓶颈压缩过滤了 30D GelSight 噪声；
2. **单域物理目标假说**：物理增量预测目标优于重构/抽象 JEPA，但仅限于同本体同任务；
3. **可扩展通用接触表征假说**：随着异构接触数据规模扩大（$500 \to 2\text{k} \to 10\text{k} \to 50\text{k}+$），免标注物理预测预训练能持续提升未见任务、未见机器人配置的少样本下游控制与跨数据集迁移能力。

**五个待实验回答的核心问题**（均作为待检验假设，结果由实测决定）：
1. 数据规模增大后，PAEC 相对同维度/同增强 `MAE` 和 `LatentJEPA_d16` 的优势是保持、拉大还是发生反转？
2. 当接触数据跨越数十种任务与多款机械臂时，$d_z=16$ 紧凑瓶颈是否仍够用，还是必须扩大至 $32/64$？
3. 大规模免标注 6D F/T 预训练后，在未见任务（如 `PRISM` / `RDP`）上仅用 5–10 条示教能否获得显著优于域内自训与纯本体的控制增益？
4. 跨机器人配置、跨夹爪、跨传感器时，冻结编码器是否仍保持正向迁移？
5. 大规模公开 6D F/T 预训练表征能否有效迁移到 `NewGripper` 真机少样本闭环操作？

---

## 2. V4.1 协议冻结：已修复的 4 处物理零点漏洞与公平基线补齐

在扩充外部数据前，V4.1 已在 [models_paec.py](../experiments/v4_paec_contact_compression/models_paec.py)、[run_v4_paec_benchmark.py](../experiments/v4_paec_contact_compression/run_v4_paec_benchmark.py) 与 [run_v4_cross_task_transfer.py](../experiments/v4_paec_contact_compression/run_v4_cross_task_transfer.py) 中完成以下修复并纳入自动单元审计：

### 2.1 物理零点对齐（Zero-Centered Action Scaling）
- **原 V4.0 缺陷**：对局部本体增量 `proprio_delta / action_next` 使用 `(a - act_mean) / act_std` 减均值标准化，导致归一化后的 `0` 向量对应“数据集平均运动”而非物理静止 $\Delta q = \mathbf{0}$，同时偏移了：
  1. `DecomposedPhysicsPredictor` 中 $g_\phi(z_t, \mathbf{0}) \equiv \mathbf{0}$ 的零动作基准；
  2. `compute_loss` 中 `is_hold = (||a_agg|| < hold_thresh)` 的静止锚定掩码；
  3. `compute_virtual_probe` 中沿 $\pm x, \pm y, \pm z$ 单轴微动时的其余轴零位；
  4. `eval_codex_wrench_matching` 中判断过去运动方向相反（$\cos < -0.5$）的速度零点。
- **V4.1 修复**：默认改用 `--act-norm-mode rms`（仅按训练集 RMS 缩放、**严禁减均值**）或 `--act-norm-mode si_fixed`（固定 SI 物理步长缩放），并通过单元审计强制检验 `normalize_action(0) == 0.0`、`aggregate_action(0, delta) == 0.0` 与 `g_phi(z, 0) == 0.0` 逐比特成立。

### 2.2 公平对照与消融补齐
| 新增/对齐项 | 检验目的 |
| :--- | :--- |
| `B1b_RawWrench6D_E2E` | 端到端 GRU 仅喂入 6D Wrench（去掉 30D GelSight），排除 `B1` 输给 PAEC 仅因 30D 视触觉噪声的替代解释 |
| `B2b_MAE_DriftAug` | 给 `MAE` 输入施加与 PAEC 完全相同的轨迹级零漂增强（`drift_aug_std=0.35`） |
| `B3_LatentJEPA_d16` & `B3b_LatentJEPA_d16_DriftAug` | 将 `LatentJEPA` 瓶颈维度严格对齐至 $d_z=16$，并补齐在线编码器零漂不变性增强（保留 `B3_d64` 作 V4.0 历史对照） |
| `--contact-subset wrench6d` | 在 RDP 同一数据划分上直接对比 **纯 6D F/T** vs **37D（6D F/T + 夹爪力 + 双 GelSight）** |
| `--con-norm-mode si_fixed` | 固定 SI 物理单位缩放（见第 3 节），避免每数据集独立 Z-score 破坏刚度单位（N/m） |

---

## 3. 跨数据集统一物理规范（Canonical 6D Contact Schema）

V4.0 跨任务实验已实证：用目标数据集各自做 Z-score（`tgt_norm`）会把不同任务的真实刚度比例强行拉成 `std=1`，导致零样本物理力预测 $R^2$ 立即变负；而保留物理量纲（`src_norm`）才能实现零样本反转段力学预测。因此，跨 `RH20T` / `PRISM` / `RDP` / `NewGripper` 的大规模预训练必须统一到以下物理契约：

| 字段 | 维度 | 物理单位与坐标系契约 | 默认 SI 固定缩放常数（`si_fixed`） |
| :--- | :---: | :--- | :--- |
| `wrench_tcp` | 6 | $[F_x, F_y, F_z, \tau_x, \tau_y, \tau_z]$，单位 $\text{N}, \text{N}\cdot\text{m}$，**必须转换至 TCP/工具局部坐标系** | $[10, 10, 10]\,\text{N},\; [0.5, 0.5, 0.5]\,\text{N}\cdot\text{m}$ |
| `delta_pose_tcp` | 6 | 局部相对平动 $\Delta p_{\text{tcp}}$ 与轴角旋转 $\Delta r_{\text{tcp}}$，单位 $\text{m}, \text{rad}$，零中心 | $[0.005]\,\text{m},\; [0.05]\,\text{rad}$ |
| `delta_grip` | 1 | 夹爪开度步长增量（可选/缺失时填 0 并带 mask），单位 $\text{m}$，零中心 | $0.005\,\text{m}$ |
| `dt_sec` | 1 | 物理步长时间间隔（秒） | 统一重采样至 **20 Hz**（$\Delta t = 50\,\text{ms}$）或显式输入预测器 |
| `meta` | — | `dataset_id`, `robot_config_id`, `task_id`, `episode_id`, `contact_active_frac`, `gravity_corr_flag` | 用于严格 **Held-Out Task / Held-Out Robot** 分组 |

### 外部数据入库前强制审计红线（由 `audit_rh20t_prism_lowdim.py` 执行）
1. **频率审计**：检查 F/T 流与 TCP 位姿流的真实中位 `dt` 与时间抖动；禁止把 10 Hz 和 100 Hz 数据直接按相同帧步长 `delta ∈ {1, 3, 6}` 混训。
2. **F/T 健康度与有效接触密度**：统计非零占比、NaN/Inf、饱和帧占比、接触活跃帧占比（如 $\|\mathbf{F}\| > 1.0\,\text{N}$ 或 $\|\Delta \mathbf{F}\| > 0.2\,\text{N}$）；对弱接触/纯自由空间 pick-and-place 轨迹实施分层过滤或采样加权，避免有效信号被稀释。
3. **工具重力随姿态耦合审计**：计算静态/低速段力矢量与末端姿态 $R_t$ 的相关性，标记未做工具重力补偿的配置（常值零漂增强 `DriftAug` 无法消除随旋转 $R_t$ 变化的伪力增量）。

---

## 4. V4.1–V4.5 分阶段执行矩阵与 Go/No-Go 门控

| 阶段 | 数据与规模 | 模型配置 | 评测切分与硬指标 | Go / No-Go 门槛 |
| :--- | :--- | :--- | :--- | :--- |
| **V4.1** | RDP `peel_v3` (60) + `wipe_v1.5` (80) + `lift_v2` | `PAEC-Tiny` (96 / 2L / d16) + 全部公平基线 | 域内 + 双向迁移；`37D` vs `6D-only`；`rms` vs `si_fixed`；多预训练种子 | 4 项单元审计 100% PASS；修复零点且对齐 `JEPA_d16` / `MAE_DriftAug` / `RawWrench6D` 后，明确 PAEC 净优势来源 |
| **V4.2** | `RH20T_lowdim` (500–1,000 eps) + `PRISM` 小样本审计 | 固定 `PAEC-Tiny` (纯 6D F/T) | **留出未见任务 + 留出未见机器人配置**；前向力学 $R^2$ + 下游少样本 BC + 零样本测 RDP | 外部数据通过物理审计；在未见任务/未见配置上 `PAEC` 优于 `MAE_DriftAug` 与 `JEPA_d16_DriftAug` 且无负迁移，才允许扩至 10k |
| **V4.3** | `RH20T_lowdim` 分级子集：$500 \to 2\text{k} \to 10\text{k}$ eps（按接触活跃度分层） | 先固定 `PAEC-Tiny` 控制变量 | 完整绘制 `PAEC vs MAE vs JEPA_d16 vs NoBG` Scaling 曲线（Held-Out Task / Embodiment + Zero-Shot RDP/PRISM） | 确认随数据规模扩大呈现单调或稳定改善斜率 |
| **V4.4** | 50,000+ eps（RH20T 全量接触活跃子集 + PRISM 训练集） | 控制变量扩容量与瓶颈：<br>• `Tiny` (96, 2L, 16)<br>• `Small` (256, 2L, 16)<br>• `Base` (512, 4L, 16)<br>• 瓶颈消融：`Small` × $d_z \in \{8, 16, 32, 64\}$ | 检验“计算容量随数据增长、物理接触状态仍保持低维”的核心科学假设 | 如实记录最优 $d_z$（若 32D/64D 显著优于 16D 则接受实验事实，不预设结论） |
| **V4.5** | 跨数据集零样本 & 5–10 shot 适配：`PRISM` 未见工业任务 + `RDP` 全任务 | 冻结 V4.4 预训练主干 | 对比 `纯 6D F/T` vs `6D F/T + 触觉`（在同子集配对评估） | 验证大规模预训练在 5–10 条新任务示教下优于域内从头预训练 |
| **并行闭环线** | 轻量接触仿真（独立外扰/断裂真值）+ `NewGripper` 真机少样本闭环 | 冻结主干 + 下游策略 / 高频安全护栏 | **闭环任务成功率**、物理接触力安全代价、**独立事件真值惊奇度 AUROC**（替换原 `||d²w||` 同义反复标签） | 不等 50k 全部完成即可并行启动；最终论文主结论必须包含真实闭环证据 |

---

## 5. V4.1 立即执行命令（`nvidia-desktop` 或本地 CUDA）

```bash
# 1. RDP peel_v3 V4.1 全量（37D，物理零点 RMS 缩放 + 完整公平基线）
python experiments/v4_paec_contact_compression/run_v4_paec_benchmark.py \
  --cache <peel_cache.npz> \
  --out-dir experiments/v4_paec_contact_compression/outputs/rdp_peel_v4_1_37d \
  --act-norm-mode rms --con-norm-mode zscore --contact-subset all --save-ckpt

# 2. RDP peel_v3 V4.1 纯 6D Wrench + SI 固定缩放（为 RH20T/PRISM 跨数据集预训练建锚）
python experiments/v4_paec_contact_compression/run_v4_paec_benchmark.py \
  --cache <peel_cache.npz> \
  --out-dir experiments/v4_paec_contact_compression/outputs/rdp_peel_v4_1_w6_si \
  --act-norm-mode si_fixed --con-norm-mode si_fixed --contact-subset wrench6d --save-ckpt

# 3. 远程审计 RH20T_lowdim 与 PRISM 文件目录结构（零大文件下载）
python experiments/v4_paec_contact_compression/audit_rh20t_prism_lowdim.py --mode remote-list
```
