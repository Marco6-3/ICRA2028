# Stage 4 (`V2.2`)：蠕变补偿多极正交双指流形接触场（`CMQ-BPI-MCF` 48D）、本体-触觉伴随拷贝解耦双头架构与防坍缩在线残差强化学习归档

日期：2026-11-03。本目录归档 **Stage 4 (`V2.2`)** 的完整离线 10 阶段算子消融、36 工况 `pilot_tangent_02` 7 类 OOD 拆解、本体-触觉双头架构与在线残差强化学习防坍缩验证、以及 Isaac Lab 6 控制器 6-DoF 物理闭环抗扰动与 3 阶段长时序插孔操作基准测试。仓库既有的 Stage 1、Stage 2P、Stage 3S 与 Stage 3S–V2.1 原始数据、代码与哈希全部原样保留。

根据本轮实验突破，项目研究范围在 `V2.1`（`BPI-MCF` 32D + 跨模态辅助监督 + 3-DoF 单头启发式模仿）基础上正式推进至 **Stage 4 (`V2.2`)** 三层闭环架构：
1. **底层物理算子创新（`CMQ-BPI-MCF` 48D）**：针对 [MCF_OBSERVABILITY_LIMITS_DISCUSSION.md](../../research/MCF_OBSERVABILITY_LIMITS_DISCUSSION.md) 揭示的弹性体粘弹性法向蠕变（Creep）、跨增益/跨预紧尺度混叠（Scale Confounding）与桌面支撑反演（Support Inversion）三大可观测性极限，构建 **法向蠕变解耦 + 偶极单位方向场（$\hat{\mathbf{d}}_{\text{anti}}, \hat{\mathbf{d}}_{\text{sym}}$）+ 二阶空间四极矩（Spatial Quadrupole $\mathbf{Q}$）+ 双时间尺度新息包络（Fast-Slow Envelope）** 的 48D 物理结构化算子。
2. **中层策略网络创新（`ProprioTactileDualHeadPolicyV22`）**：引入 **机械臂腕部动作伴随拷贝（Efference-Copy Proprioceptive-Tactile Decoupling）**，将主动顺应运动引起的接触形变与外源未知切向扰动正交分离；同时将单一动作输出头解耦为 **高频顺应卸载头（$\mathbf{u}_{\text{damp}}$）+ 慢速姿态回正头（$\mathbf{u}_{\text{recenter}}$）**。
3. **顶层强化学习创新（`Online Warm-Start Residual RL + Anti-Collapse Self-Supervision`）**：突破 `V2.1` 离线启发式教师（$u^*$）的模仿上限，以 Warm-Start BC 为基策略，学习受限残差策略 $\mathbf{u} = \text{clip}(\mathbf{u}_{\text{BC}} + \alpha \Delta \mathbf{u}_{\pi_\theta}, -1, 1)$，并证明 **若无辅助力螺旋与方向自监督正则，在线强化学习梯度会在 40 轮内发生表征坍缩（Action MSE 恶化至 `0.285`，Isaac Lab 静默抖动暴增至 `2.25 mm`）**；加入防坍缩辅助自监督后，策略在保持零静默抖动的同时全面超越离线教师。

详见 **[Stage 4 (V2.2) 进展报告](../../research/STAGE4_V2_2_CMQ_PROPRIO_RESIDUAL_RL_20261103.md)**、**[可观测性极限与物理破局讨论](../../research/MCF_OBSERVABILITY_LIMITS_DISCUSSION.md)** 与更新后的 **[研究范围修订与冻结稿](../../research/ICRA2028_MINIMAL_PAPER_FREEZE.md)**。

---

## 一、核心已验证成果总览

1. **创新点 1（`CMQ-BPI-MCF` 48D 蠕变补偿多极接触算子，10 阶段逐项消融 + 36 工况 OOD 破局）**：
   - **独立盲测集 `Fresh12` 跃升 `+18.46%`**：在完全零样本的 12 组独立物理盲测集 `Fresh12` 上，方向辨识准确率从 `V2.1 BPI-MCF`（Stage 7）的 `63.19%` 跃升至 **`81.65%`**（其中斜坡加载 `fresh_b1` 与反向扰动 `fresh_b2` 子集从 `V2.1` 的 `31.0% / 0.0%` 提升至 **`81.9% / 72.2%`**；`b0+b1` 阶跃+斜坡盲测集从 `68.05%` 跃升至 **`86.36%`**）。
   - **36 工况桌面支撑 OOD 集 `pilot_tangent_02` 跃升 `+13.76%`**：在跨预紧、跨摩擦、跨 SDF 增益、斜坡与切向反转的 36 组桌面支撑 OOD 试次上，零样本质心分类准确率从 `Standard MCF` 的 `34.91%` 和 `V2.1 BPI-MCF` 的 `47.21%` 提升至 **`60.97%`**（配合二阶四极矩 Stage 9；其中 `sdf_gain` 达 **`85.83%`** / Ridge **`94.17%`**，未见力臂偏置 `step_unseen_offset` 达 **`82.50%`**，切向反转 `reverse` 从 `40.00%` 提升至 **`61.67%`**）。
2. **创新点 2 & 3（本体-触觉伴随拷贝双头架构 + 防坍缩在线残差强化学习）**：
   - **揭示并解决在线 RL 表征坍缩（Representation Collapse）**：在离线 Warm-Start BC 基础上直接进行无辅助监督的在线残差策略梯度优化（`4_V22_Online_Residual_RL_No_Aux_Collapse`），策略为迎合短期防滑奖励而采取盲目过夹紧捷径，导致盲测集 `fresh_b0+b1` 上的力螺旋解码 $R^2$ 崩溃至 `-1.7774`、动作跟踪误差恶化至 `0.28469`。
   - **防坍缩在线残差 RL（`5_V22_Online_Residual_RL_With_Aux_Proposed`）**：引入跨模态力螺旋回归与剪切方向分类自监督锚定后，在 `fresh_b0+b1` 独立盲测集上将冻结特征探针准确率推高至 **`86.28%`**（`V2.1` 单头为 `81.15%`，无辅助监督双头为 `77.03%`），力螺旋解码 $R^2$ 提升至 **`+0.5382`**（`V2.1` 为 `-0.0225`）。
3. **Isaac Lab 6-DoF 物理闭环抗扰动基准（`Part A`：零载 + `0.20 N @ 15 mm` 四向偏心冲击，0% `Stage 3S` 混杂）**：
   - **受载四向平均滑移再降 `23.5%`**：`6_v2_2_online_residual_rl_proposed` 将受载四向平均指尖相对滑移从 `Stage 3S` 1D 基线的 `1.2303 mm`、`V2.1` 纯策略的 `0.7795 mm` 以及 `V2.2 Warm-Start BC` 的 `0.7349 mm` 进一步压低至 **`0.5964 mm`**（较 `V2.1` 降低 **`23.5%`**，较 `Stage 3S` 降低 **`51.5%`**，较开环 `8.2011 mm` 降低 **`92.7%`**）。
   - **最差方向（`+Fx`）滑移突破 `< 1.9 mm` 关口**：`+Fx` 最坏滑移从 `Stage 3S` 的 `3.7210 mm` 和 `V2.1` 的 `2.1754 mm` 降至 **`1.8606 mm`**（`Warm-Start BC` 达 **`1.8585 mm`**），首次在 `0.20 N @ 15 mm` 全部四个方向上实现 **全方向单次滑移 `< 2.0 mm` 安全线（4/4 达标）**！
   - **受载平均倾角降低 `26.6%` 且保持零静默抖动**：四向平均倾角从 `V2.1` 的 `0.7847°` 降至 **`0.5758°`**，预紧静默期抖动保持 **`0.0000 mm`**（而无辅助监督的坍缩 RL 策略 `4_v2_2_online_rl_no_aux_collapse` 产生 **`2.2485 mm`** 剧烈预紧抖动，最差滑移退化至 `3.0793 mm`）。
4. **Isaac Lab 3 阶段长时序插孔操作基准（`Part B`：连续 135 步预紧 $\to$ 遮挡受扰搬运 $\to$ 插孔倒角反向切向对齐）**：
   - **长时序插孔准入达标率从 `75% (3/4)` 跃升至 `100% (4/4)`**：`6_v2_2_online_residual_rl_proposed` 取得 Stage 1 预紧抖动 **`0.0000 mm`**、Stage 2 遮挡搬运平均滑移 **`0.5950 mm`**（较 `V2.1` 的 `0.7510 mm` 降低 **`20.8%`**）、Stage 3 反转对齐平均滑移 **`0.5677 mm`**（较 `V2.1` 的 `0.7186 mm` 降低 **`21.0%`**）、Stage 3 平均倾角 **`0.5583°`**（较 `V2.1` 的 `0.7185°` 降低 **`22.3%`**）、末端最终对齐误差 **`0.7881 mm`**（最困难的 `+Fx` 方向末端误差降至 `2.5393 mm < 2.60 mm` 倒角捕获容差，对齐滑移 `1.8466 mm < 1.90 mm`，倾角 `1.4544° < 1.50°`），在全部 4 个方向上实现 **`100.0% (4/4)`** 准入达标！

![10阶段 CMQ-BPI-MCF 算子消融](outputs/figures/fig1_10stage_cmq_ablation.png)

![pilot_tangent_02 36工况 OOD 分类拆解](outputs/figures/fig2_pt02_7category_ood.png)

![在线残差强化学习表征坍缩与防坍缩自监督对比](outputs/figures/fig3_online_rl_anticollapse.png)

![Isaac Lab 6控制器闭环与3阶段长时序基准对比](outputs/figures/fig4_isaac_closedloop_and_longhorizon.png)

---

## 二、定量结果总表

### 1. 创新点 1：从原始触觉与 `V2.1 BPI-MCF` 到 `V2.2 CMQ-BPI-MCF` 的 10 阶段算子消融

| 阶段编号与特征名称 | 维度 | 核心物理结构设计 | Dev (`c005–c008`) | 独立盲测 `Fresh12` (Trial) | 独立盲测 `Fresh12` (Inst) | 独立盲测 `b0+b1` (Step+Ramp) | `pilot_tangent_02` (36 OOD) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `0_Raw_State4_16D` | 16D | 双指原始 `[mean, max, cx, cy]` + 一阶差分 | 47.50% | 24.69% | 27.47% | 24.54% | 33.46% |
| `1_Scalar_8D` | 8D | 仅双指压力标量 `[p_mean, p_max]` + 差分 | 25.00% | 28.81% | 31.59% | 28.64% | 26.67% |
| `2_Standard_MCF_32D` | 32D | 滑动窗口局部去中心化 `MCF` | 46.67% | 29.70% | 33.85% | 31.74% | 34.91% |
| `3_Initial_Centered_MCF_32D` | 32D | 锁定初始接触参考帧 $k_{\text{base}}=5$ 去中心化 | 72.50% | 44.92% | 49.30% | 54.62% | 38.97% |
| `4_PerFinger_Preload_Norm_32D` | 32D | + 单指预紧力归一化 $f_{i,0}$ | 72.50% | 44.71% | 49.09% | 54.62% | 39.72% |
| `5_Bilateral_SymAnti_Dipole_16D` | 16D | + 双指对称/反对称正交分解 + 偏心力矩偶极 | 68.33% | 56.97% | 59.98% | 61.29% | 47.29% |
| `6_Static_Tangent_Manifold_24D` | 24D | + 静态切向形变流形（3+4+5 拼接，无动态块） | 95.83% | 62.13% | 64.44% | 66.29% | 46.10% |
| `7_Full_Frozen_BPI_MCF_32D` (`V2.1`) | 32D | 完整 `V2.1 BPI-MCF`（静态流形 24D + 偶极速率 8D） | 97.50% | 63.19% | 65.65% | 68.05% | 47.21% |
| **`8_Creep_ScaleInv_BPI_32D` (`V2.2`)** | **32D** | **+ 法向粘弹性蠕变解耦 + 偶极单位方向场 $\hat{\mathbf{d}}$** | **98.33%** | **81.65%** | **77.12%** | **86.36%** | **59.67%** (Ridge **60.41%**) |
| **`9_CMQ_BPI_MCF_48D` (`V2.2 Full`)** | **48D** | **+ 二阶空间四极矩 $\mathbf{Q}$ + 快慢双包络新息 (16D)** | **98.33%** | **81.65%** | **77.12%** | **86.36%** | **60.97%** (Ridge **59.85%**) |

### 2. 创新点 1B：`pilot_tangent_02` 36 工况桌面支撑 OOD 6 大物理子集拆解

| OOD 扰动子集 | 试次数 / 帧数 | `2_Standard_MCF_32D` | `7_V2.1_BPI_MCF_32D` | `8_V2.2_Creep_ScaleInv_32D` | **`9_V2.2_CMQ_BPI_MCF_48D` (Centroid / Ridge)** | 物理破局机制解释 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `friction` ($\mu \in \{0.8, 1.2\}$) | 8 / 240 | 34.58% | 43.75% | 52.08% | **52.08%** / 45.83% | 单位偶极方向场消除摩擦系数引起的幅值缩放畸变 |
| `preload` ($\pm 20\%$ 预紧偏移) | 8 / 240 | 36.67% | 30.42% | 37.50% | **40.00%** / 35.83% | 法向蠕变基线减法抑制高预紧下的法向压入伪影 |
| `ramp` (慢变斜坡加载) | 4 / 116 | 26.72% | 43.10% | 49.14% | **48.28%** / **50.00%** | 慢速累积新息包络保留准静态斜坡方向积分 |
| `reverse` (加载中途切向反转) | 4 / 120 | 42.50% | 40.00% | 62.50% | **61.67%** / **57.50%** | 快慢双时间尺度新息差分捕捉切向力符号跳变 |
| `sdf_gain` (弹性体增益缩放) | 8 / 240 | 26.67% | 71.25% | 85.83% | **85.83%** / **94.17%** | 尺度不变单位偶极 $\hat{\mathbf{d}}$ 对齐跨增益几何流形 |
| `step_unseen_offset` (未见力臂) | 4 / 120 | 49.17% | 50.83% | 74.17% | **82.50%** / **79.17%** | 二阶空间四极矩 $\mathbf{Q}$ 解耦偏心力臂与纯切向力 |

### 3. 创新点 2 & 3：本体-触觉双头架构与防坍缩在线残差强化学习离线盲测对比（`fresh_b0+b1`）

| 模型编号与名称 | 触觉算子与架构 | 训练范式 | `Clean_Action_MSE` ↓ | `Quiet_Preload_MAE` ↓ | `Latent_Probe_DirAcc` ↑ | `Aux_Head_DirAcc` ↑ | `Wrench_R2` (力螺旋解码) ↑ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `1_V21_SingleHead_NoProprio` | `BPI-MCF` 32D + 单头 (无本体) | 离线 BC + Aux | 0.00535 | 0.02020 | 81.15% | 82.86% | -0.0225 |
| `2_V22_DualHead_NoAux_Supervised` | `CMQ-BPI-MCF` 48D + 本体双头 | 离线 BC (无 Aux) | 0.00518 | 0.03467 | 77.03% | 54.44% | -7.0269 |
| `3_V22_DualHead_WarmStart_BC` | `CMQ-BPI-MCF` 48D + 本体双头 | 离线 Warm-Start BC + Aux | **0.00382** | 0.02637 | 85.43% | 85.43% | +0.4871 |
| `4_V22_Online_Residual_RL_No_Aux_Collapse` | `CMQ-BPI-MCF` 48D + 本体双头 | **在线残差 RL (无 Aux，表征坍缩)** | 0.28469 | 0.58493 | 84.97% | 63.03% | -1.7774 |
| **`5_V22_Online_Residual_RL_With_Aux_Proposed`** | **`CMQ-BPI-MCF` 48D + 本体双头** | **在线残差 RL + 防坍缩 Aux (本文)** | **0.00465** | **0.02066** | **86.28%** | **86.70%** | **+0.5382** |

### 4. Isaac Lab 6-DoF 物理闭环抗扰动基准（`Part A`：零载 + `0.20 N @ 15 mm` 四向偏心剪切冲击）

| 控制器编号与名称 | 动作空间与架构 | 受载四向平均滑移 `LoadedMeanSlip` ↓ | 最差方向滑移 `WorstSlip (+Fx)` ↓ | `-Fx` 滑移 ↓ | `+Fy` 滑移 ↓ | `-Fy` 滑移 ↓ | 受载平均倾角 `MeanTilt` ↓ | 预紧静默段抖动 `PreloadJitter` ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_nominal_open_loop` | 0D 开环保持 | 8.2011 mm | 21.5004 mm | 10.4213 mm | 0.1123 mm | 0.7705 mm | 6.1453° | 0.0000 mm |
| `2_stage3s_1d_grip_only` | 1D `Stage 3S` 握力反射 | 1.2303 mm | 3.7210 mm | 0.3400 mm | 0.1100 mm | 0.7501 mm | 0.9139° | 0.0000 mm |
| `3_v2_1_pure_bpi_3dof_singlehead` | 3-DoF `V2.1` 单头纯策略 | 0.7795 mm | 2.1754 mm | 0.1897 mm | 0.1123 mm | 0.6408 mm | 0.7847° | 0.0000 mm |
| `4_v2_2_online_rl_no_aux_collapse` | 3-DoF `V2.2` 无辅助坍缩 RL | 0.9462 mm | 3.0793 mm | 0.2400 mm | 0.0568 mm | 0.4085 mm | 0.7931° | **2.2485 mm** (失稳) |
| **`5_v2_2_warmstart_bc_dualhead`** | **3-DoF `V2.2` 本体双头 BC** | **`0.7349 mm`** | **`1.8585 mm`** | 0.3384 mm | **0.0368 mm** | 0.7059 mm | **`0.6196°`** | **`0.0000 mm`** |
| **`6_v2_2_online_residual_rl_proposed`** | **3-DoF `V2.2` 本体双头 + 在线残差 RL** | **`0.5964 mm`** | **`1.8606 mm`** | **`0.1766 mm`** | **`0.0520 mm`** | **`0.2966 mm`** | **`0.5758°`** | **`0.0000 mm`** |

### 5. Isaac Lab 3 阶段长时序插孔操作基准（`Part B`：预紧 $\to$ 遮挡搬运 $\to$ 倒角反向切向对齐）

| 控制器编号与名称 | Stage 1 预紧抖动 ↓ | Stage 2 遮挡搬运平均滑移 ↓ | Stage 3 反向对齐平均滑移 ↓ | Stage 3 平均倾角 ↓ | Stage 3 四向平均末端误差 ↓ | 最差方向 (`+Fx`) 末端误差 ↓ | 插孔准入达标率 (`4/4` 方向) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_nominal_open_loop` | 0.0000 mm | 8.3726 mm | 5.1601 mm | 3.7427° | 3.9892 mm | 4.2394 mm | 25.0% (1/4) |
| `2_stage3s_1d_grip_only` | 0.0000 mm | 1.2307 mm | 1.3254 mm | 0.9727° | 1.6716 mm | 4.3668 mm | 75.0% (3/4) |
| `3_v2_1_pure_bpi_3dof_singlehead` | 0.0000 mm | 0.7510 mm | 0.7186 mm | 0.7185° | 0.9627 mm | 2.6992 mm | 75.0% (3/4) |
| **`6_v2_2_online_residual_rl_proposed`** | **`0.0000 mm`** | **`0.5950 mm`** | **`0.5677 mm`** | **`0.5583°`** | **`0.7881 mm`** | **`2.5393 mm`** | **`100.0% (4/4)`** |

---

## 三、目录入口与归档清单

| 文件 / 目录 | 说明 |
| :--- | :--- |
| [verify_three_innovations_v2_2_cmq.py](verify_three_innovations_v2_2_cmq.py) | 离线 10 阶段 `CMQ-BPI-MCF` 算子消融、36 工况 OOD 拆解、本体-触觉双头网络与防坍缩在线残差 RL 训练评测主脚本 |
| [verify_isaac_closedloop_v2_2_residual_rl.py](verify_isaac_closedloop_v2_2_residual_rl.py) | Isaac Lab 6 控制器 6-DoF 物理闭环抗扰动与 3 阶段长时序插孔操作基准测试主脚本 |
| [plot_v2_2_figures.py](plot_v2_2_figures.py) | 论文级对比图表生成脚本（输出 4 组 PNG 与矢量 PDF） |
| [outputs/v2_2_verification_results.json](outputs/v2_2_verification_results.json) | 10 阶段消融、7 类 OOD 拆解、架构对比与在线 RL 训练曲线完整 JSON 报告 |
| [outputs/isaac_closedloop_v2_2_residual_rl_report.json](outputs/isaac_closedloop_v2_2_residual_rl_report.json) | Isaac Lab 6 控制器闭环与 3 阶段长时序逐试次完整 JSON 报告 |
| [outputs/checkpoints/](outputs/checkpoints/) | 训练并冻结的 5 组 PyTorch 模型权重（`.pt`，含 `V21_SingleHead`、`V22_WarmStart_BC`、`V22_Online_RL_No_Aux`、`V22_Online_RL_Proposed`） |
| [outputs/figures/](outputs/figures/) | 4 组高清 PNG 与矢量 PDF 论文级图表 |
| [archive_manifest.json](archive_manifest.json) | 本目录全部脚本、JSON 报告、图表与模型权重的 SHA-256 校验清单 |

---

## 四、复现命令

### 1. 离线复核与重新生成图表（仅需 CPU PyTorch + NumPy + Matplotlib）
在克隆仓库并按 [../stage3s_isaac/README.md](../stage3s_isaac/README.md) 恢复 `stage3s_isaac` 的 NPZ 数据后，于本目录执行：

```powershell
python verify_three_innovations_v2_2_cmq.py
python plot_v2_2_figures.py
```

### 2. 在 Isaac Lab 中重放 6 控制器闭环与 3 阶段长时序基准
需要 Isaac Sim 5.1 / Isaac Lab 环境（`C:\isaacsim_env\Scripts\python.exe`）及原作者 `FlexiTac-IsaacSim-Simulation` 资产目录：

```powershell
& "C:\isaacsim_env\Scripts\python.exe" verify_isaac_closedloop_v2_2_residual_rl.py --headless --device cpu --official-repo C:/isaac_ws/FlexiTac-IsaacSim-Simulation
```
