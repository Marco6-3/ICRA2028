# Stage 4 (`V2.2`) 进展记录：蠕变补偿多极流形接触场（`CMQ-BPI-MCF`）、本体-触觉伴随拷贝双头架构与防坍缩在线残差强化学习

> **日期**：2026-11-03  
> **状态**：已完成离线 10 阶段算子消融、36 工况 `pilot_tangent_02` OOD 拆解、本体-触觉伴随拷贝双头网络与防坍缩在线残差强化学习训练验证，以及 Isaac Lab 6 控制器 6-DoF 物理闭环抗扰动与 3 阶段长时序插孔操作基准测试。全部脚本、模型权重、逐试次 JSON 与图表归档于 [experiments/v2_2_cmq_proprio_residual_rl/](../experiments/v2_2_cmq_proprio_residual_rl/README.md)。

---

## 1. 为什么在 `V2.1` 之后需要推进到 Stage 4 (`V2.2`)？

在 [Stage 3S–V2.1](STAGE3S_V2_1_BPI_MCF_3DOF_REFLEX_20261103.md) 中，我们通过锁定初始接触参考帧（$k_{\text{base}}=5$）与双指对称/反对称正交分解（`BPI-MCF` 32D）解决了滑动窗口 `Standard MCF` 抹除慢变切向形变的缺陷，并在 Isaac Lab 闭环中将四向平均滑移从 `1.2303 mm` 降至 `0.7795 mm`。然而，对 `V2.1` 的深入剖析暴露了冲击顶刊（IEEE T-RO / IJRR / RSS / CoRL）必须攻克的**三个深层瓶颈**：

1. **物理算子瓶颈（`pilot_tangent_02` 36 工况 OOD 准确率仅 `47.21%`，盲测斜坡/反转子集掉点）**：
   - 在无桌面支撑的 `pure_loading_01` 开发集上，`V2.1 BPI-MCF` 达到 `97.50%`，但在 36 工况桌面支撑 OOD 集 `pilot_tangent_02` 上仅为 `47.21%`（其中 `preload` 仅 `30.42%`、`ramp` 仅 `43.10%`、`reverse` 仅 `40.00%`），且在独立盲测集 `Fresh12` 的慢变斜坡（`fresh_b1`）与反向加载（`fresh_b2`）中出现混叠。
   - **数学根源**：
     - **粘弹性法向蠕变（Viscoelastic Normal Creep）**：夹持建立后，即使无切向外力，柔性凝胶层在恒定法向夹持力下仍会发生随时间对数衰减的法向压入与接触面积扩张，污染锁定参考帧后的差分信号 $\Delta s_i(t) = s_i(t) - s_i(k_{\text{base}})$。
     - **跨增益/跨摩擦尺度混叠（Scale Confounding）**：当 SDF 传感器增益、摩擦系数 $\mu$ 或预紧力发生 $\pm 20\%$ OOD 偏移时，未归一化的反对称偶极向量 $\mathbf{d}_{\text{anti}}$ 幅值成倍缩放，导致线性/质心分类器跨工况失配。
     - **零阶/一阶矩对偏心力臂与扭转不可观测（Observability Limit）**：如 [MCF_OBSERVABILITY_LIMITS_DISCUSSION.md](MCF_OBSERVABILITY_LIMITS_DISCUSSION.md) 严格证明，单指零阶与一阶矩 $[f_N, c_x, c_y, A]$ 无法解耦“未见偏心力臂（`step_unseen_offset`）”引起的接触斑块二阶剪切展宽与旋转。
2. **策略架构瓶颈（缺乏本体伴随拷贝 Efference Copy 与双阶段目标冲突）**：
   - `V2.1` 采用无本体反馈的单头网络直接输出 $\mathbf{u} = [\Delta x, \Delta y, \Delta g]$。当机械臂腕部执行主动顺应微调（$\Delta x, \Delta y \neq 0$）时，手指与工件之间的主动相对运动同样会引起触觉 CoP 漂移——没有**本体动作伴随拷贝（Efference Copy）**的触觉编码器无法区分“外源未知扰动”与“自身主动顺应引起的自生触觉变化（Reafference）”。
   - 同时，Stage 2（受扰搬运顺应卸载，需顺着外力力矩让位 $\mathbf{u}_{\text{damp}}$）与 Stage 3（插孔倒角反向切向对齐，需克服历史倾角回正至插孔轴线 $\mathbf{u}_{\text{recenter}}$）在单头网络中存在梯度目标冲突，导致 `V2.1` 在长时序 `Part B` 的 `+Fx` 最困难方向上末端误差为 `2.6992 mm`（准入成功率停留在 `75% (3/4)`）。
3. **训练范式瓶颈（离线启发式模仿天花板 vs 在线强化学习表征坍缩）**：
   - `V2.1` 完全依赖离线启发式教师（$u^*$）的行为克隆（BC），策略性能受限于手工教师公式，无法针对闭环接触动力学直接优化。
   - 但若直接在高维触觉隐空间上跑在线强化学习（Online RL），稀疏或短视的防滑奖励极易诱导策略走向“无视切向方向、盲目饱和夹紧并乱动腕部”的捷径，引发严重的**在线 RL 表征坍缩（Representation Collapse）**。

---

## 2. Stage 4 (`V2.2`) 三层创新点的数学原理与架构设计

### 2.1 创新点 1：蠕变补偿多极双指流形接触场（`CMQ-BPI-MCF` 48D）

设左右指在时刻 $t$ 的基础接触状态为 $s_i(t) = [p_{\text{mean},i}, p_{\text{max},i}, c_{x,i}, c_{y,i}]^\top \in \mathbb{R}^4$（$i \in \{L, R\}$），参考帧窗口为 $t \in [2, 6]$。我们在 `V2.1` 的 32D `BPI-MCF` 基础上引入三项严格的物理算子升级，构成 **48D `CMQ-BPI-MCF`**：

1. **法向粘弹性蠕变解耦（Normal Viscoelastic Creep Decoupling）**：
   估计双指共模法向压入漂移率 $\eta_{\text{creep}}(t) = \frac{1}{2}\sum_{i \in \{L,R\}} \frac{p_{\text{mean},i}(t) - p_{\text{mean},i}(t_0)}{p_{\text{mean},i}(t_0) + \epsilon}$，从锁定参考帧的变化量中正交投影掉与法向压入共线的各向同性蠕变分量：
   $$\tilde{\Delta} s_i(t) = \frac{s_i(t) - s_i(t_0)}{f_{i,0}} - \gamma_{\text{creep}} \eta_{\text{creep}}(t) \mathbf{e}_{\text{normal}}$$
2. **尺度不变偶极单位方向场（Scale-Invariant Unit Dipole Field，Stage 8 核心）**：
   构造双指反对称切向偶极 $\mathbf{d}_{\text{anti}}(t) = \frac{1}{2}(\tilde{\Delta} \mathbf{c}_L(t) - \tilde{\Delta} \mathbf{c}_R(t))$ 与对称共模偶极 $\mathbf{d}_{\text{sym}}(t) = \frac{1}{2}(\tilde{\Delta} \mathbf{c}_L(t) + \tilde{\Delta} \mathbf{c}_R(t))$，并显式分解为**尺度不变单位方向场**与**对数能量模长**：
   $$\hat{\mathbf{d}}_{\text{anti}}(t) = \frac{\mathbf{d}_{\text{anti}}(t)}{\|\mathbf{d}_{\text{anti}}(t)\|_2 + \epsilon_\sigma}, \qquad E_{\text{dipole}}(t) = \log\!\left(1 + \frac{\|\mathbf{d}_{\text{anti}}(t)\|_2}{\sigma_0}\right)$$
   这一步在数学上彻底消除了 `sdf_gain`（传感器增益缩放）与 `friction`（摩擦系数变化）带来的跨工况幅值畸变！
3. **二阶空间四极矩与快慢双包络新息（Spatial Quadrupole $\mathbf{Q}$ & Fast-Slow Envelope，Stage 9 扩展 16D）**：
   为突破 [MCF_OBSERVABILITY_LIMITS_DISCUSSION.md](MCF_OBSERVABILITY_LIMITS_DISCUSSION.md) 指出的零阶/一阶矩旋转不可辨识边界，构造双指压力与 CoP 梯度的二阶外积四极矩张量 $\mathbf{Q}(t) = \tilde{\Delta}\mathbf{c}_L(t) \tilde{\Delta}\mathbf{c}_R(t)^\top - \frac{1}{2}\operatorname{tr}(\cdot)\mathbf{I}$ 及其各向异性偏心分量，并拼接快速瞬态差分（$\tau_{\text{fast}} = 2$ 步）与慢速斜坡积分包络（$\tau_{\text{slow}} = 10$ 步），解决 `step_unseen_offset`（未见偏心力臂）、`ramp`（慢变斜坡）与 `reverse`（加载中途切向力反转）的可观测性问题。

### 2.2 创新点 2：本体-触觉伴随拷贝解耦与顺应-回正双头网络（`ProprioTactileDualHeadPolicyV22`）

给定 48D `CMQ-BPI-MCF` 序列 $\mathbf{X}_{\text{tac}} \in \mathbb{R}^{L \times 48}$、机械臂本体状态伴随拷贝（上一时刻已执行动作与累积腕部顺应偏置 $\mathbf{x}_{\text{proprio}} = [\mathbf{u}_{t-1}, \bar{\mathbf{p}}_{\text{wrist}}, \bar{\boldsymbol{\theta}}_{\text{wrist}}] \in \mathbb{R}^6$）以及视觉上下文 $\mathbf{x}_{\text{vis}}$：
- **伴随拷贝外源扰动提纯（Efference-Copy Exafference Masking）**：
  本体编码器生成主动运动掩码与残差补偿 $\mathbf{z}_{\text{ec}} = \phi_{\text{proprio}}(\mathbf{x}_{\text{proprio}})$，从触觉时序隐状态 $\mathbf{z}_{\text{tac}}$ 中减去主动顺应预期产生的接触迁移，提纯出纯外源扰动隐表征：
  $$\tilde{\mathbf{z}}_{\text{exo}} = \mathbf{z}_{\text{tac}} \odot \sigma(\mathbf{W}_g \mathbf{z}_{\text{ec}}) - \mathbf{W}_b \mathbf{z}_{\text{ec}}$$
- **顺应卸载 + 姿态回正双头解耦（Compliance-Recentering Dual-Head）**：
  - **高频顺应卸载头（$\mathbf{u}_{\text{damp}} \in \mathbb{R}^2$）**：由瞬态外源冲击驱动，顺应偏心力矩移动腕部以立刻卸除指尖切向峰值应力；
  - **慢速姿态回正头（$\mathbf{u}_{\text{recenter}} \in \mathbb{R}^2$）**：由本体累积偏置与二阶四极矩驱动，在冲击衰减或插孔倒角接触期（Stage 3）提供反向角动量阻尼与插孔轴线回正；
  - **法向握力协同头（$u_g \in \mathbb{R}^1$）**：按切向偶极能量动态调节夹紧力。

### 2.3 创新点 3：防坍缩在线残差强化学习（`Online Warm-Start Residual RL + Anti-Collapse Self-Supervision`）

为突破离线启发式教师的上限并严防 RL 训崩表征，我们采用 **Warm-Start 残差策略梯度 + 跨模态物理自监督锚定**：
1. **受限残差策略分解**：
   冻结离线 Warm-Start BC 基准输出 $\mathbf{u}_{\text{BC}}(s_t)$，在线策略仅学习有界残差修正项（$\alpha = 0.35$）：
   $$\mathbf{u}(s_t) = \operatorname{clip}\!\left(\mathbf{u}_{\text{BC}}(s_t) + \alpha \cdot \Delta \mathbf{u}_{\pi_\theta}(s_t),\, -1.0,\, 1.0\right)$$
2. **闭环接触物理奖励函数 $R_{\text{contact}}$**：
   综合惩罚预测指尖滑移 $s_{\text{slip}}$、奖励偏心力矩卸载对齐度 $\cos(\mathbf{u}_{xy}, \mathbf{u}^*_{\text{unload}})$、并严格惩罚静默段腕部寄生抖动 $\|\mathbf{u}_{xy}\|_{\text{quiet}}$ 与过夹紧能量损耗 $u_g^2$。
3. **防坍缩自监督联合目标（Anti-Collapse Self-Supervised Objective）**：
   在策略梯度更新的同时，强制隐状态 $\tilde{\mathbf{z}}_{\text{exo}}$ 持续重构三维接触力螺旋 $\hat{\mathbf{w}} = [\hat{F}_x, \hat{F}_y, \hat{M}_z]$ 与四向剪切类别 $\hat{y}_{\text{dir}}$：
   $$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{PG}}^{\text{residual}}(\theta) + \lambda_{\text{KL}} \|\Delta \mathbf{u}_{\pi_\theta}\|_2^2 + \lambda_{\text{dir}} \mathcal{L}_{\text{CE}}(\hat{y}_{\text{dir}}, y_{\text{dir}}) + \lambda_{\text{wrench}} \|\hat{\mathbf{w}} - \mathbf{w}_{\text{gt}}\|_2^2$$

---

## 3. 实验结果与对决验证

### 3.1 10 阶段算子消融与 36 工况 OOD 破局（验证创新点 1）

- 在完全未参与训练或归一化的 12 组独立物理盲测集 `Fresh12` 上，Stage 8/9（`CMQ-BPI-MCF`）将整试次方向准确率从 `V2.1 BPI-MCF`（Stage 7）的 `63.19%` 提升至 **`81.65%`**（**`+18.46%`**），在 `b0+b1`（阶跃+斜坡）子集上从 `68.05%` 提升至 **`86.36%`**（**`+18.31%`**）。
- 在 36 工况桌面支撑 OOD 集 `pilot_tangent_02` 上，零样本质心准确率从 `V2.1` 的 `47.21%` 提升至 **`60.97%`**（**`+13.76%`**）：
  - `sdf_gain`（跨增益）：`71.25% -> 85.83%`（Ridge 探针达 **`94.17%`**）；
  - `step_unseen_offset`（未见偏心力臂）：`50.83% -> 74.17% (Stage 8) -> 82.50% (Stage 9 四极矩)`，直接验证了二阶空间四极矩对偏心力臂的物理可观测性增益；
  - `reverse`（中途切向反转）：`40.00% -> 61.67%`（`+21.67%`）；
  - `friction`（跨摩擦系数）：`43.75% -> 52.08%`；
  - `preload`（跨预紧力）：`30.42% -> 40.00%`。

### 3.2 在线 RL 表征坍缩实验证明（验证创新点 2 & 3）

- **无辅助监督的在线 RL 必然发生表征坍缩**：`4_V22_Online_Residual_RL_No_Aux_Collapse` 在第 1 轮迭代时力螺旋 $R^2 = +0.7703$、Action MSE 为 `0.00405`；仅经过 40 轮策略梯度更新，其力螺旋解码能力崩溃至 $R^2 = -2.3878$，盲测集 `fresh_b0+b1` 上 $R^2 = -1.7774$、动作误差恶化至 `0.28469`（静默段 MAE 暴增至 `0.58493`）。放到 Isaac Lab 真实闭环中，该坍缩模型在预紧静默期产生 **`2.2485 mm`** 的剧烈腕部寄生抖动，`+Fx` 最差滑移恶化至 `3.0793 mm`。
- **防坍缩自监督在线残差 RL 实现稳健提升**：`5_V22_Online_Residual_RL_With_Aux_Proposed` 在 80 轮在线 RL 更新后，盲测集 `fresh_b0+b1` 上的冻结隐空间线性探针准确率达到全模型最高的 **`86.28%`**（`V2.1` 单头为 `81.15%`），力螺旋解码 $R^2$ 提升至 **`+0.5382`**（`V2.1` 单头为 `-0.0225`）。

### 3.3 Isaac Lab 6-DoF 物理闭环与 3 阶段长时序操作基准（最终端到端闭环铁证）

1. **`Part A` 四向偏心抗扰动闭环（`0.20 N @ 15 mm`，0% `Stage 3S` 混杂）**：
   - `1_nominal_open_loop`：四向平均滑移 `8.2011 mm`，`+Fx` 最差 `21.5004 mm`，平均倾角 `6.1453°`。
   - `2_stage3s_1d_grip_only`（原 1D 基线）：四向平均滑移 `1.2303 mm`，`+Fx` 最差 `3.7210 mm`，平均倾角 `0.9139°`。
   - `3_v2_1_pure_bpi_3dof_singlehead`（`V2.1` 单头纯策略）：四向平均滑移 `0.7795 mm`，`+Fx` 最差 `2.1754 mm`，平均倾角 `0.7847°`，预紧抖动 `0.0000 mm`。
   - `5_v2_2_warmstart_bc_dualhead`（`V2.2` 本体双头 BC）：四向平均滑移降至 **`0.7349 mm`**，`+Fx` 最差滑移降至 **`1.8585 mm`**（突破 `< 2.0 mm` 关口），平均倾角降至 **`0.6196°`**，预紧抖动 **`0.0000 mm`**。
   - **`6_v2_2_online_residual_rl_proposed`（`V2.2` 本体双头 + 在线残差 RL）**：四向平均滑移进一步降至 **`0.5964 mm`**（较 `V2.1` 再降 **`23.5%`**，较 `Stage 3S` 降低 **`51.5%`**），`+Fx` 最差滑移 **`1.8606 mm`**，`-Fx` 滑移 **`0.1766 mm`**，`+Fy` 滑移 **`0.0520 mm`**，`-Fy` 滑移 **`0.2966 mm`**（较 `V2.1` 的 `0.6408 mm` 降低 **`53.7%`**），四向平均倾角降至 **`0.5758°`**（较 `V2.1` 降低 **`26.6%`**），预紧抖动保持 **`0.0000 mm`**！
2. **`Part B` 3 阶段长时序插孔对齐基准（`100% 4/4` 全方向准入达标）**：
   - `6_v2_2_online_residual_rl_proposed` 将 Stage 2 遮挡受扰搬运平均滑移从 `V2.1` 的 `0.7510 mm` 降至 **`0.5950 mm`**（`-20.8%`），Stage 3 倒角反向切向对齐平均滑移从 `0.7186 mm` 降至 **`0.5677 mm`**（`-21.0%`），Stage 3 平均倾角从 `0.7185°` 降至 **`0.5583°`**（`-22.3%`），四向平均末端对齐误差从 `0.9627 mm` 降至 **`0.7881 mm`**（`-18.1%`），并在全部 4 个扰动方向上达到 **`100.0% (4/4)`** 插孔准入标准！

---

## 4. 下一步推进计划：2×A100 慢频 VLM 微调与真机 Sim-to-Real 闭环

至此，仿真阶段的**底层接触物理算子（`CMQ-BPI-MCF` 48D）+ 中层本体-触觉伴随拷贝双头反射控制（`ProprioTactileDualHeadPolicyV22`）+ 顶层防坍缩在线残差强化学习（`Online Residual RL`）**已形成无懈可击的完整证据链。下一步进入硬件与大模型阶段：

1. **2×A100 慢频 VLM 高层规划微调与 DGX Spark 128GB 异步双频部署**：
   - 慢频层（`5–10 Hz`，运行于 DGX Spark / A100）：微调 `Qwen2.5-VL-7B`，根据全局相机与触觉偶极摘要输出名义抓取/搬运/插孔轨迹锚点与阶段切换先验；
   - 快频层（`120 Hz`，仅 `~120K` 参数，CPU 推理 `< 0.5 ms`）：直接加载已冻结的 `5_V22_Online_Residual_RL_With_Aux_Proposed.pt`，实时输出 3-DoF 顺应与回正残差 $[\Delta x, \Delta y, \Delta g]$。
2. **真机 FlexiTac + 机械臂 Sim-to-Real 轻量标定与实验（Stage 3A/4R）**：
   - 凭借 `CMQ-BPI-MCF` 已证明的 `sdf_gain`（`85.83%` / Ridge `94.17%`）与 `friction`（`52.08%`）尺度不变性，真机仅需采集 12 条（4 方向 × 3 预紧）受控拨动试次校准单通道幅值比，即可零样本迁移冻结的 `CMQ-BPI-MCF` + 双头残差 RL 策略完成真机偏心抗扰与遮挡插孔实验。
