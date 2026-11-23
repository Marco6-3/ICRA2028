# V4（路线 1）算法蓝图：物理锚定的动作效应接触状态压缩（PAEC）

> **状态（2026-11-21 实测更新）**：基于 Round-1 `ManiWAV/flip` 证伪后的结构性转向。**已完成** RSS-2025 `RDP` 两大完整真机物理数据集（`peel_v3` 60 eps、`wipe_v1.5` 80 eps）域内审计基准及 `wipe ↔ peel` 双向 100% 冻结零样本跨任务迁移（见文末第 4 节已验证结果）；**尚未运行**：闭环仿真/真机成功率、独立外扰真值惊奇度测试与去六维力跨模态测试。  
> **为什么从 V3（纯音频抽象 JEPA）转向 V4（物理锚定接触压缩）？**
> 1. **硬件冗余证伪**：在 `NewGripper` 已配备 500 Hz 腕部六维力 + 1–2 kHz 工具根部应变 + GelSight Mini 的前提下，20 kHz 接触麦克风在撬边/揭膜/卡扣任务中高度疑似冗余且易受频响传递函数偏移影响；因此主传感聚焦于**高信噪比、可校准的力觉 / 应变 / 视触觉 + 6-DOF 本体**（麦克风降级为百元台架可选项）。
> 2. **跳出世界模型“抽象隐变量 $z$ 三难困境”**：Round-1 实测证实，在机器人几百条轨迹尺度下，纯隐空间 JEPA + VICReg 会强行把 33 维传感器噪声撑成单位方差，导致下游少样本 BC 过拟合、输给低秩 `MAE`（有效秩 5.71）；而 `MAE` 重构又记死静态偏置导致 OOD 崩盘。
> 3. **核心解法**：**不再预测自造的抽象潜向量 $\bar{z}_{t+\Delta}$**，而是直接利用机器人/手持采集端**免人工标注、自动记录的真实物理力学演变** $(\Delta \mathbf{F}, \Delta \boldsymbol{\tau}, \Delta \varepsilon, \Delta e^{\text{tac}})$ 作为多步动作条件预测目标，在极紧凑低维瓶颈（$d_z = 12\text{–}16$）内实现**天然零坍缩、天然低秩、无需 VICReg 噪声膨胀**的接触状态压缩！

---

## 1. 核心科学问题

> **单帧瞬时力/触觉读数无法唯一区分当前接触处于弹性加载、粘滑临界、卸载反转、刚性顶边还是高粘附剥离状态。能否仅利用机器人自身记录的连续接触力/触觉与本体历史，以“多步连续动作条件物理力学净响应（Action-Conditioned Wrench/Compliance Delta）”为免标注自监督目标，将高维非马尔可夫接触历史压缩为极紧凑低维状态 $z_t \in \mathbb{R}^{12\text{–}16}$，使下游少样本策略显著提升、实现跨任务/跨刚度泛化，并以校准物理力残差直接检出临界相变？**

---

## 2. 数学定义与网络结构（PAEC: Physics-Anchored Action-Effect Contact Compression）

### 2.1 输入与严格因果压缩瓶颈（零未来泄露）
在控制步 $t$，系统接收严格因果的历史观测：
- **接触力学流 $o^{\text{con}}_\tau$**：校准后的局部坐标系 6D 力/力矩 $\mathbf{w}_\tau = [\mathbf{F}_\tau, \boldsymbol{\tau}_\tau] \in \mathbb{R}^6$（`NewGripper` 中可并入 1–2 kHz 根部应变特征 $\varepsilon_\tau$）及可选视触觉嵌入 $\mathbf{e}^{\text{tac}}_\tau$（如 GelSight Mini PCA/CNN 嵌入）；
- **已执行局部本体增量 $\Delta q_\tau$**：从 $\tau-1 \to \tau$ 的末端局部相对位移、旋转与夹爪开度增量（$\tau=0$ 时置零）。

因果编码器 $E_\theta$（2 层严格因果 GRU，输出经小维度线性瓶颈投影）将历史 $H_t = \{(o^{\text{con}}_\tau, \Delta q_\tau)\}_{\tau \le t}$ 压缩为极紧凑隐状态：

$$z_t = E_\theta(H_t) \in \mathbb{R}^{d_z}, \quad d_z \in \{12, 16\}.$$

> **与 V3 的本质区别**：
> - **取消 EMA 目标编码器与 VICReg 强方差膨胀**：不再把 64 维撑成高维白噪声；
> - **严格低维物理瓶颈**：只给下游策略提供 12–16 维高度凝练的接触力学状态，防止 10%–30% 少样本下游过拟合。

---

### 2.2 零动作基线 + 连续动作净力学响应预测器（核心物理公式）
对多时间尺度预测步长 $\Delta \in \mathcal{D} = \{1, 3, 6\}$ 帧，令真实物理接触增量目标为：

$$\Delta \mathbf{y}_{t, \Delta} = \mathbf{y}_{t+\Delta} - \mathbf{y}_t$$

其中 $\mathbf{y}_t$ 为局部坐标系下的 6D 接触力螺旋 $\mathbf{w}_t$（当存在视触觉或应变时同步预测其物理增量 $\Delta \mathbf{e}^{\text{tac}}_{t,\Delta}$ / $\Delta \varepsilon_{t,\Delta}$）。

预测器显式分解为**零动作自然力松弛/漂移**与**连续动作引起的净力学响应**：

$$\widehat{\Delta \mathbf{y}}_{t, \Delta} = \underbrace{b_\phi(z_t, \Delta)}_{\text{零动作蠕变/松弛/外扰基线}} + \underbrace{g_\phi(z_t, a_{t:t+\Delta}, \Delta)}_{\text{连续动作净力学效应}}$$

结构严格保证零动作恒等为零：

$$g_\phi(z_t, a, \Delta) = \tilde{g}_\phi(z_t, a, \Delta) - \tilde{g}_\phi(z_t, \mathbf{0}, \Delta) \implies g_\phi(z_t, \mathbf{0}, \Delta) \equiv \mathbf{0}.$$

**物理直觉**：
- $g_\phi(z_t, a, \Delta)$ 关于 $a$ 的局部雅可比 $\left.\frac{\partial g_\phi}{\partial a}\right|_{a=0}$ 在物理上就是**历史依赖的局部方向刚度 / 柔顺张量 $K_{\text{contact}}(H_t)$**！
- 在自由空间中，$g_\phi \approx \mathbf{0}$；
- 在贴面滑动中，法向推压产生高 $\Delta F_z$，切向滑动受库仑摩擦饱和约束；
- 在顶死脆边或加载反转临界点，正向微动与反向微动给出**完全非对称的力学响应**（直接解决 Codex 提出的非马尔可夫路径依赖与反转辨识问题，且无需离散随机抖动！）。

---

### 2.3 免人工标注复合损失（零坍缩 + 防静态退化）

1. **多步物理力学净响应损失（主损失）**：
   $$\mathcal{L}_{\text{mech}} = \frac{1}{|\mathcal{D}|} \sum_{\Delta \in \mathcal{D}} w_t \cdot \operatorname{SmoothL1}\!\left(\widehat{\Delta \mathbf{y}}_{t,\Delta},\; \Delta \mathbf{y}_{t,\Delta}\right)$$
   其中 $w_t \ge 1$ 按接触活跃度（$\|\mathbf{w}_t\|$ 与 $\|\Delta \mathbf{w}_t\|$）与低速 Hold 锚定加权（低速段自动锚定 $b_\phi$ 吸收松弛与漂移）。
2. **可选接触期逆动力学正则（Contact-Active IDM）**：
   仅在接触活跃样本（$\|\mathbf{w}_t\| > \epsilon_F$）上，由 $(z_t, \Delta \mathbf{y}_{t,\Delta})$ 反推局部动作 $\hat{a}_{t:t+\Delta}$，强化 $z_t$ 对接触力学因果方向的敏感度（自由空间不施加，避免无解噪声）。
3. **绝对静态偏置鲁棒性（Causal Preload / Zero-Drift Regularization）**：
   训练期对每条轨迹的力/力矩输入施加轻微随机零漂偏置（模拟传感器温漂或换物体自重偏置，但不改变差分目标 $\Delta \mathbf{y}_{t,\Delta}$！），强迫编码器学会**对静态零点漂移不变、只依赖因果演化与相对加载历史**！

---

### 2.4 虚拟方向刚度探针 & 校准物理惊奇度（零真机抖动！）

训练冻结后，PAEC 天然为下游策略与高频护栏提供两项独有能力：
1. **虚拟方向力学探针（Virtual Directional Compliance Readout）**：
   在不让真实机器人做任何物理抖动的前提下，将冻结的 $z_t$ 代入 $g_\phi(z_t, \pm \delta \mathbf{e}_k, \Delta_1)$（例如沿 $\pm x, \pm y, \pm z$ 六个方向施加单位虚拟微位移），即可**零成本读出当前接触的 6 向虚拟反力/刚度特征**（例如“往前推刚度爆表但往后退卸载”），直接与 $z_t$ 拼接喂给下游策略（消融对比 `z_t only` vs `z_t + virtual_probe`）！
2. **物理单位校准的惊奇度（Calibrated Wrench Surprise）**：
   $$r_t = \left\| \Delta \mathbf{y}_{t-\Delta_1, \Delta_1} - \widehat{\Delta \mathbf{y}}_{t-\Delta_1, \Delta_1} \right\|_2$$
   直接以物理力/力矩误差衡量接触是否发生未预期的相变（碰边卡死、滑移起跳、剥离断脱、卡扣 5 ms 阻力骤降），不再被高频音频噪声淹没！

---

## 3. 公开数据集验证与消融计划（先在 `nvidia-desktop` 跑穿，再接 `NewGripper`）

### 3.1 公开数据集（真实 6D F/T + GelSight 视触觉 + 6-DOF 连续动作）
采用 **Reactive Diffusion Policy（RSS 2025，`WendiChen/reactive_diffusion_policy_dataset`）** 的三项接触丰富操作任务（均含真实机械臂 6D Wrench + GelSight 视触觉 + TCP 位姿）：
1. **`peel_v3`（削皮/剥离）**：与 `NewGripper` 剥离/撬边高度同构；
2. **`wipe_v1.5`（擦拭）**：连续往复滑动接触与方向反转（完美检验非马尔可夫路径依赖与赶气泡工况）；
3. **`lift_v2`（柔顺抬举/抓取）**：抗偏心/防滑移握持，可做**跨任务预训练 $\to$ 未见任务迁移测试**！

### 3.2 严格对照与消融矩阵（全部同数据、同下游策略、3 seeds、严格 episode 划分）

| 组别 | 表征方法 | 检验目标 |
| :--- | :--- | :--- |
| **B0** | `ProprioOnly`（仅位姿历史，无力/触觉） | 确认任务依赖接触感知 |
| **B1** | `RawFT_End2End`（原始力/触觉窗端到端接入下游 BC） | 对比预训练压缩 vs 端到端学习 |
| **B2** | `MAE_Recon`（同容量 GRU 做力/触觉信号掩码重构） | 对比“重构” vs “动作条件物理力学预测” |
| **B3** | `Latent_JEPA_VICReg`（同容量抽象隐空间 JEPA + VICReg） | 复现并直接对比“抽象 $z$ 墙” vs “物理锚定压缩” |
| **B4** | `NoBG_Forward`（去掉 $b_\phi + g_\phi$ 分解的普通力预测） | 验证零动作基线分解的净增益 |
| **B5** | `SingleFrame / ShortWin`（单帧/短窗力预测） | 借用 Codex 诊断验证**长历史递归记忆**的真实必要性 |
| **Ours** | **`PAEC`（紧凑 $d_z=16$ 物理锚定压缩 + 可选虚拟探针）** | 检验 `10% / 30% / 100%` 少样本下游 BC、接触活跃段、反转段、以及**传感器零点漂移 / 刚度缩放 OOD** 下的全面胜出！ |

### 3.3 明确 Go / No-Go 红线
- 若在 `peel` / `wipe` 上 `PAEC` 的紧凑冻结表征在 `10%/30%` 少样本下游 BC 及零漂/刚度偏移 OOD 上仍打不过 `B1 (RawEnd2End)` 或 `B2 (MAE)`，必须如实记录并剖析原因，不硬凑结论。

---

## 4. 已完成实测验证记录（2026-11-21，离线真机公开数据集）

全部实验均在 `nvidia-desktop`（Jetson Orin CUDA, PyTorch 2.5, 严格 FP32 `allow_tf32=False`）运行，通过前缀严格因果误差 `z = 0.0`、零动作恒等 `max|g(z, 0)| = 0.0`、`36/36` 全参数非零有限梯度审计及 SHA-256 冻结权重审计：

1. **域内任务 1 — `peel_v3`（60 eps / 25,710 帧，[完整报告](../experiments/v4_paec_contact_compression/outputs/rdp_peel_v4/RDP_PEEL_V4_RESULTS.md)）**：
   - `B1_RawFT_End2End`、`B2_MAE`、`B3_LatentJEPA`（有效秩 `40.21`）全部劣于 `B0_ProprioOnly`；
   - `Ours_PAEC`（有效秩 `10.98`）前向 3 步 6D 力预测 Clean $R^2 = \mathbf{0.4620}$、Drift OOD $R^2 = \mathbf{0.3759}$（第 1）；下游 BC `30%` `PAEC_z16 = 0.59032`、`100%` `PAEC_Probe = 0.54654`（反转段 `0.96928` 第 1、Drift OOD `0.54970` 仍优于 `B0 0.55349`）。
2. **域内任务 2 — `wipe_v1.5`（80 eps / 37,267 帧，[完整报告](../experiments/v4_paec_contact_compression/outputs/rdp_wipe_v4/RDP_WIPE_V4_RESULTS.md)）**：
   - 完整复现上述规律：`B1 / B2 / B3`（有效秩 `41.79`）全部劣于 `B0`；
   - `Ours_PAEC`（有效秩 `9.73`）前向 Clean $R^2 = \mathbf{0.6343}$、Drift OOD $R^2 = \mathbf{0.5043}$（去漂移增强跌至 `0.1932`）；下游 BC `10%` 仅 6 条轨迹 `PAEC_Probe = 0.55967`（第 1）、`30%` `PAEC_z16 = 0.44408`（第 1，反转段 `0.63370` 第 1）、`100%` 反转段 `PAEC_z16 = 0.56202`（第 1）。
3. **双向 100% 冻结零样本跨任务迁移 — `wipe ↔ peel`（[完整报告](../experiments/v4_paec_contact_compression/outputs/RDP_CROSS_TASK_V4_RESULTS.md)）**：
   - `wipe → peel`：`src_norm` 纯零样本下反转段 6D 力预测 $R^2 = \mathbf{+0.2626}$（追平域内 `0.2637`），30% BC `Xfer_PAEC_z16 = 0.58805`（超越 `B0 0.59301` 与域内 `0.59032`）；`tgt_norm` 100% BC `Xfer_PAEC_z16 = 0.54915`（反转段 `0.96816` 第 1）。
   - `peel → wipe`（3-DOF 平动 $\to$ 6-DOF 旋转不对称压测）：`src_norm` 10% `Xfer_PAEC_z16 = 0.56327`（超 `B0 0.56462`）、100% `0.39521`（零漂 `0.39711` 仍超 `B0 0.40057`）；`tgt_norm` 100% `Xfer_PAEC_z16 = 0.39557` 是**唯一战胜 `B0`** 的冻结模型（去掉 `b+g` 退化为 `0.40297`，去掉 `DriftAug` 退化为 `0.40077`）。
   - **架构分工实证**：域内使用 `z + 虚拟探针` 更优；跨任务迁移时直接迁移任务无关的 16D 状态 `z_16` 全面优于绑定源动作空间的探针头。
4. **未闭环 / 待检验项（严格不提前主张）**：
   - 上述离线验证通过了第 3.3 节的 Go/No-Go 红线，但尚不构成闭环控制成功率证据；
   - 惊奇度指标因公开数据缺少独立外扰/断裂真值标签（原脚本用 `||d²w||` 与 `raw_dw` 同义反复），待闭环仿真或 `NewGripper` 台架独立标签检验。
