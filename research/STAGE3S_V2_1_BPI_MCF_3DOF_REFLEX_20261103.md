# 2026-11-03 Stage 3S–V2.1 进展记录：从单维握力反射到 `BPI-MCF`、防坍缩多模态融合与 3-DoF 闭环反射

日期：2026-11-03。本文件记录在 [2026-10-13 Stage 3S 仿真闭环](STAGE3S_SIM_REFLEX_20261013.md) 与 [2026-10-15 等预紧触发研究](../experiments/stage3s_alignment/README.md) 基础上完成的 **V2.1 体系升级与实验验证**。既有 Stage 1、Stage 2P 及早期 Stage 3S 归档原样保留；本轮全部代码、权重、JSON 报告与图表归档于 **[experiments/v2_1_bpi_mcf_3dof_reflex](../experiments/v2_1_bpi_mcf_3dof_reflex/README.md)**。

---

## 1. 为什么对 2026-10-10 冻结范围进行证据驱动的修订？

前序实验（`stage3s_isaac` 与 `stage3s_alignment`）暴露了原冻结方案的三个物理与控制瓶颈：

1. **滑动窗口 `Standard MCF` 抹除准静态切向偏移**：原 `MCF` 对窗口均值或相邻帧做差分，在持续/慢变切向载荷下会把稳态弹性形变当作基线滤除，导致在 12 组独立盲测集 `Fresh12` 上四向切向分类准确率仅为 **`33.9%`**（接近 `25%` 随机水平）。
2. **单自由度夹爪收紧（$\Delta g$）的物理上限**：仅靠锁紧夹爪只能提高法向正压力 $F_n$，无法主动顺应外部倾覆力矩 $M_x, M_y$。在最困难的 `0.20 N +Fx`（15 mm 偏心）工况下，原 1D `Stage 3S` 控制器仍存在 **`2.697 mm ~ 3.721 mm`** 滑移，且伴随较高的握持力积分成本。
3. **高预紧下的零载误触发与多模态表征坍缩**：原 `stage3s_alignment` 的 CoP 噪声门限在高预紧（`4.42 N`）零载条件下于 `0.6000 s` 发生误触发；同时，当引入视觉（腕部/全局位姿代理）构建长时序策略时，朴素拼接网络会出现严重的**视觉捷径（Visual Shortcutting）与触觉表征坍缩**，一旦发生视觉遮挡或接触切向力反转即产生反向错误动作。

因此，本轮研究在 **[研究范围修订与冻结稿](ICRA2028_MINIMAL_PAPER_FREEZE.md)** 中正式将系统升级为三项相互咬合的核心机制。

---

## 2. 三项核心机制与验证结果

### 2.1 创新点 1：参考帧锁定的双指预紧不变流形接触场（`BPI-MCF`，32D）

对左右双指 $i \in \{L, R\}$ 的原始 4D 观测 $s_i(t) = [p_{\text{mean},i}(t), p_{\text{max},i}(t), c_{x,i}(t), c_{y,i}(t)]^\top$：

1. **初始接触参考帧锁定（$k_{\text{base}}=5$）与单指预紧力归一化（$f_{i,0}$）**：
   $$\bar{s}_i^{(0)} = \frac{1}{k_{\text{base}}}\sum_{\tau=1}^{k_{\text{base}}} s_i(\tau), \qquad f_{i,0} = \max\left(\bar{p}_{\text{mean},i}^{(0)}, \epsilon\right)$$
   $$\tilde{\Delta c}_{x,i}(t) = \frac{c_{x,i}(t) - \bar{c}_{x,i}^{(0)}}{f_{i,0}}, \qquad \tilde{\Delta c}_{y,i}(t) = \frac{c_{y,i}(t) - \bar{c}_{y,i}^{(0)}}{f_{i,0}}$$
2. **双指对称/反对称正交分解与空间偏心力矩偶极（Dipole）**：
   $$d_{\text{sym}}(t) = \frac{1}{2}\left(\tilde{\Delta s}_L(t) + \tilde{\Delta s}_R(t)\right), \qquad d_{\text{anti}}(t) = \frac{1}{2}\left(\tilde{\Delta s}_L(t) - \tilde{\Delta s}_R(t)\right)$$
   $$\tau_{\text{ecc},x}(t) = \Delta p_L(t)\,\tilde{\Delta c}_{x,L}(t) - \Delta p_R(t)\,\tilde{\Delta c}_{x,R}(t), \qquad \tau_{\text{ecc},y}(t) = \Delta p_L(t)\,\tilde{\Delta c}_{y,L}(t) - \Delta p_R(t)\,\tilde{\Delta c}_{y,R}(t)$$

**8 阶段冻结消融结论**：
- 从 `2_Standard_MCF_32D` 到 `3_Initial_Centered_MCF_32D`（锁定初始接触参考帧 $k_{\text{base}}=5$），开发集准确率从 `46.7%` 跃升至 **`72.5%`**，12 组独立盲测集 `Fresh12` 从 `33.9%` 跃升至 **`49.3%`**（**+15.4%**）。
- 引入双指对称/反对称正交分解与偏心偶极（`5_Bilateral_SymAnti_Dipole_16D`）后，`Fresh12` 进一步跃升至 **`60.0%`**，36 工况 OOD `pilot_tangent_02` 跃升至 **`47.3%`**。
- 完整 32D `BPI-MCF`（`7_Full_Frozen_BPI_MCF_32D`）在开发集达 **`97.5%`**，在 `Fresh12` 全集达 **`65.7%`**（无严重姿态漂移的 `b0+b1` 达 **`68.1%`**）。

### 2.2 创新点 2：跨模态接触力螺旋与剪切方向自监督（防止多模态表征坍缩）

- **全路径梯度核验**：修复 `vis_wrench_head` 的梯度阻断问题，将 `0.35 * loss_vis_wrn` 纳入联合训练损失；单步反向传播自动化审计（`part0_gradient_audit`）确认全模型 26 组权重/偏置张量梯度 100% 严格大于零（`all_positive = True`）。
- **公平 Post-Hoc 冻结隐空间线性探针（`FrozenProbeDirAcc`）**：为消除未训练分类头与已训练分类头的不公平比较，对所有模型（`A1..A3, B1..B5`）训练完成并完全冻结后的触觉隐变量 $z_{\text{tac}}$ 独立拟合相同正则化强度（$\lambda = 1.0$）的 Ridge 线性分类器。在 `pure_loading_01` 上，朴素多模态融合 `B1_Naive` 的 `FrozenProbeDirAcc` 为 `85.8%`，加入物理辅助监督后提升至 **`94.4%`**（`B5_Proposed_Full`），同时将视觉遮挡+接触反转下的动作跟踪误差 `RevNeg(Occ)` 从 `0.9034`（`A0_Vis_Only`）和 `0.3490`（`B1_Naive`）降至 **`0.2202`**。

### 2.3 创新点 3：因果偶极新息能量门控 3-DoF 空间反射控制

- **物理符号对齐与因果门控**：统一离线 Teacher 与 Isaac Lab 6-DoF 阻尼最小二乘雅可比控制器的顺应符号约定（`ux_star = -tanh(0.65*fx + 8.0*my)`，`uy_star = -tanh(0.65*fy - 8.0*mx)`），并利用预紧窗口后的因果偶极新息能量驱动门控 $g_{\text{tac}}(t) \in [0, 1]$，直接调制 3-DoF 输出 $u_{\text{out}}(t) = g_{\text{tac}}(t) \cdot [\Delta x_{\text{wrist}}, \Delta y_{\text{wrist}}, \Delta g_{\text{grip}}]^\top$。
- **Isaac Lab 无混杂 7 控制器闭环验证（`Part A`）**：
  - 完全移除 `Stage 3S` 1D 基线混杂（0% `Stage 3S`）的纯神经策略 `6_v2_1_pure_bpi_3dof_no_stage3s` 取得四向平均滑移 **`0.7795 mm`**（较开环 `8.2011 mm` 降低 **`90.5%`**，较原仓库 `2_stage3s_1d_grip_only` 的 `1.2303 mm` 降低 **`36.6%`**，较 3-DoF 标量触觉 `6.5608 mm` 降低 **`88.1%`**）。
  - 在最困难的 `+Fx` 方向上，滑移从 `Stage 3S` 的 `3.7210 mm` 降至 **`2.1754 mm`**（降低 **`41.5%`**）。
  - 预紧静默阶段腕部抖动从无门控 `B3` 的 `1.4605 mm` 严格压制为 **`0.0000 mm`**。
- **Isaac Lab 3 阶段长时序操作基准（`Part B`）**：
  - 在 `Stage 1 预紧建立 -> Stage 2 视觉遮挡受扰搬运 -> Stage 3 接触切向力反转插孔对齐` 连续 135 步测试中，`6_v2_1_pure_bpi_3dof_no_stage3s` 取得 Stage 1 抖动 **`0.0000 mm`**、Stage 2 搬运滑移 **`0.7510 mm`**、Stage 3 反转对齐滑移 **`0.7186 mm`**、末端最终对齐误差 **`0.9627 mm`**（较 `Stage 3S` 1D 基线的 `1.6716 mm` 降低 **`42.4%`**）。

---

## 3. 下一步执行路线（算力与真机衔接）

1. **Step 1（Isaac Lab 长时序残差强化学习 Residual PPO/SAC）**：冻结 `BPI-MCF`（32D）与 V2.1 `B5_Proposed_Full` 编码器作为 Warm-Start，训练受限残差策略（$\alpha \le 0.35$）并挂载在线物理自监督损失，将 `+Fx` 最差工况的阶段 3 最终尖端误差从 `2.70 mm` 进一步压入 `< 2.0 mm`。
2. **Step 2（2×A100 慢频 VLM 微调 + DGX Spark 128GB 双频推理部署）**：在 2×A100 上微调 `Qwen2.5-VL-7B-Instruct` 输出结构化阶段与阻抗原语（2–5 Hz 慢环），与运行于 100–200 Hz 的 18K 参数 V2.1 3-DoF 触觉反射网络（快环，单步推理 `< 0.15 ms`）组成慢快双频系统。
3. **Step 3（真机 Sim-to-Real 轻量标定与闭环验证）**：利用 `BPI-MCF` 的初始参考帧锁定（$k_{\text{base}}=5$）与单指预紧归一化（$f_{i,0}$），在真实双指触觉机械臂上执行空间极性对齐、预紧窗口设定与少量样本末端微调，完成真实物理偏心扰动与插孔对齐实验。
