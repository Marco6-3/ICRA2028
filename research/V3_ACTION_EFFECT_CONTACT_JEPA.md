# V3 算法蓝图：上下文分解的动作效应接触表征学习（CAEC-JEPA）

> **状态（2026-11-20 研究转型立项）**：候选算法设计与验证规划（**未运行实验，所有主张均为待检验假设**）。  
> **双线研究定位**：
> - **Paper A（`NewGripper` 仓库）**：YUBI 无本体采集 + 两端物理共用实体声-振-触觉快换盒 + 光学全盲脆性薄层操作基准（硬件与数据采集系统主线）；
> - **Paper B（本仓库 V3 转型主线）**：面向多频异步、路径依赖、跨工具/跨材质接触传感的**通用预测式压缩表征学习算法（CAEC-JEPA）**，早期在公开接触操作数据集（优先 `ManiWAV`、`RDP / UMI-FT / FreeTacMan`）上完成算法验证与消融，待 `NewGripper` 硬件采数就绪后零改动接入真机闭环。  
> - **历史资产约束**：遵循 `AGENTS.md`，旧版 Stage 1–4（`MCF / BPI-MCF / CMQ-BPI-MCF V2.2`）代码、数据和审计记录原样封存，不改写；后续不再采购 FlexiTac。

---

## 1. 核心科学问题与差异化边界

### 1.1 唯一核心科学问题

> **在多频异步（如 20 kHz 接触声振 / 1 kHz 应变 / 500 Hz 力觉 vs. 10–30 Hz 下游策略）、存在漂移与传递函数变化的接触丰富操作中，当瞬时观测相似而历史交互路径或工具/材质属性不同时，能否在无需六维力真值、无需帧级人工标注、无需重构原始高频波形/像素的条件下，通过上下文分解的连续动作净效应隐空间预测学出紧凑因果表征，显著提升下游少样本模仿学习与 OOD 泛化，并同步提供毫秒级接触相变告警？**

### 1.2 与 2025–2026 既有工作的明确边界（防撞车）

| 相关工作范式 | 代表工作 | 它们的边界 | V3 (`CAEC-JEPA`) 的差异化突破 |
| :--- | :--- | :--- | :--- |
| **静态/空间触觉 JEPA** | `Tactile-JEPA`、`LingBot-Touch (2026.11)` | 把触觉阵列当作 2D 空间图像做 taxel 掩码预测，面向同频直接触觉阵列 | 面向**非马尔可夫时序动力学**与**多频异步（20 kHz 声振/应变 + 30 Hz 视觉）**，不依赖 2D 电子皮肤网格 |
| **端到端声振/力觉模仿学习** | `ManiWAV (CoRL 2024)`、`RDP (RSS 2025)`、`UMI-FT (ICRA 2026)` | 直接端到端输入原始频谱或视触觉，易受视觉捷径、换工具频响漂移与低频下采样抹平瞬态影响 | 预训练冻结/可微调的**3-Token 压缩瓶颈**，把工具/材质慢上下文、因果状态与毫秒级惊奇事件显式解耦后再交给下游策略 |
| **经典离散 PSR / R-Learner** | `Predictive State Representations`、离散 $\{+,-,0\}$ 固定传感统计量 $Y$ 响应差 | 需随机分配离散微动作且依赖手工挑选低维目标 $Y$（如 CoP/面积），难以直接用于 6-DOF 连续自然示教与 20 kHz 声振 | 保留“**剥离零动作基线、专学动作净效应**”的核心数学思想，推广至**连续 6-DOF 动作 + 隐空间多步预测**，适配自然人类示教与任意传感模态 |

---

## 2. 数学建模与网络架构

### 2.1 统一高频接触时钟（Unified Contact Clock）

不同传感器采样率跨度达 3 个数量级（30 Hz 视觉/视触觉 ~ 20–48 kHz 接触声振）。为既不浪费算力在 20 kHz 逐点自回归，又不因 30 Hz 下采样丢失 5 ms 级物理事件，算法建立统一的**高频接触时钟**（步长 $\Delta t_{\text{fast}}$，真机 `NewGripper` 设为 **5 ms / 200 Hz**；在 10–50 Hz 公开数据集上按实际动作帧率与音频帧率对齐为整数倍子步 $K \ge 1$）：

- **声振通道 $o^{\text{aud}}_\tau$**：因果短时窗 log-Mel / STFT（如窗长 20 ms、步长 5 ms、64 频带）$\xrightarrow{\text{Causal 1D-CNN}} e^{\text{aud}}_\tau \in \mathbb{R}^{64}$；
- **应变 / 力觉通道 $o^{\text{ft}}_\tau$**：每 $\Delta t_{\text{fast}}$ 内分段时域特征 / 1D-Conv $\to e^{\text{ft}}_\tau \in \mathbb{R}^{32}$；
- **低频视触觉 / 接触图像通道 $o^{\text{tac}}_\tau$**（可选）：异步更新，附带可用性掩码 $m^{\text{mod}}_\tau \in \{0,1\}$；
- **已执行本体状态增量 $\Delta q_\tau$**（末端相对位姿/速度增量、夹爪开度增量）：$\to e^{\text{prop}}_\tau \in \mathbb{R}^{32}$。

单步融合特征 $u_\tau = \operatorname{Fuse}(\{m^{\text{mod}}_\tau e^{\text{mod}}_\tau\}, e^{\text{prop}}_\tau) \in \mathbb{R}^{128}$ 输入**严格因果编码器 $E_\theta$**（2 层 GRU，隐维 $d_z = 64$，可替换因果 Mamba 作消融），输出瞬时路径依赖隐状态：

$$z_\tau = E_\theta(u_{\le \tau}) \in \mathbb{R}^{d_z}.$$

> **因果硬约束**：所有归一化、滤波、STFT 窗口与 RNN 更新严禁偷看未来帧，必须通过前缀因果单元测试（$\max \|z_{0:\tau}^{(T_1)} - z_{0:\tau}^{(T_2)}\|_\infty = 0$）。

---

### 2.2 慢上下文编码器（隐式工具传递函数与材质辨识）

针对换工具（$Y(\omega) = H_{\text{tool}}(\omega) X_{\text{tip}}(\omega) + N(\omega)$）与换物体（软膜 / 钢化膜 / 硅片 / 不同表面摩擦）导致的分布漂移，引入**慢上下文编码器 $E_{\text{ctx}}$**，聚合过去 $W_{\text{ctx}}$ 步（约 0.5–1.0 s）的因果交互历史 $(u_{\tau - W_{\text{ctx}}:\tau}, a_{\tau - W_{\text{ctx}}:\tau})$：

$$c_\tau = E_{\text{ctx}}\!\left(u_{\tau - W_{\text{ctx}}:\tau},\, a_{\tau - W_{\text{ctx}}:\tau}\right) \in \mathbb{R}^{d_c}, \quad d_c = 32.$$

为防止 $c_\tau$ 退化为瞬态状态拷贝，对其施加**段内时间慢变一致性约束**（见第 2.4 节），迫使其只吸收在几百毫秒至单次轨迹内缓变或不变的物理隐变量（工具固有频响、物体刚度/摩擦/粘附特性）。

---

### 2.3 分解式动作净效应隐空间预测器（Core JEPA Predictor）

维护目标编码器 $E_{\bar{\theta}}$（$E_\theta$ 的指数移动平均 EMA，$\bar{\theta} \leftarrow \gamma \bar{\theta} + (1-\gamma)\theta$，$\gamma \in [0.99, 0.996]$），提供停止梯度的未来目标隐状态：

$$\bar{z}_{\tau+\Delta} = \operatorname{sg}\!\left(E_{\bar{\theta}}(u_{\le \tau+\Delta})\right), \quad \Delta \in \mathcal{D}.$$

其中 $\mathcal{D}$ 为多时间尺度预测步长集合（例如 200 Hz 下对应 $\{10\text{ ms}, 40\text{ ms}, 120\text{ ms}\}$；30 Hz 数据集下对应 $\{1, 3, 8\}$ 帧）。

核心设计是**将预测器显式分解为零动作基线演化与连续动作净效应**：

$$\hat{z}_{\tau+\Delta} = z_\tau + \underbrace{b_\phi(z_\tau, c_\tau, \Delta)}_{\text{零动作自然演化（吸收蠕变/漂移）}} + \underbrace{g_\phi(z_\tau, c_\tau, a_{\tau:\tau+\Delta}, \Delta)}_{\text{连续动作净效应}}$$

并在结构上**严格强制零动作恒等为零**：

$$g_\phi(z_\tau, c_\tau, a, \Delta) = \tilde{g}_\phi(z_\tau, c_\tau, a, \Delta) - \tilde{g}_\phi(z_\tau, c_\tau, \mathbf{0}, \Delta) \implies g_\phi(z_\tau, c_\tau, \mathbf{0}, \Delta) \equiv \mathbf{0}.$$

**物理与数学含义**：
1. 当机器人保持静止（$a_{\tau:\tau+\Delta} = \mathbf{0}$）时，预测退化为 $z_\tau + b_\phi(z_\tau, c_\tau, \Delta)$，传感器粘弹性松弛、缓慢漂移和自然衰减被 $b_\phi$ 吸收；
2. 当机器人执行连续 6-DOF 动作 $a_{\tau:\tau+\Delta}$ 时，$g_\phi$ 专职刻画“**在当前历史状态 $z_\tau$ 与工具/材质上下文 $c_\tau$ 下，动作 $a$ 相对于静止所引起的净接触变化**”——这是将因果推断 R-Learner 的“动作响应差”推广至连续动作与高维多模态隐空间的直接形式；
3. 当发生**接触反转或临近滑移边界**时，即使瞬时读数相似，只要 $z_\tau$ 保留了加载历史，预测器就能对相同动作 $a$ 给出截然不同的净响应 $g_\phi$。

---

### 2.4 无标签防坍缩训练目标（含自然示教解混杂机制）

在自然人类示教（`ManiWAV` / `FreeTacMan` / `YUBI`）中，动作 $a$ 并非完全随机分配，因此 $a$ 与当前状态 $z_\tau$ 存在相关性（混杂）。仅用前向预测损失容易出现两种退化：① $b_\phi$ 偷吃动作效应、$g_\phi \to 0$；② 静态帧上 $z_\tau$ 学出恒等或常数坍缩。为此总损失由 5 项职责正交的无标签项组成：

1. **多步隐空间预测损失（Forward JEPA Loss）**：
   $$\mathcal{L}_{\text{pred}} = \frac{1}{|\mathcal{D}|} \sum_{\Delta \in \mathcal{D}} w_\tau \cdot \operatorname{SmoothL1}\!\left(\hat{z}_{\tau+\Delta},\, \bar{z}_{\tau+\Delta}\right)$$
   其中 $w_\tau \ge 1$ 为**接触活跃加权**（按窗口内传感能量变化率与动作变化率加权，避免 80% 空载/平稳帧主导梯度）。

2. **零动作基线解混杂锚定（Hold-Anchor Regularization）**：
   对于自然示教中存在的静止/低速/预紧保持样本（$\|a_{\tau:\tau+\Delta}\|_2 < \epsilon_{\text{hold}}$），由于 $g_\phi \approx 0$，样本直接锚定 $b_\phi$；同时对活跃动作样本施加**基线停止梯度交替更新或动作反事实敏感性监控**（检查 $\|g_\phi(z_\tau, c_\tau, a)\|_2 / \|b_\phi(z_\tau, c_\tau)\|_2$，确保动作分支不被旁路）。

3. **轻量逆动力学辅助损失（Inverse Dynamics, IDM）**：
   $$\mathcal{L}_{\text{idm}} = \frac{1}{|\mathcal{D}|} \sum_{\Delta \in \mathcal{D}} \left\| \operatorname{IDM}_\omega\!\left(z_\tau,\, \bar{z}_{\tau+\Delta},\, c_\tau,\, \Delta\right) - \bar{a}_{\tau:\tau+\Delta} \right\|_2^2$$
   其中 $\bar{a}_{\tau:\tau+\Delta}$ 为步长 $\Delta$ 内的累计动作/位移增量。IDM 迫使 $z_\tau$ 保留动作-接触因果敏感信息，过滤与交互无关的环境底噪。
   > *注（重要审慎原则）*：为避免 IDM 仅通过本体输入 $e^{\text{prop}}_\tau$ 走捷径猜出动作，IDM 消融与诊断必须检验**纯接触支路 $z^{\text{contact}}_\tau$** 是否真正贡献了动作/状态可辨识性。

4. **免标签防坍缩正则（VICReg / SIGReg 方差与去相关项）**：
   $$\mathcal{L}_{\text{reg}} = \lambda_v \frac{1}{d_z}\sum_{j=1}^{d_z} \max\!\left(0,\, \gamma_v - \sqrt{\operatorname{Var}(z_{:,j}) + \varepsilon}\right) + \lambda_c \frac{1}{d_z}\sum_{i \neq j} \left(\operatorname{Cov}(z)_{ij}\right)^2$$
   直接约束 batch 内隐变量维持满秩与非冗余（不依赖任何六维力或人工类别标签）。

5. **慢上下文一致性损失（Slow Context Invariance）**：
   $$\mathcal{L}_{\text{slow}} = \left\| \operatorname{norm}(c_\tau) - \operatorname{sg}\!\left(\operatorname{norm}(c_{\tau - \delta})\right) \right\|_2^2$$
   对同一轨迹相邻时间块（如间隔 $\delta = 0.5\text{ s}$）约束 $c$ 缓变。

总目标：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{idm}} \mathcal{L}_{\text{idm}} + \mathcal{L}_{\text{reg}} + \lambda_{\text{slow}} \mathcal{L}_{\text{slow}}.$$

---

### 2.5 预测惊奇度与下游 3-Token 压缩接口

#### (A) 条件化隐空间惊奇度（Surprise / Regime-Shift Score）
在每一高频时钟步 $\tau$，计算一步（或短步 $\Delta_1$）预测残差范数：

$$r_\tau = \left\| E_{\bar{\theta}}(u_{\le \tau}) - \hat{z}_\tau^{(\Delta_1)} \right\|_2.$$

因为高速运动本身残差尺度大于低速运动，使用训练集正常交互段按速度/动作幅值分箱（或轻量条件方差头 $\hat{\sigma}_\phi(z_{\tau-\Delta_1}, c_{\tau-\Delta_1}, a)$）做标准化，得到无量纲惊奇度分数：

$$s_\tau = \frac{r_\tau - \mu_r(\|a\|)}{\sigma_r(\|a\|) + \varepsilon}.$$

- 正常平稳接触/滑移时，$s_\tau$ 接近 0；
- 遇到**外源扰动、碰触微缝边缘、顶死脆边、剪切反转突跳、微裂纹前兆或卡扣 Snap-Through** 时，$s_\tau$ 剧烈抬升，可直接作为 `NewGripper` 200–500 Hz 毫秒级安全反射的零标注核心触发特征。

#### (B) 面向 10–30 Hz 下游策略的惊奇度门控 3-Token 接口
下游策略（Diffusion Policy / ACT / VLA）在时间步 $t$（低频，间隔 $K$ 个高频步）读取接触表征时，直接喂入固定预算的 **3 个紧凑接触 Token**（可拼接为 160D 向量或作为 3 个 Cross-Attention Token）：

1. **`[CTX]`** $= c_t \in \mathbb{R}^{32}$：工具与材质慢上下文；
2. **`[STATE]`** $= z_t \in \mathbb{R}^{64}$：最新因果路径依赖接触状态；
3. **`[EVENT]`** $= \sum_{\tau = t-K+1}^{t} \alpha_\tau z_\tau \in \mathbb{R}^{64}$，其中 $\alpha_\tau = \operatorname{softmax}(s_\tau / T_{\text{temp}})$：按隐空间惊奇度加权池化的高频事件摘要，**确保控制周期内发生的 5 ms 瞬态事件不被低频下采样平均抹平**。

---

## 3. 分阶段验证路线与实验矩阵（公开数据集 $\to$ `NewGripper` 真机）

所有实验遵循**先证伪后扩张、同数据/同参数/同下游公平对比、独立外部指标验收**纪律。

### Phase 0（当前立即执行）：远程计算节点 `nvidia-desktop` 公开数据集 MVP 验证
- **首选公开数据集**：
  1. **`ManiWAV`（CoRL 2024，`real-stanford/maniwav`）**：手持 UMI + 接触式麦克风声振 + 腕部相机 + 6-DOF 连续动作，模态与 `NewGripper` 的 YUBI+声振听诊最为同构；
  2. **备选/第二阶段公开数据集**：`Reactive Diffusion Policy (RDP)` / `UMI-FT` / `FreeTacMan`（覆盖力觉 + 视触觉 + 多任务接触操作）。
- **MVP 首轮必须回答的 4 个可否证诊断（E0）**：
  1. **工程与数学正确性审计**：前缀因果误差 $= 0$；零动作恒等 $g_\phi(\cdot, \mathbf{0}) \equiv 0$；单步反传全参数梯度有效；隐空间无坍缩（有效秩 Effective Rank、维度方差均值、协方差非对角能量）；
  2. **历史与动作净效应有效性（借用 Codex 核心诊断）**：在匹配低差异瞬时观测但前后运动/接触阶段不同的切片上，检验 `因果 GRU > 固定短窗 > 单帧`，以及 `b+g 分解 > 普通前向预测`；
  3. **惊奇度 $s_\tau$ 对真实接触突变事件的有效性**：在接触建立/分离、刮擦突变、方向反转等物理事件上，对比 `惊奇度 s_t` 与 `原始音频能量/频谱变化率` 的检出 AUROC 与固定误报率下检出延迟；
  4. **下游冻结表征少样本模仿学习增益（核心试金石）**：固定同一个小型下游策略（如基于视觉 + 接触 Token 的 Diffusion Policy / MLP-Chunk BC），在 `10% / 30% / 100%` 轨迹预算及留出测试轨迹（按 episode / 物体 / 背景噪声划分）上对比以下 5 个公平基线：

| 对比编号 | 表征方法 | 核心检验问题 |
| :--- | :--- | :--- |
| **B0** | Vision + Proprio only（无声振/触觉） | 确认任务确实依赖接触感知 |
| **B1** | End-to-End Raw Audio/Tactile Encoder（如 ManiWAV 原版） | 检验预训练压缩相比端到端训练的少样本与 OOD 优势 |
| **B2** | Masked Reconstruction Pretraining（MAE 波形/频谱重构） | 检验“隐空间动力学预测（免重构）”是否优于“信号重构” |
| **B3** | Vanilla Forward JEPA（无 $c_t$、无 $b+g$ 分解、无 IDM、无 `[EVENT]` 池化） | 检验我们的三大结构设计相对朴素 JEPA 的真实增量 |
| **Ours** | **`CAEC-JEPA`（`[CTX] + [STATE] + [EVENT]` 3-Token）** | 完整方法（并进一步逐项消融 `w/o CTX`、`w/o b+g`、`w/o IDM`、`w/o EVENT pooling`） |

### Phase 1（与 `NewGripper` 硬件联动）
- 先用 `NewGripper` Phase 0 的**百元手工压电台架数据** + **MuJoCo 剥离/卡扣机理仿真数据**，验证跨撬片厚度 $c_t$ 聚类分离度与顶死/进缝/脆断前兆的惊奇度尖峰提前量；
- 待 `NewGripper` YUBI 样机就绪后，直接接入 **20 kHz 声振 + 1 kHz 应变 + 500 Hz 六维力 + 30 Hz GelSight** 四模态同步数据流，执行最终真机闭环少样本学习曲线与跨工具/跨材质 OOD 评测。

---

## 4. Go / No-Go 红线（防自我欺骗纪律）

1. **红线 1（防坍缩与本体捷径）**：若去掉视觉与本体后，纯接触支路特征的冻结探针与随机基线无显著差异，说明编码器在走本体/视觉捷径，必须停止扩模型并重查损失与 IDM 支路隔离；
2. **红线 2（防“复杂不如简单”）**：若在相同数据与参数预算下，`b+g` 分解或 `c_t` 慢上下文在跨轨迹/跨工具测试上不优于 `B3 (Vanilla JEPA)`，或 `Ours` 在少样本下游任务上不优于 `B1 (Raw)` / `B2 (MAE)`，必须如实记录负结果并精简或重构算法，绝不靠调单一种子掩盖；
3. **红线 3（防伪事件检出）**：惊奇度 $s_\tau$ 必须在**控制动作速度/幅值后**仍显著区分真实接触突变与正常空载运动；若只跟着机械臂加速度响，则属于自生运动混叠失败；
4. **红线 4（统计严谨性）**：最终论文对比必须按独立 episode / object / session 划分训练测试集、报告多种子均值与置信区间，严禁把同一轨迹相邻帧混入训练集和测试集。
