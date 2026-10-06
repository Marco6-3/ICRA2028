# 候选主线：Physics-Guided Contact-State Compression

状态：候选研究方向，尚未形成已验证贡献。更新：2026-10-06。

## 核心问题

> **What is the minimal tactile state that preserves control-relevant spatial contact information over long horizons?**

更具体地说：在固定感知、计算和闭环延迟预算下，能否把高维触觉阵列压缩成低维、可解释的 contact state，使它既保留快速接触控制所需的信息，又适合长时间因果记忆？

这条路线不是“给 TacMamba 换一个传感器”，也不是预先决定必须使用 Mamba。它先检验表示瓶颈本身，再决定是否需要状态空间模型。

## 与已有工作的关系

- **TacMamba**：证明了高频低维力信号可以通过递推状态维持长历史与低单步延迟，但其真实系统使用单点 1D force sensing，缺少丰富空间接触拓扑。
- **T-Rex**：把时间力动态与空间 deformation 分开编码，说明 temporal / spatial tactile information 可以分工处理。
- **TacForcing**：强调触觉信息在执行时刻进入动作生成，提醒本项目最终仍需验证端到端闭环价值，而不能停留在离线表示指标。
- **FlexiTac / VTAP**：提供柔性高密度法向接触阵列与指尖部署参考，适合把“高维空间接触场 → 低维接触状态”作为一个干净的研究对象。

这些工作提供方法来源和比较对象；本项目不把已有模块直接拼接后称为贡献。

## 候选硬件：FlexiTac

优先把 FlexiTac 作为第一阶段 testbed，而不是宣称它优于所有触觉传感器。

当前公开 FlexiTac V2 工作展示了约 12×32 taxels、2 mm pitch、100 Hz 串口流的指尖阵列；不同板型和固件配置需在真实实验前重新固定版本并实测。压阻阵列原始输出不是天然的 Pa / N，需要 baseline 与标定，因此本项目首先把每个 taxel 的处理后响应记为 contact intensity / force proxy，而不是未经验证的绝对压力。

FlexiTac 的主要优势是二维法向接触分布清晰；主要边界是缺少直接的切向剪切观测，因此不能把 CoP 漂移直接等同于 slip。

参考：
- FlexiTac: https://arxiv.org/abs/2604.28156
- VTAP / FlexiTac fingertip deployment: https://yuhao-zhou.com/vtap/index.html
- TacMamba: https://arxiv.org/abs/2603.01700
- T-Rex: https://arxiv.org/abs/2606.17055
- TacForcing: https://arxiv.org/abs/2608.25798

## 解析 contact-state bottleneck

原始触觉帧：

$$
X_t \in \mathbb{R}^{H\times W}.
$$

经过 baseline / calibration 后得到非负或有统一语义的 taxel 权重 $w_i$。第一版候选状态为：

$$
s_t=
[
P'_t,
c_x,c_y,
A'_t,
\sigma_1,\sigma_2,
o_x,o_y,
\Delta P'_t,
\Delta c_x,\Delta c_y
]
\in \mathbb{R}^{11}.
$$

其中：

- $P_t=\sum_i w_i$：总接触强度 / force proxy；
- $(c_x,c_y)$：pressure-weighted / intensity-weighted contact centroid；
- $A'_t=N_{active}/N_{taxel}$：归一化有效接触面积；
- $(\sigma_1,\sigma_2)$：归一化二阶矩主尺度；
- $(o_x,o_y)$：带各向异性置信度的主方向编码；
- $\Delta P'_t,\Delta c_x,\Delta c_y$：接触动态变化。

这组量称为 **physically interpretable contact descriptors**，不称为“完整空间信息”或“物理不变量”。

## 底层预处理必须固定的规则

### 1. 空间无量纲化

直接用传感器归一化坐标：

$$
\tilde x_i=2\frac{i}{N_x-1}-1,\qquad
\tilde y_j=2\frac{j}{N_y-1}-1.
$$

由这些坐标计算 CoP 与二阶矩，使空间 descriptor 对阵列物理尺寸和 taxel 数量更稳定。

$$
\tilde\Sigma_t=
\frac{
\sum_i w_i
(\tilde{\mathbf r}_i-\tilde{\mathbf c})
(\tilde{\mathbf r}_i-\tilde{\mathbf c})^T
}{
\sum_i w_i+\epsilon
}.
$$

### 2. 总压力尺度不要默认在线滑动 Z-score

在线滑动 Z-score 可能把长期稳定抓持重新中心化到 0，破坏本来希望保留的长期载荷状态。默认采用：

$$
P'_t=
\log\left(
1+
\frac{\max(P_t-P_{base},0)}{P_{ref}}
\right).
$$

- $P_{base}$：只在 no-contact 状态更新，接触时冻结；
- $P_{ref}$：由训练 / 标定数据固定，测试时不自适应；
- 任何归一化参数不得从测试 episode 的未来数据估计。

### 3. 主方向避免直接求 \(\theta\)

令：

$$
\Sigma=
\begin{bmatrix}
a & b\\
b & d
\end{bmatrix},
\qquad
D=\sqrt{(a-d)^2+4b^2}.
$$

定义各向异性置信度：

$$
q_{ani}=
\frac{D}{a+d+\epsilon}.
$$

方向编码：

$$
o_x=q_{ani}\frac{a-d}{D+\epsilon},
\qquad
o_y=q_{ani}\frac{2b}{D+\epsilon}.
$$

当接触近似各向同性时，$(o_x,o_y)\to(0,0)$，避免随机主轴方向污染模型。

### 4. 差分特征保留方向且需滤波

优先保留 $\Delta c_x,\Delta c_y$，而不是只保留 $\|\Delta\mathbf c\|$。差分前需要固定因果滤波或短窗口 slope estimator；必须测量滤波造成的额外事件检测延迟。

CoP 漂移只称为 **contact redistribution cue**，不能直接宣称为 slip。

## Stage 0：Zero-Network Physics Sanity Check

这一阶段不使用 Mamba、VLA 或复杂神经网络。

采集重复的：

$$
\text{no-contact}
\rightarrow
\text{contact}
\rightarrow
\text{load}
\rightarrow
\text{redistribution / roll}
\rightarrow
\text{release}.
$$

检查：

- descriptor noise 与 drift；
- episode / repeated-trial repeatability；
- 接触开始与释放的事件响应；
- 持续载荷是否被 baseline tracking 错误抹掉；
- $(P,\Delta P)$、$(c_x,\Delta c_x)$、$(A,\sigma_1/\sigma_2)$ 等 phase-plane trajectory 是否具有可重复结构；
- raw acquisition、descriptor computation 与 filtering 的真实 P50 / P95 latency。

**继续条件**：descriptor 在重复实验中数值稳定、事件响应具有可重复性，且不存在明显由 baseline / threshold / filtering 人工制造的结构。漂亮的 phase plot 只生成 hypothesis，不证明论文主张。

项目级预注册门槛：task-relevant descriptor repeated-trial ICC 目标 ≥ 0.90；固定稳定载荷下 total-load proxy CV 目标 ≤ 5%；preprocessing + descriptor compute P95 ≤ 一个 tactile frame interval 的 20%，event-to-descriptor P95 ≤ 一个 tactile frame interval。硬件/固件/记录格式冻结后，以 4 周主动实验或 3 个独立采集批次（先到者）为一次 timebox；若仍失败则 pivot，而不是无限调阈值。

## Stage 1：Representation Bottleneck Experiment

比较三种表示在相同数据划分下保留和丢弃的信息：

1. **Raw**：原始 $H\times W$ tactile map；
2. **Physics**：11D $s_t^{phys}$；
3. **Learned**：从 raw 学得、维度与 Physics 匹配的 11D latent。

最小 baseline：

| 输入 | 模型 | 目的 |
| --- | --- | --- |
| Physics 11D | Linear / tiny MLP | 物理 bottleneck 的低样本可分性 |
| Raw | Linear | 检查简单读出能否直接利用高维输入 |
| Raw | Tiny CNN | 高维 learned baseline |
| Learned 11D | Tiny encoder + linear head | 与 Physics 同维度比较 inductive bias |

主要关注：

- **data efficiency**：1%、5%、10%、25%、50%、100% training data；
- **OOD**：新物体、新接触位置、新刚度 / 表面条件；
- **latency / compute**：descriptor、encoder 与 classifier 的实测成本；
- **failure boundary**：哪些局部 tactile patterns 被 Physics bottleneck 丢失。

Physics representation 的目标不是“信息比 Raw 更多”，而是验证它是否提供更好的 **inductive bias / compactness / OOD robustness / latency trade-off**。

## 对抗性判决沙盒

### A. Moment-matched but locally different

不要人工强行同时调平所有 $P,CoP,A,\Sigma$。先采集不同接触 pattern，再跨类别寻找：

$$
(i,j)^*=
\arg\min_{y_i\neq y_j}
\|s_i-s_j\|_2
$$

同时要求 raw map 距离较大。这样寻找真实的“physics descriptor 几乎相同，但局部接触图明显不同”的样本对，直接量化 many-to-one compression 丢失的信息。

### B. Different contact dynamics

建议至少包含两类动力学：

- bistable / click-like event：$\Delta P$ 大而 CoP 变化小；
- rolling / tilting / redistribution：总压力变化较缓而 CoP / 主轴持续变化。

先观察 compact state 的轨迹，再做事件识别。不要预设 Physics 一定击败 Raw；要检验在少数据、OOD 和计算约束下是否更高效。

## Stage 2P：低成本 temporal interaction probe

Issue #1 指出了严格串行门控可能漏掉 **representation × memory** 交互。因此即使 Stage 1 尚未正式通过，也允许一个小型、预注册的 probe：

- Scalar → matched streaming memory；
- Scalar + centroid → matched streaming memory；
- Physics11D → matched streaming memory；
- Learned11D → matched streaming memory。

四者必须共享 temporal backbone、hidden/parameter budget、history visibility、训练数据和 decision rate。该 probe 只回答“长历史是否暴露单帧读出没有显示的增量”，不接 VLA，不等于认可完整 11D。

若 Physics11D 不能超过 Scalar+centroid，则优先缩减 descriptor；若 Learned11D 更好，则考虑 learned/hybrid bottleneck；若 spatial variants 均无增量，则不继续用更大 Mamba 掩盖表示问题。

## Stage 2：Formal Temporal Memory

如果 compact contact state 在 Stage 1 中表现出足够信息保真和明确边界，或 Stage-2P 给出预注册且可重复的 interaction evidence，再扩大研究：

$$
s_{1:t}^{phys}
\rightarrow
h_t.
$$

TacMamba / Mamba 是此时的强候选，而不是预先锁死的答案。需要比较：

- scalar / total-force history；
- physics-state history；
- learned low-dimensional history；
- 必要时 raw temporal encoder。

真正要回答的是：**physically structured bottleneck 是否是长时、高频 tactile memory 的更好接口？**

若纯 physics bottleneck 在某类任务上明确缺少局部空间细节，可进一步评估低频或 event-triggered rich tactile token 作为补充；这属于后续 hybrid extension，不是 Stage 0/1 的默认组件。

## 停止与降级条件

出现以下情况之一，应停止扩大该方向并重新判断：

- descriptor 主要由阈值 / baseline / calibration 人工产生，不具重复性；
- 11D Physics 在少数据、OOD、延迟等维度均没有相对 learned compact latent 的优势；
- Physics 丢失的局部信息恰好是主要闭环任务必需信息，且需要高频 raw processing 才能恢复；
- 后续 temporal model 的收益仅来自更多参数或更长输入，而与 physics bottleneck 无关；
- Stage-2P 中 Physics11D 与 Scalar+centroid 长历史表现近似，则停止维护“完整 11D 必要”的 claim，优先采用更小状态。

本文件记录的是**待验证假设与实验协议**，不能将任何候选 descriptor、延迟、泛化或控制收益写成已证明结果。
