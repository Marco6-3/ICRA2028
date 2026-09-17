# Physical Intelligence Roadmap：从触觉表征到高频物理反应

更新：2026-09-17。状态：长期研究路线草案，不代表当前 IROS 2027 主线已经冻结。

## 1. 核心判断

近期通用视觉-语言-动作模型在粗粒度操作、开放词汇理解、目标识别、空间推理和长时程任务分解上的能力快速增强。一个值得持续检验的工作假设是：

- 高层的“做什么、去哪里做”越来越可能由通用模型统一处理；
- 真正接触后的“发生了什么、是否将滑移、力该如何调整、毫秒到几十毫秒内怎样反应”仍然需要面向物理交互的专门感知、状态估计与控制；
- 因此值得研究通用策略与高频 physical policy 之间的接口，而不是预设一个巨大 Transformer 必须端到端承担所有控制频率。

这只是研究假设，不把“通用模型一定替代传统感知/规划”或“LLM 一定无法完成低层控制”写成既定事实。后续需要用明确任务、频率、延迟与消融实验验证边界。

## 2. 一个可能的多时间尺度架构

可以把机器人控制链暂时抽象为三层：

### A. General / semantic policy（约 5–20 Hz，具体频率待实测）

输入可能包括 RGB、语言指令、低频本体状态和任务历史。

主要承担：
- 开放世界语义理解；
- 目标/物体关系理解；
- 粗粒度空间推理；
- 长时程任务分解；
- skill 或末端目标生成。

输出不必直接是电机控制量，可以是 skill token、目标位姿、子任务或低维 latent command。

### B. Manipulation / physical policy（约 20–200 Hz，具体频率取决于任务与硬件）

输入更偏向局部、近期和物理相关状态，例如：

\[
z_t = f(I_{t-k:t}, q_{t-k:t}, \tau_{t-k:t}, a_{t-k:t-1})
\]

其中 \(I\) 为视觉，\(q\) 为本体状态，\(\tau\) 为触觉/力觉。

主要承担：
- 接触阶段识别；
- 局部轨迹修正；
- slip / shear / contact change 估计；
- 物体柔顺性、摩擦等隐含物理属性的在线估计；
- 根据物理状态对高层动作做 residual correction。

一种简单形式：

\[
a_t = a_t^{global} + \Delta a_t^{physical}
\]

### C. Low-level controller（数百 Hz 到 kHz 级，按真实系统实测）

负责：
- 关节位置/速度/力矩闭环；
- impedance / admittance；
- 电机与驱动器约束；
- 安全限制。

重点不是追求所有层频率一致，而是研究哪些信息真正需要在哪个时间尺度上闭环。

## 3. 为什么“只加速 Transformer”可能不够

提高推理速度当然可能提高闭环频率，但完整反应延迟应拆成：

\[
\tau_{total}=\tau_{sensor}+\tau_{encode}+\tau_{policy}+\tau_{comm}+\tau_{actuator}
\]

因此研究时应分别测量：
- 传感器采样率；
- 数据时间戳与传输延迟；
- encoder 延迟；
- policy inference latency；
- command publish rate；
- controller update rate；
- actuator 实际响应时间。

如果 RGB 只有 30 Hz，即使 policy 运行 200 Hz，也不意味着每次都获得新的视觉信息。相反，触觉、力/力矩、本体状态可能具有更高有效带宽，适合作为高速闭环输入。

因此问题不应只写成“怎样把大模型跑到 1 kHz”，而应写成：

> 哪些状态必须高频更新？哪些信息可以低频更新？不同时间尺度的状态与动作怎样共享，才能在不浪费计算的情况下提高闭环可靠性？

## 4. 本科前期可切入：触觉表征与真实模态依赖

近期更现实的切入口仍然是触觉，而不是直接做完整高频控制系统。

### 问题 A：触觉到底提供了哪些视觉不可替代的信息？

候选物理量包括：
- contact onset / release；
- normal force；
- shear；
- slip / incipient slip；
- contact patch change；
- vibration / transient event；
- task phase；
- compliance / hardness 的可辨识线索。

实验应比较：

\[
(I,q) \rightarrow a
\]

与

\[
(I,q,\tau) \rightarrow a
\]

但仅比较成功率不够，还需要做触觉遮挡、错序、时间偏移、跨轨迹替换等干预，以判断策略是否真正依赖触觉中的当前物理信息。

### 问题 B：是否能把历史压缩成“足够”的物理状态？

希望学习：

\[
z_t=f(I_{\le t},q_{\le t},\tau_{\le t},a_{<t})
\]

并让 \(z_t\) 对未来和控制尽量充分。目标不是让 attention 图“看起来合理”，而是验证：给定 \(z_t\) 后，更早历史是否还显著改善未来预测或动作质量。

这可以把“Transformer 选择重要历史信息”转化为更严格的问题：

> 能否学习一个紧凑但足够的 latent physical state，使局部策略近似 Markov 化？

## 5. 后续方向：高频 tactile / physical reflex policy

当触觉表征和状态估计可靠后，可以进一步研究快慢策略分层。

示意：

\[
\pi_G(I,q,g) \rightarrow a_t^{global}
\]

低频运行；

\[
\pi_L(\tau_{t-k:t}, q_{t-k:t}, z_t) \rightarrow \Delta a_t
\]

高频运行；最终：

\[
a_t=a_t^{global}+\Delta a_t
\]

需要回答：
- 高频局部策略相对单一 policy 是否真的降低失败率？
- 收益来自更高频率、触觉模态，还是模型容量变化？
- 多快才足够？从 20/50/100/200 Hz 提升时收益是否饱和？
- 延迟与采样抖动是否比 nominal Hz 更重要？
- 快策略应该输出末端 residual、gripper force、joint target，还是更低层变量？
- 当局部 reflex 与高层策略冲突时，如何仲裁？

## 6. 机械/力学先验可以如何进入学习策略

机械背景的价值不只是做结构，也可以用于构造可解释的物理状态和约束。

例如简化的 Coulomb friction 条件：

\[
|F_t| \le \mu F_n
\]

若触觉或力传感器能估计法向力 \(F_n\) 与切向力 \(F_t\)，可以构造摩擦裕度的候选量：

\[
m_t = \hat\mu F_n-|F_t|
\]

当 \(m_t\) 下降时，可能提示接近滑移边界。

但这里不能直接假设 \(\mu\) 已知，也不能把静态 Coulomb 模型当作真实接触的完整描述。可研究的方向包括：
- 用物理先验定义 feature / constraint，再让网络估计未知参数；
- 学习 residual：经典模型给出粗预测，网络补偿未建模动力学；
- 用 friction cone / contact wrench 约束动作空间；
- 比较纯数据驱动与 physics-informed 表示在未见材料、未见载荷上的泛化。

理论力学、材料/接触知识在这里可以成为“提出可测状态和约束”的工具，而不是把复杂摩擦简化成一个未经验证的公式。

## 7. 推荐的阶段性路线

### Stage 1：Tactile representation

目标：知道触觉里到底有什么，哪些量与成功/失败有关。

优先任务：接触、滑移、剪切、时序事件、任务阶段。

### Stage 2：Physical state estimation

目标：从视觉 + 本体 + 触觉历史形成紧凑 latent state，并验证其未来预测与控制充分性。

### Stage 3：Fast local policy

目标：让局部 tactile policy 在高于高层策略的频率运行，验证是否带来可重复的闭环收益。

### Stage 4：Multi-rate hierarchy

目标：把通用 VLA / general policy 与快速 physical policy、低层 controller 接起来，研究接口、频率分配、状态共享与冲突处理。

这是一条长期路线，不要求一篇论文同时完成四个 Stage。每个阶段都应能独立形成清楚的问题、基线与证据。

## 8. 当前最值得保留的研究问题

1. 触觉中的“当前物理状态”与“任务阶段/历史线索”如何区分？
2. 哪些触觉事件需要高采样率才能保留，哪些可以低频压缩？
3. 一个 learned latent state 何时可以近似替代长历史，使局部 policy 更接近 Markov？
4. 高频局部策略的收益究竟来自频率、低延迟还是模态本身？
5. 能否用力学先验构造比 raw tactile 更有跨材料/跨物体泛化能力的状态？
6. general policy 与 physical policy 最合适的接口是什么：pose、skill token、residual action，还是 latent physical state？
7. 什么变量应该在 5–20 Hz、50–200 Hz、500 Hz+ 三个时间尺度上传递？

## 9. 证据标准

未来如果沿此路线做实验，至少要同时报告：
- 真正的 sensor / policy / command / controller 频率，而不是只写 nominal Hz；
- end-to-end latency 和 jitter；
- 无触觉、低频触觉、错位触觉等消融；
- 多物体、多材料、多载荷条件；
- 成功率之外的 slip rate、peak force、恢复时间、接触冲击等物理指标；
- 对失败案例的时序日志，而不是只保留成功视频。

任何 attention heatmap、embedding 可视化或线性 probe 都只能作为诊断证据，不能单独证明“模型真正使用了某个物理变量”。优先采用可干预、可复现的闭环实验。
