# ICRA2028 · 触觉接触状态与长时记忆

更新：2026-10-06。状态：已完成数学检查与两轮公开数据探索性离线验证；完整 11D 的增量价值尚未可靠成立，Stage 1 尚未通过，未启动 Mamba / VLA 或实机闭环实验。

## 已完成的验证与当前判断

- **数学检查**：加权零至二阶矩的数值关系通过检查，同时构造出不同原始场对应相同 11D 的精确反例；统计量自洽不等于控制充分或无损。
- **LeFlexiTac 动作预测**：64 episode、75,041 帧；Physics11D 相对 State-only 的平均 normalized MSE 高约 4.5%，没有证明空间算子的额外收益。
- **TaF 空间诊断**：以独立 ATI 力矩／法向力比为标签，17 个完整源记录、417,986 测试帧、3 个神经训练种子。稳载子集上 11D 相对 Scalar 的平均 RMSE 低 6.4%，但逐源配对 MSE 区间跨零，且没有超过“总量＋质心”。去掉窗口固定偏置后，11D 的变化预测误差比零变化参照高 10.2%。

因此先保留质心作为候选，继续受控物理验证和消融；不把完整 11D 的必要性、滑移识别、高频闭环或实机收益写成已成立结论。

详见 [成果与决策总览](research/OFFLINE_RESULTS.md) 和 [实验归档／复现入口](experiments/contact_operator/README.md)。

## 一条长期主线

**研究机器人应当保留哪些接触信息，以及如何把这些信息压缩成能够支持长时间、低延迟闭环控制的内部状态。**

当前最高优先级候选问题：

> **What is the minimal tactile state that preserves control-relevant spatial contact information over long horizons?**

当前候选切入点是 **Physics-Guided Contact-State Compression + Streaming Memory**：先用 FlexiTac 一类二维触觉阵列研究“高维接触场 → 低维、物理可解释 contact state”的信息保真与失败边界；同时允许一个严格受限的 Stage-2P probe 检验 representation × memory interaction。正式 Mamba/VLA 系统仍需前序证据，但不再用严格串行门控错过“单帧不显著、长历史才显现”的可能价值。

当前实验定位锚定三类强基线：**TacMamba = long tactile memory**、**LeFlexiTac = dense FlexiTac-to-policy/VLA**、**RDP = fast tactile-reactive control**。T-Rex 与 TacForcing继续作为多速率触觉与 execution-time conditioning 的关键 supporting work。详见 [Core Baselines](research/BASELINES.md)。

## 当前候选证据链

Raw tactile field → Physics contact state → Representation evidence → Temporal memory → Closed-loop control

对应三个阶段：

1. **Stage 0 — Physics sanity check**：不训练网络，检查 descriptor 的噪声、漂移、重复性、事件响应和真实延迟。
2. **Stage 1 — Representation bottleneck**：比较 Raw / LeFlexiTac-style dense tactile tokens / Physics 11D / Learned 11D 的数据效率、OOD、延迟、训练代价和信息丢失。
3. **Stage 2P — matched temporal probe**：在 Stage 1 尚未正式通过时，允许 Scalar / Scalar+centroid / Physics11D / Learned11D 接入同一 streaming backbone，专门检查表示×记忆交互；不接 VLA，不视为正式 Stage 2。
4. **Stage 2/3 — Formal temporal memory → Closed loop**：只有前序证据支持后才扩大到正式时序系统与真机闭环。

详细定义见 [Physics-Guided Contact-State Compression](research/PHYSICS_BOTTLENECK.md)。

## 目标年份

以 **2027 年完成研究并投稿 ICRA 2028** 为目标，而非承诺录用。ICRA 2028 的正式截止、页数与投稿细则以最终 CFP / PaperPlaza 为准；内部暂按 **2027-07-31 完整初稿** 留出缓冲。来源与日期冲突见 [路线与时间表](ROADMAP.md)。

## 导航

| 文档 | 用途 |
| --- | --- |
| [研究主线](research/CORE.md) | 核心问题、候选假设、方法边界与可能的论文贡献 |
| [Physics Bottleneck](research/PHYSICS_BOTTLENECK.md) | 当前最高优先级候选：FlexiTac contact state、预处理、Stage 0/1/2 与停止条件 |
| [Core Baselines](research/BASELINES.md) | TacMamba / LeFlexiTac / RDP 的角色、三类诊断任务与 Go/No-Go 逻辑 |
| [实验方案](research/EXPERIMENTS.md) | 公平比较、因果时间约束、指标和投稿前证据门槛 |
| [离线成果与决策](research/OFFLINE_RESULTS.md) | 数学反例、两轮离线结果、负结果、协议修订与当前停止条件 |
| [实验代码与归档](experiments/contact_operator/README.md) | 数据来源、固定版本、协议、图表、模型和复现命令 |
| [研究路线](ROADMAP.md) | 从当前小实验到 2027 年投稿准备 |
| [论文地图](papers/README.md) | 核心与邻近文献、来源、已读范围和与本路线的关系 |
| [T-Rex](papers/T_REX.md) | 快慢专家、时序触觉编码与去噪过程 |
| [TacMamba](papers/TACMAMBA.md) | 流式历史压缩、1D force sensing 与证据边界 |
| [TacForcing](papers/TACFORCING.md) | 执行时触觉、EATA，以及对既往理解的纠正 |
| [协作规则](AGENTS.md) | 维护假设、证据和实验记录约定 |

## 现在先做什么

1. 固定 FlexiTac 硬件 / 固件版本与真实数据格式，实测采样率、传输和 preprocessing latency；不要沿用文献数字代替本机测量。
2. 已有解析 descriptor 与公开数据实现；仍需完成真实 FlexiTac 的 Stage 0：contact → load → redistribution / roll → release 的受控重复数据。公开数据数值检查不能替代硬件噪声、漂移和事件重复性。
3. 在真实校准与独立标签下比较 Scalar、Scalar+centroid、Physics11D、Raw 与 Learned11D；现有离线结果不足以通过继续条件。dense tactile token 路径尚未运行，不能宣称优于它。
4. 现在可并行启动一个低成本 Stage-2P：固定同一 temporal backbone，比较 Scalar、Scalar+centroid、Physics11D、Learned11D；若 Physics11D 不能超过更小的 spatial state，则优先缩 descriptor，而不是把 11D 当既定贡献。
5. 真正的 coupled spatiotemporal 任务必须同时包含“Why Spatial”和“Why Memory”的可观测性缺口；在独立真值确认前，不把 CoP/shape 漂移直接称为微滑移或失稳前兆。

**不把好看的相平面图当成论文证据，不把 Mamba、FlexiTac 或“physics”本身当 novelty。** 最终贡献必须来自可重复的增量证据和明确失败边界。

本仓库管理研究问题与证据，不重建通用 infra。
