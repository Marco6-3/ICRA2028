# ICRA2028 长期研究路线与投稿安排

更新：2026-10-07。

## 稳定问题

长期问题不变：

> 机器人应该把高维、快速变化的触觉场压缩成什么样的因果内部状态，才能支持低延迟、长历史、接触丰富的闭环控制？

但当前入口已经根据 2026-10-06 的离线负结果收缩：

**Dense tactile field → Minimalist Contact Flow → Streaming contact memory → Closed-loop recovery**

完整静态 11D 不再是必须证明的贡献，而是历史 ablation / failure boundary。

## 当前 Stage 1：先体面收尾，不采新数据

默认 Minimalist Contact Flow：

```text
z_t=[f_N, CoP_x, CoP_y, A]
MCF_t=[z_t, z_t-z_{t-1}] ∈ R^8
```

第一轮复用已有 LeFlexiTac / TaF 数据，不以平滑全轨迹 action MSE 为主指标，而聚焦 contact-active windows。

优先顺序：

1. **TaF torque-delta / eccentric-load-change**：使用同步 ATI 独立标签；
2. **contact-instability / redistribution event prediction**；
3. 只有存在独立 slip / object-motion 真值时才升级为 **slip-onset prediction**。

固定比较 Scalar、Static first-order state、MCF 8D、Dense Raw；Physics11D 仅作历史消融。

Stage 1 详细协议见 [MINIMALIST_CONTACT_FLOW.md](research/MINIMALIST_CONTACT_FLOW.md)。

## Stage 1 通过条件

必须同时满足核心科学与工程条件：

- MCF 在至少一个独立标注 contact-active target 上相对 Scalar 有可重复增量；
- 消融支持增量来自 CoP / causal dynamics；
- 与 Dense Raw 的差距落在 test 前冻结的 non-inferiority margin 内，或明确承认 dense equivalence 尚未证明；
- 紧凑性/计算成本优势可量化。

若 MCF > Scalar 但 < Dense Raw，则不是失败：转向 compact flow + sparse/event-triggered rich tactile 的 hybrid 路线。

若 MCF ≈ Scalar，则停止在现有数据上继续调 descriptor，改做真正需要 spatial observability 的受控任务。

## 2026-10 至 2027 投稿路线

| 阶段 | 计划时间 | 必须形成的产物 | 继续条件 |
| --- | --- | --- | --- |
| **Stage 1A：离线 MCF 收尾** | 2026-10 | 8D extractor；contact-active subset；Scalar/Static/MCF/Raw 对照；独立标签事件/力矩结果 | MCF 对 Scalar 有可靠增量；dense gap 有明确结论 |
| Stage 1B：必要时确认性采集 | 2026-10 至 11 | 只针对离线证据暴露的缺口采受控数据；随机化偏心方向/载荷；独立真值 | 排除数据集固定偏置与伪标签解释 |
| Stage 2：temporal memory | 2026-11 至 2027-01 | current / short-window / streaming-memory matched comparison；early prediction | 长历史带来可重复增量，且不是参数量/可见信息不公平 |
| Stage 3：closed-loop recovery | 2027-01 至 04 | contact redistribution / disturbance recovery 等闭环任务；真实 event→action latency | 表示/记忆收益转化为闭环成功率或安全性 |
| 泛化与关键消融 | 2027-04 至 05 | unseen objects/contact conditions；核心组件消融；nearest baselines | 核心 claim 跨条件成立或得到清晰边界 |
| 完整证据与论文 | 2027-06 至 07 | 结果表、延迟、失败、视频、初稿 | 形成一个清晰主张，不靠“用了 Mamba/FlexiTac” |
| 投稿准备 | 2027-08 起至正式截止前 | 导师审阅、相关工作、匿名/格式/补充材料 | 以最终 ICRA 2028 CFP 为准 |

内部完整初稿仍以 **2027-07-31** 为项目管理目标，不代表官方截止。

## Stage 2：Streaming Contact Memory

只有 Stage 1 证明 MCF 至少比 Scalar 更有用，才正式研究：

- current MCF；
- fixed short-window MCF；
- GRU/TCN 等简单 causal memory control；
- TacMamba/Mamba-style streaming memory。

目标不是证明“Mamba 好”，而是：

> **接触变化是否需要比当前帧/短窗口更长的历史，以及 streaming state 能否以低延迟保存这种信息？**

主要看 early prediction、扰动状态识别、历史依赖歧义解除和计算/延迟，而不是只看普通 frame-level accuracy。

## Stage 3：Closed-loop recovery

最终论文必须把 representation/memory 转成行为证据。优先任务围绕：

- eccentric-load/contact redistribution recovery；
- disturbance during grasp；
- insertion/search 中当前 tactile state 歧义但 history 不同的场景。

最终报告 success/quality、event→action P50/P95、危险接触事件、unseen condition 与 failure taxonomy。

## 失败时如何收缩

- MCF ≈ Scalar：当前任务缺 spatial necessity，换受控问题，不继续刷特征。
- MCF > Scalar but < Raw：走 hybrid compact+rich tactile。
- MCF ≈ Raw：Stage 1 最理想结果，进入 memory。
- Long memory ≈ short window：论文不强行使用 Mamba，缩成 compact dynamic tactile representation + reactive control。
- 离线收益不能转闭环：缩小 claim，不用更多离线 benchmark 掩盖。

## 第一篇之后的自然延伸

只有第一条证据链成立后，再考虑 event-triggered rich tokens、action-conditioned tactile state、跨传感器 adaptation、VLA 接口等扩展。这些都不是当前 Stage 1 必须完成的组件。
