# 一条长期研究路线与投稿安排

更新：2026-10-06。下表为长期计划，完成状态不能从日期自动推断。已完成数学检查、公开数据筛查与两轮探索性离线验证；真实硬件 Stage 0 尚未完成，完整 11D 的 Stage 1 继续条件尚未通过。见 [成果与决策](research/OFFLINE_RESULTS.md)。

## 稳定问题与当前候选入口

长期问题是：机器人如何把快速、高维的触觉交互压缩成有用内部状态，并在长时间执行过程中利用这些状态进行低延迟闭环控制？

当前第一候选入口是 **Physics-Guided Contact-State Compression**：

高维触觉阵列 → compact physical contact state → representation validation → temporal memory → closed-loop control

这条路线必须逐层通过继续条件。FlexiTac、11D descriptor 和 Mamba 都是候选工具，不是预先保证成立的贡献。

## 2026-10 至 2027 投稿路线

| 阶段 | 计划时间 | 必须形成的产物 | 继续条件 |
| --- | --- | --- | --- |
| Stage 0：硬件与 physics sanity check | 2026-10 | 固定 FlexiTac / 触觉硬件版本；raw recorder；11D descriptor；重复 contact/load/redistribution/release 数据；P50/P95 latency | descriptor 数值稳定、事件响应可重复、没有明显 preprocessing artifact |
| Stage 1：representation bottleneck | 2026-10 至 12 | Raw / Physics 11D / Learned 11D；data-efficiency 曲线；OOD；moment-matched failure pairs；compute/latency | Physics 至少在 compactness、低样本、OOD、延迟或诊断价值上有清晰收益，且信息损失可接受 |
| Stage 2：temporal memory | 2026-12 至 2027-02 | scalar / physics / learned / rich history 的公平 temporal comparison；固定刷新率与延迟预算 | 长历史机制带来可重复增量，且收益不是更多参数、历史或刷新率造成 |
| Stage 3：closed-loop task | 2027-02 至 04 | 由前两阶段选择的 contact-rich tasks；真实 event→action latency；成功率与失败 taxonomy | 表示 / memory 的收益能转化为实际闭环行为 |
| 泛化与关键消融 | 2027-04 至 05 | unseen objects/contact conditions；主要机制消融；强邻近方法比较 | 论文核心 claim 跨条件成立或得到清晰边界 |
| 完整证据与论文 | 2027-06 至 07 | 完整结果表、延迟曲线、失败、视频、初稿 | 证据支持一个清晰主张，不依赖“用了 Mamba / physics / 新传感器”本身 |
| 投稿准备 | 2027-08 起至正式截止前 | 更新相关工作；导师与作者审阅；格式、匿名、补充材料核查 | 以最终 ICRA 2028 CFP 为准 |

内部完整初稿日期继续设为 **2027-07-31**，只是项目管理目标，不是官方截止。

## 近期最小产物

近期工作不以“再增加研究 idea”为目标，而以以下可检查产物推进：

1. 固定实际 FlexiTac / 触觉硬件、固件、串口和时间戳格式。
2. 产生一份原始数据记录，包含 no-contact → contact → load → redistribution / roll → release。
3. 生成 11D descriptor 与 preprocessing 配置，能从同一 raw recording 完全重放。
4. 输出 Stage 0 报告：重复性、漂移、event response、P50/P95 preprocessing latency 与 phase-plane 图。
5. 根据 Stage 0 决定是否投入 Stage 1；未通过则先修表示，不训练 temporal model。

## 文献与母体关系

继续精读 TacMamba、T-Rex、TacForcing，并补充 FlexiTac / VTAP 与最接近的 tactile representation 工作。

当前不再要求先选出唯一“母体论文”才允许做 Stage 0，因为 Stage 0 是一个低成本表示 sanity check；但在 Stage 2 / Stage 3 形成正式方法前，必须明确主要方法母体、原论文已有能力和本项目增量。

## 投稿日期核验

查询日期：2026-10-05。

- ICRA 2027 常规论文投稿已经关闭，因此本项目按 **2027 年投稿、目标 ICRA 2028** 管理。
- ICRA 2028 的官方日历目前存在冲突记录；正式 CFP / PaperPlaza 发布后重新核验截止日期、时区、页数与匿名要求。
- 不把仓库名或内部时间表解释为录用承诺。

## 失败时如何收缩

- Stage 0 不稳定：优先修传感器、标定、threshold、baseline 和 filtering，不引入大模型。
- Stage 1 Physics 无优势：保留负结果，转 learned compact latent 或重新定义 contact state。
- Stage 2 temporal memory 无增量：不要为了 Mamba 更换更多骨干，先判断任务是否真的需要长历史。
- Stage 3 离线指标不能转化为闭环：相应缩小论文 claim，而不是继续堆离线实验。

## 第一篇之后的自然延伸

只有第一条证据链成立后，再考虑：

1. fast compact physics state + sparse rich tactile token；
2. event-triggered rich tactile update；
3. 行动条件化 contact state，用于区分自身动作和外界扰动；
4. 跨物体、材料、传感器的 contact-state adaptation。

这些均不是当前第一篇必须一次实现的组件。
