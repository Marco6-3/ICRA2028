# 触觉记忆与执行时修正：论文地图

检索/核验日期：2026-10-05。使用 Exa 检索与原始论文/作者项目页核对；只收录与主线直接相关的 11 项，不宣称穷尽。读过论文不等于运行过代码或复现过结果。未列论文的会议录用状态均不作推断。

## 阅读顺序与差异

| 优先级 | 论文与原始来源 | 本次已读范围 | 与主线的关系 / 需要警惕的重复 |
| --- | --- | --- | --- |
| 核心 | [T-Rex: Tactile-Reactive Dexterous Manipulation](https://arxiv.org/html/2606.17055v2)；[项目](https://tactile-reactive-dexterous.github.io/) | v2 方法 §4、实验 §5 及部分附录文本 | 已有短时历史和快慢修正，不能把“历史触觉 + 异步”本身当贡献；[笔记](T_REX.md) |
| 核心 | [TacMamba: A Tactile History Compression Adapter Bridging Fast Reflexes and Slow VLA Reasoning](https://arxiv.org/html/2603.01700v1) | v1 方法 §IV、实验 §V 的设置与比较 | 历史压缩与输入选择性；需强流式基线和完整闭环证据；[笔记](TACMAMBA.md) |
| 核心 | [TacForcing: Streaming Action Generation with Execution-Time Tactile Feedback](https://arxiv.org/html/2608.25798) | 方法 §3、实验 §4 含 EATA 消融 | 执行时更新与动作块对齐；“后续块不更新”是误读；[笔记](TACFORCING.md) |
| 硬件/表示 | [FlexiTac: Flexible and Low-Cost Tactile Sensing](https://arxiv.org/abs/2604.28156) | 当前核对硬件规格、阵列读出与传感原理；尚未运行实物 | 为 spatial tactile field → compact contact state 提供第一候选 testbed；绝对压力仍需标定，不能把 ADC 直接当 N/Pa |
| 核心基线 | [LeFlexiTac: Giving Robots a Sense of Touch](https://tna001-ai.github.io/LeFlexiTac/) | 项目页与公开说明；需在正式实验前固定代码提交和 π0.5 适配实现 | dense FlexiTac map / tactile tokens → LeRobot policies / π0.5；是“为什么不直接 token 化 tactile map”的直接 baseline。项目在其设置中报告 full fine-tuning 优于 action-expert-only / LoRA，但不能推广成 π0.5 的普遍必需条件 |
| 部署参考 | [VTAP](https://yuhao-zhou.com/vtap/index.html) | 当前核对 FlexiTac 指尖部署与 tactile-reactive manipulation 用法 | 证明 FlexiTac 可进入机器人指尖操作平台；不代表本项目的 physics bottleneck 已有现成实现 |
| 核心基线 | [Reactive Diffusion Policy](https://reactive-diffusion-policy.github.io/)；[论文](https://arxiv.org/abs/2503.02881) | 作者项目页与论文方法、任务、推理时间说明 | slow-fast tactile-reactive control；代表 fast action reaction。不要把理论 forward rate 等同真实控制频率，也不要未经源码核查声称其没有长时状态 |
| 强基线 | [ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning](https://arxiv.org/html/2512.10946)；[项目](https://implicit-rdp.github.io/) | 摘要、引言、方法检索摘录，尚未逐项审计实验 | 已有因果 GRU 力编码、因果注意力和统一快慢融合；是新记忆方案必须认真面对的比较对象 |
| 预测分支 | [ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation](https://arxiv.org/html/2506.15953) | 摘要/引言，另核对方法检索摘录 | 视觉触觉融合 + 未来触觉预测辅助动作；预测是备选拓展，不强塞进第一篇 |
| 长期延伸 | [DexTacWAM: A Visuo-Tactile World-Action Model for Dexterous Manipulation](https://arxiv.org/html/2609.24976v1)；[项目](https://dextacwam.github.io/) | v1 摘要/引言、方法及消融检索摘录 | 视觉触觉联合世界状态预测服务动作生成；不能描述成只能预测、不能出动作 |
| 方法基础 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)；[官方代码](https://github.com/state-spaces/mamba/) | 摘要及选择性机制检索摘录；未全面复核实现 | 输入依赖的状态更新和流式压缩；不是物理系统辨识或无损记忆的保证 |

未带版本号的 HTML 来源只固定本次读取日期；实现前需固定具体论文版本与代码提交。以上项目链接提供来源入口，不代表已核验权重、训练脚本、许可证或硬件兼容性。

## 把论文放在同一个问题里读

- **表示什么**：FlexiTac / VTAP 提供二维法向接触场；LeFlexiTac 提供 dense tactile map/token 直接接现代 policy/VLA 的现实 baseline；本项目检验这些空间观测能否进一步压缩成 compact physical contact state。
- **记住什么**：TacMamba 是核心 long-memory baseline；T-Rex 提供近期 temporal force + spatial deformation 编码参照。
- **何时使用**：RDP 是核心 fast-reactive baseline；TacForcing / ImplicitRDP 提供 execution-time feedback 与统一 slow-fast 融合参照。
- **是否需要预测**：ViTacFormer、DexTacWAM 提供预测路径；预测误差低不自动等于控制收益。
- **如何改进已有方法**：以选定论文的原方法为首要基线，围绕新增机制设计消融；Mamba 等仅是可能的方法来源，具体比较由贡献决定。

当前优先核查顺序：TacMamba → LeFlexiTac → RDP → T-Rex / TacForcing / ImplicitRDP → FlexiTac / VTAP。三篇核心 baseline 分别锚定 memory、dense tactile-to-policy、fast reactivity；其余工作用于限制 novelty 边界和补充设计依据。

## 本项目需要持续维护的差异表

每次新论文出现，只更新它对核心问题的影响：已有方法是否已经包含选择性记忆？是否输出执行时修正？是否比较了同延迟预算的递归模型？是否在未见接触条件下测试？

若近作已经解决当前提案，不靠换名继续推进；缩小到尚未被证据覆盖的问题，或改变假设。不要把这份文献表当成“证明新颖性”的最终结论。
