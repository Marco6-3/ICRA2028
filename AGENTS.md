# ICRA2028 research collaboration

用户当前目标：围绕触觉接触状态、长历史记忆与闭环操作整理一条长期研究路线，在 2027 年准备并投稿 ICRA 2028。以 README.md、research/CORE.md、research/PHYSICS_BOTTLENECK.md 为当前问题定义；用户后续明确指令优先。

- 当前最高优先级候选问题：高维 tactile field 能否压缩成低维、physically interpretable contact state，并作为高频、长历史 tactile memory 的有效接口。
- 当前证据顺序固定为：Stage 0 physics sanity check → Stage 1 representation bottleneck → Stage 2 temporal memory → Stage 3 closed-loop control。未经用户明确要求，不要跳过前面阶段直接训练 Mamba / VLA。
- FlexiTac、11D descriptor、Mamba 都是候选工具，不是已成立贡献。不能把“用了 physics / Mamba / 新传感器”当 novelty。
- Stage 0 不使用复杂神经网络；先检查噪声、漂移、重复性、event response、preprocessing artifact 和真实 P50 / P95 latency。
- Stage 1 必须至少比较 Raw、Physics compact state、同维 Learned compact latent。Physics 的目标是 inductive bias / compactness / OOD / latency / interpretability，不得声称其信息量高于 Raw。
- CoP 漂移只称为 contact redistribution cue，不能未经验证直接解释为 slip。FlexiTac 等法向阵列也不能未经标定声称直接输出绝对 N / Pa。
- preprocessing 必须因果：baseline / normalization / threshold / filtering 不得使用测试 episode 的未来信息。no-contact baseline 若在线更新，接触时按协议冻结。
- 空间 descriptor 优先使用归一化传感器坐标；总压力尺度不默认使用在线滑动 Z-score，以免长期稳定载荷被重新中心化。
- 主方向编码避免直接使用裸 theta；对各向同性接触必须有 orientation confidence / degeneracy handling。
- Moment-matched 对抗样本优先从真实采集数据中检索，而不是通过人工调压力宣称多个统计量“完全一致”。
- Stage 2 只有在 Stage 1 通过继续条件后启动。TacMamba / Mamba 是强候选，但 temporal backbone 由证据决定，不预设网络架构。
- 最终若声称 tactile-reactive manipulation 改善，必须报告真实物理事件到动作生效的端到端 latency，而不是只报 descriptor 或网络 forward。
- 先读最接近论文及其消融再提出缺口。T-Rex 已编码 temporal force 与 spatial deformation；TacMamba 已有 1D force 流式长历史；TacForcing 已有 execution-time tactile feedback。不能声称这些能力尚不存在。
- 论文事实、作者报告、项目推断、待验证假设和本项目结果分开记录。未运行不能声称复现，未确认不能声称首次、SOTA 或必然录用。
- 新结果应记录设置、硬件 / 固件版本、代码提交、独立试验单位、全部失败与负结果、原假设和更新后的判断。不要追溯性改写假设。
- 仿生 mechanoreceptor / spinal-cortical 类比最多作为设计启发；没有定量对应时不能作为机器人方法正确性的证据。
- 不自动扩张到世界模型、跨本体、事件触发通信、通用 VLA、多传感器大系统；这些只在当前核心 claim 成立后考虑。
- 本仓库管理研究问题和证据；已有 infra 另行使用，不扩张成泛用机器人平台建设任务。

- **Core Triad**：TacMamba = long tactile memory baseline；LeFlexiTac = dense FlexiTac-to-policy/VLA baseline；RDP = fast tactile-reactive control baseline。正式实验设计优先围绕这三条轴组织，而不是堆叠散乱 baseline。
- LeFlexiTac 只能表述为：其公开项目在特定 π0.5 设置中采用 tactile tokens，并报告 full fine-tuning 优于其 action-expert-only / LoRA 尝试。不能扩写成“π0.5 触觉融合普遍必须 full fine-tune”，也不能未经实测写死其实际 tactile loop 频率。
- 不用“first FlexiTac + Mamba”作为主要 novelty。未检索到关键词组合不是新颖性证明；novelty 必须落在 dense tactile → physical bottleneck → persistent streaming memory → policy interface 的完整问题与证据链。
- RDP 的角色是 fast reactive control baseline。不要把其理论网络吞吐当作真实闭环频率，也不要未经论文/源码核查写成“没有长时记忆”。
- 三个诊断任务的逻辑优先于任务数量：Task 1 temporal non-regression；Task 2 spatial necessity；Task 3 history necessity。不要提前写死成功率、50 ms 等结果。
