# 投稿日期与研究资料索引

核查日期：**2026-09-16**。

本页是服务于当前选题的定向检索，**不是穷尽综述，也不是对论文结论的独立复现**。论文信息主要依据原作者的 arXiv 摘要和项目页；“选题边界”是本仓库基于这些资料的判断，不是原论文已经证明的结论。完整方法、实验设置和代码可用性需在正式立项前继续核验。

专题索引：[OM-1 的公开证据与未知项](OM1_EVIDENCE.md)、[触觉相关工作与选题边界](TACTILE_RELATED_WORK.md)。

## 投稿时间依据

| 项目 | 本次核查结果 | 来源状态与边界 |
|---|---|---|
| IROS 2027 论文截止日期 | **2027-03-01** | [IEEE RAS 官方活动页](https://www.ieee-ras.org/event/2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-70525/)已列出该日期，已打开读取；并非仅依据往年估算。页面未列具体时刻、时区、视频及补充材料截止时间，仍需核对详细 CFP 与投稿系统。 |
| IROS 2027 会期与地点 | **2027-09-26 至 2027-10-01，意大利佛罗伦萨** | 上述活动页与[IEEE RAS 历届及未来会址页](https://www.ieee-ras.org/conferences-workshops/financially-co-sponsored/iros/iros-past-and-future-venues/)交叉确认，两页均已读取。 |
| IROS 2027 详细 CFP | 本轮未取得可核验的详细 CFP | 本轮打开 `2027.ieee-iros.org` 时重定向至 2023 网站；`www.iros2027.org` 未能打开。不能把这两个入口当作已读到的 2027 CFP，也不能因此否定官方活动日历已经给出的日期。 |
| IROS 2026 历史日期 | 本页不采用其日期推算 2027 截稿 | [2026 CFP 入口](https://2026.ieee-iros.org/contribute/call-for-papers/)本轮受访问限制，未完成原文核验；不把第三方日期记录写成已核验的官方事实。 |
| ICRA 2027 新论文提交 | 官方列 **2026-09-15 23:59 PST**；不适合作为本次新立项的赶投目标 | [ICRA 2027 官方 CFP](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)已读取，FAQ 写明没有延长论文截止日期的计划。其视频补交窗口不等于延长新论文提交时间。 |

本项目按 **2027-03-01** 准备，不依赖延期。从核查日至该日期约 **166 天，即 5.5 个月**；这是日历日期差，不是按尚未确认的截止时区计算的精确倒计时。内部实验与写作里程碑见仓库任务计划。正式投稿前须再次核验最新 CFP、投稿系统和材料要求。

## 动作语义、上下文学习与短期接触预测

| 研究与一手链接 | 原作者公开内容 | 对本仓库选题的边界 |
|---|---|---|
| **FAST: Efficient Action Tokenization for Vision-Language-Action Models**，2025；[论文](https://arxiv.org/abs/2501.09747) | 用离散余弦变换压缩连续动作序列，使高频动作能够更有效地进行自回归 token 预测。 | 动作压缩不自动建立“拿起杯子”这样的任务语义。研究语义不变性时，应检查相同物体效果但不同运动轨迹的表征与迁移表现，不能只展示 token 或嵌入聚类。 |
| **Latent Action Pretraining from Videos（LAPA）**，ICLR 2025；[论文](https://arxiv.org/abs/2410.11758)、[作者项目](https://latentactionpretraining.github.io/) | 从视频帧之间学习离散潜在动作，用其预训练 VLA，再用少量机器人动作数据将表征适配到机器人动作空间。 | 无动作标签预训练不等于零机器人数据部署；潜在动作也不保证只包含任务信息。需区分物体效果、机械臂运动与背景或相机变化。 |
| **UniVLA: Learning to Act Anywhere with Task-centric Latent Actions**，RSS 2025；[论文](https://arxiv.org/abs/2505.06111) | 结合语言和 DINO 特征空间学习以任务为中心的潜在动作，减少与任务无关的动态因素，进行跨本体策略学习。 | 这是“动作语义空间”想法的直接相关工作。仅提出让同类动作靠近不足以构成新意；接触阶段或物体状态变化是否带来额外不变性，需要控制变量和下游任务证据。 |
| **In-Context Imitation Learning via Next-Token Prediction（ICRT）**，2024 预印本；[论文](https://arxiv.org/abs/2408.15980)、[作者项目](https://icrt.dev/) | 以图像、状态和动作组成的机器人示教轨迹为上下文，测试新任务时不更新策略权重。作者项目页介绍了预训练及多任务数据，并强调多任务环境有助于模型学习使用 prompt。 | 测试时无梯度更新不等于此前无需训练；其示教是机器人遥操作轨迹，并非任意人类视频。固定当前观测后替换、删除、乱序或提供错误 prompt，可帮助检验策略是否真实依赖上下文；这些干预实验不等于已经解释内部机制。 |
| **TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation**，2026-06 预印本；[论文](https://arxiv.org/abs/2606.11184) | 在紧凑潜空间中预测短期触觉动态，以高频腕部力/力矩为条件，并将预测表征提供给策略，辅助接触操作。 | “小型触觉世界模型加策略”已有直接研究。原方法涉及双指触觉与腕部力矩，不能默认当前平台具备。应检验在相同历史信息、容量和延迟下，未来接触预测何时优于纯反应策略，而不只比较预测误差。 |

阅读范围：上表五篇的 arXiv 摘要均已读取；ICRT 作者项目页的方法、数据及评估介绍亦已读取。本页未运行上述仓库、训练模型或复现论文结果；页面中的开源承诺不等于本轮已经验证相应数据与权重可下载。

## 主动触觉探索与预测控制的早期直接工作

- **SwingBot: Learning Physical Features from In-hand Tactile Exploration for Dynamic Swing-up Manipulation**，论文页标注 IROS 2020，arXiv 上传于 2021；[论文](https://arxiv.org/abs/2101.11812)。作者通过倾斜和摇晃获取触觉信息，学习物体物理特征表征，预测动态摆起角度，并据此搜索控制参数。这说明“先触摸或晃动，再利用识别到的物理特征执行任务”已有直接先例；若研究短期物理交互上下文，应明确相比此类物理特征识别的新问题。
- **Manipulation by Feel: Touch-Based Control with Deep Predictive Models**，ICRA 2019；[论文](https://arxiv.org/abs/1903.04128)。作者以高分辨率触觉输入学习动态预测模型，通过 deep tactile MPC 实现朝目标触觉状态的操作，并演示球、摇杆和多面骰子的操纵。这说明“预测未来触觉并用来控制”本身并非新颖点；新工作需落实到预测对象、决策收益、数据效率、扰动适应或迁移范围等可检验差异。

以上两篇已读取摘要，尚未核验全部实验细节或独立复现。后续若采用相关方法作基线，应补读正文与代码，并记录传感器、动作接口、训练数据和评价设置的差异。
