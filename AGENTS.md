# Research Workspace Rules

本仓库是机器人操作研究 idea 工作区。README.md 是方向索引，具体研究问题放在 `ideas/`。

## 1. 不恢复已删除旧方向

用户已经明确要求：此前仓库中的所有旧研究方向均不再保留。不要从 Git 历史、旧摘要、旧 TODO 或其他对话中自动恢复 OM-1、旧 D1-D7、夹爪宽度、旧 pilot 等内容。需要追溯时只通过 Git history 查看，不重新写回当前目录，除非用户再次明确要求。

## 2. 一个 idea 一个文件

每个 idea 使用独立文件：

```text
ideas/Ixxx_SHORT_NAME.md
```

README 只维护索引和极短摘要，不把某一个 candidate 自动提升为整个仓库唯一主线。

每个 idea 建议包含：

- Problem
- Why the problem may exist
- Terminology / scope
- Hypotheses
- Proposed architecture or method
- Data and sensing requirements
- Mechanics / physics variables worth measuring
- Minimal experiments
- Baselines and ablations
- Metrics
- Generalization split
- Falsification / stop conditions
- Risks and nearest-work questions
- Next actions

## 3. 科研表述

- “可能”“假设”“待验证”与“已经实验证明”必须严格区分。
- 不因为结构看起来合理就宣称新颖性；相关工作需要单独检索。
- attention、latent visualization、linear probe 等只能作为诊断，不单独证明因果机制。
- 先设计能推翻假设的实验，再考虑扩模型。
- 评价按 episode / object / trajectory / seed 等独立单位统计，不能把连续帧当独立样本。
- 所有在线方法只能使用决策时刻以前已到达的信息，避免未来信息泄漏。
- 记录 sensor rate、policy rate、communication latency、actuation rate 与 end-to-end reaction latency，不能只报告网络 forward latency。

## 4. 触觉与物理变量

研究触觉时，不默认需要完整机器人动力学模型。优先区分：

- raw tactile signal
- contact mechanics / contact state
- full robot dynamics

法向力、剪切力、力矩、滑移、接触面积、形变、振动、变化率等可以作为测量量、privileged labels、分析变量或控制变量；是否真正有用由消融和闭环结果决定。

## 5. 修改仓库时

新增方向优先扩展 `ideas/`，不要把所有相近想法写进同一个文件。若一个 idea 已经明显分裂成不同科学问题，应拆成新的 ID。

