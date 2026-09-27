# Robot Manipulation Research Ideas

这个仓库用于记录和推进具身智能 / 机器人操作中的研究 idea。它不是一个已经冻结的单一课题，也不默认任何 idea 已经具备论文贡献。

当前原则：

- 每个方向独立放在 `ideas/` 下，先写清问题、假设、最小实验、失败条件，再决定是否投入更多资源。
- 新方向可以彼此相近，例如都围绕触觉、物理交互、多时间尺度控制或 in-context adaptation，但不要为了“凑一篇论文”强行合并。
- 仓库只保留当前仍认可的研究 idea；已经放弃的旧方向不在当前目录归档，需要追溯时使用 Git history。
- 文档中的模型结构、频率、传感器与机器人平台默认都是待验证设计，不把建议写成实验事实。

## Ideas

| ID | 方向 | 状态 | 核心问题 |
| --- | --- | --- | --- |
| [I001](ideas/I001_HIGH_FREQUENCY_TACTILE_CONTEXT.md) | 高频触觉反馈中的时序上下文与 in-context adaptation | Candidate | 高频触觉策略是否应利用短期历史估计当前接触状态，并利用更长的交互历史在线推断接触动力学，从而在不更新权重的情况下调整局部控制？ |
| [I002](ideas/I002_ASYNC_SEMANTIC_TACTILE_COMMUNICATION.md) | 异步多速率 semantic–tactile policy 通信 | Candidate | 低频 semantic policy 与高频 tactile policy 在异步运行时应该共享什么、共享多少、允许信息陈旧多久，以及是否需要双向 / 事件触发通信？ |

后续新增方向时，在 `ideas/` 中创建新的 `Ixxx_*.md`，并把索引加入上表。

## 当前方向关系

### I001 — fast tactile policy 内部需要什么历史？

I001 研究的是 tactile branch 自身：

```text
tactile / action / proprioception history
                │
                ▼
      short contact context
                +
      long interaction context
                │
                ▼
       fast tactile policy
```

重点是：

1. 短时历史是否帮助估计 contact state；
2. action / proprioception history 是否帮助区分环境变化与机器人自身动作造成的变化；
3. 更长 interaction history 是否能够形成不更新权重的 in-context adaptation。

完整细节见 [I001](ideas/I001_HIGH_FREQUENCY_TACTILE_CONTEXT.md)。

### I002 — fast tactile 与 slow semantic 之间如何通信？

I002 不预设“少量共享 token 一定最好”，而是把通信机制本身作为实验变量：

```text
vision / language / task state
            │
            ▼
      semantic policy
          slow rate
            │
            ▼
      shared memory / tokens
            ▲
            │
       tactile policy
          fast rate
            │
            ▼
      refined robot action
```

核心变量包括：

- shared token / latent 的数量与内容；
- coarse action only vs compressed tokens vs full latent；
- semantic context staleness；
- semantic -> tactile 单向 vs 双向通信；
- fixed-rate vs tactile-event-triggered semantic update；
- performance / reaction latency / compute / communication bandwidth 的 trade-off。

第一阶段计划以 **Franka 仿真**作为实验平台，但研究问题不绑定 Franka。上层尽量使用 end-effector delta pose + gripper command，并通过 robot-specific controller 适配后续可能获得的其他真机机械臂。

完整细节见 [I002](ideas/I002_ASYNC_SEMANTIC_TACTILE_COMMUNICATION.md)。

## Repository structure

```text
.
├── README.md
├── AGENTS.md
└── ideas/
    ├── README.md
    ├── I001_HIGH_FREQUENCY_TACTILE_CONTEXT.md
    └── I002_ASYNC_SEMANTIC_TACTILE_COMMUNICATION.md
```
