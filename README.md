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

后续新增方向时，在 `ideas/` 中创建新的 `Ixxx_*.md`，并把索引加入上表。

## 当前 I001 的一句话版本

瞬时触觉通常不足以判断“是否应该修正动作”。当前 idea 将触觉控制拆成两个时间尺度：

```text
slow semantic / task policy
        │
        │ nominal action
        ▼
fast tactile loop
  short history  -> current contact state
  long context   -> contact dynamics / object property
        │
        ▼
 residual correction
```

重点不只是“把触觉跑得更快”，而是研究：

1. **短时历史是否是高频触觉反馈真正需要的信息；**
2. **触觉历史中是否必须同时包含 action / proprioception，才能区分环境变化和机器人自己造成的变化；**
3. **更长的 episode interaction history 能否形成类似 in-context system identification 的能力；**
4. **这种 fast tactile policy 是否应该作为 slow semantic policy 的 residual / local controller，而不是做成一个 monolithic multimodal policy。**

完整细节见 [I001](ideas/I001_HIGH_FREQUENCY_TACTILE_CONTEXT.md)。

## Repository structure

```text
.
├── README.md
├── AGENTS.md
└── ideas/
    ├── README.md
    └── I001_HIGH_FREQUENCY_TACTILE_CONTEXT.md
```

