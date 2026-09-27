# Research Ideas

这里存放独立的研究方向。编号只表示记录顺序，不表示优先级或论文主线。

| ID | Title | Status |
| --- | --- | --- |
| [I001](I001_HIGH_FREQUENCY_TACTILE_CONTEXT.md) | High-Frequency Tactile Feedback with Temporal Context and In-Context Adaptation | Candidate |
| [I002](I002_ASYNC_SEMANTIC_TACTILE_COMMUNICATION.md) | Asynchronous Multi-Rate Semantic–Tactile Policy Communication | Candidate |

## 当前边界

- **I001**：研究 fast tactile policy 自身应该利用什么短期 / 长期 interaction history。
- **I002**：研究 slow semantic policy 与 fast tactile policy 在异步、多速率执行时如何通信。

它们可以共享 simulator、dataset、logging 和 tactile encoder 等基础设施，但默认作为两个独立科学问题推进。

新增 idea 时：

1. 创建新的 `Ixxx_*.md`；
2. 写清最小可证伪假设；
3. 加入本索引和根目录 README；
4. 不自动与已有 idea 合并。
