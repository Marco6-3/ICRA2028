# Stage 1 收尾：Minimalist Contact Flow

状态：2026-10-07 用户决定条件性通过（Conditional Pass），已封存，停止单帧局部微调。当前进入 [Stage 2P](STAGE2P.md)。下文保留原始实验假设与方案。

## 目标

此前 Stage 1 把目标放在“证明完整静态 11D（包含二阶矩）显著优于标量和低阶特征”。现有离线实验没有得到理想结果，因此这一轮不继续在原有静态 11D 上死磕，而是调整特征和验证任务。

当前要验证的假设是：

> **在 contact-active 时段，CoP 及其动态演化是否能够提供比纯 Scalar 更有用的紧凑接触信息，并达到与 Dense Raw Tokens 相当的表现。**

这只是待验证假设。具体结论由实验结果决定。

## Minimalist Contact Flow

从每个 tactile frame 提取：

```text
s_t = [f_N, CoP_x, CoP_y, A]
Δs_t = s_t - s_{t-1}
MCF_t = [s_t, Δs_t]
```

其中：

- `f_N`：法向接触强度 / force proxy；
- `CoP_x, CoP_y`：接触中心；
- `A`：有效接触面积；
- `Δs_t`：一阶微差分，用于描述接触状态的动态变化。

若四个量及其差分全部保留，默认实现为 8D。后续是否删减其中某些分量，由实验决定。

## 数据

第一轮不需要采新数据。

直接复用已有 LeFlexiTac / TaF 离线数据，重新提取 Minimalist Contact Flow，并在已有数据上构造 Contact-active Subsets。

## Contact-active Subsets

不再把平滑全轨迹的开环 Action MSE 作为 Stage 1 的主要验证指标。

### 切片过滤（Filtering）

剔除完全未接触（`f_N ≈ 0`）的空载段，以及静态死锁且无外力扰动的平淡段（如 `ΔCoP ≈ 0` 且 `Δf_N ≈ 0`）。

保留以下任一变化对应的片段：

| 保留段 | 判据 |
| --- | --- |
| 接触瞬间与脱开瞬间 | `abs(Δf_N) > ε_f` |
| 受力中心移动 / 倾覆阶段 | `norm(ΔCoP, 2) > ε_p` |
| 接触面积收缩 / 膨胀 | `abs(ΔA) > ε_A` |

三个保留条件采用逻辑 OR：

```text
active_t = (abs(Δf_N,t) > ε_f)
        OR (norm(ΔCoP_t, 2) > ε_p)
        OR (abs(ΔA_t) > ε_A)
```

保留事件优先于空载 / 平淡段剔除：脱开瞬间即使当前 `f_N ≈ 0`，仍按力变化保留；即使 `ΔCoP ≈ 0` 且 `Δf_N ≈ 0`，面积变化超过阈值的片段仍保留。先在原始连续序列上计算差分，再过滤切片，避免把删段后的相邻帧误当成原始相邻帧。

目标是让 Stage 1 关注真正可能需要触觉信息的局部接触过程，而不是被大量平稳轨迹稀释。

具体筛选规则、阈值和窗口长度可以在实验实现阶段根据数据分布确定并记录。

## Stage 1 验证任务

### Task A — TaF：预测外力矩变化量 Δτ_ext（Torque Delta）

利用 TaF 同步 ATI 力/力矩信息，预测外力矩变化量：

```text
Δτ_ext,t = τ_ext,t - τ_ext,t-1
```

对比组：**Scalar vs MCF vs Dense Raw Tokens**。

**预期判定（保留原始实验假设，待实验验证）：Scalar 由于丢失空间偏心信息，预测力矩变化误差必然显著偏高；MCF 凭借 CoP 及其差分，能够以 8 维的极小体积捕捉到力矩偏置，误差显著低于 Scalar，且逼近 Dense Tokens。**

可观察指标包括：

- RMSE / MAE；
- torque change direction；
- 不同 contact-active 子集上的误差。

具体标签定义和预测 horizon 在实验实现时根据数据内容确定并记录。

### Task B — LeFlexiTac：接触状态跳变检测

利用 LeFlexiTac 离线数据，检验 MCF 是否能准确识别接触状态的突然切换（微滑移或失稳前兆）。

核心比较同样为 **Scalar vs MCF vs Dense Raw Tokens**。

可观察指标包括：

- classification accuracy；
- AUROC / AUPRC；
- event recall；
- 微滑移或失稳前兆的 prediction lead time。

具体事件标签、检测窗口和评估设置在实验实现时记录；实验结果出来后再修正判断。

## 比较对象

Stage 1 核心比较：

| Representation | 目的 |
| --- | --- |
| Scalar | 纯标量触觉基线 |
| **Minimalist Contact Flow** | 验证 CoP + 一阶差分 |
| Dense Raw Tokens / Raw tactile representation | 高维触觉参照 |

可以保留额外消融，例如 static CoP、原 Physics11D 等，用于解释实验结果，但它们不改变上述核心比较。

## Stage 1 希望验证的结果

当前希望检验：

> **CoP + 差分是否显著优于纯 Scalar，并且能否与 Dense Raw Tokens 表现相当。**

如果实验支持这一结果，Stage 1 可以形成第一核心观点：

> **高阶空间矩可能是冗余的，核心信息主要来自 CoP 及其动态演化。**

如果实验不支持，则根据实际结果重新分析表示、任务和下一步方向，不提前规定必须如何收缩或转向。

## 与 Stage 2 的接口

如果 Minimalist Contact Flow 在 Stage 1 显示出价值，下一步再研究如何把这一低维动态 tactile stream 接入 Mamba / streaming memory，验证更长历史信息是否进一步有用。

当前顺序：

**Static 11D 收尾 → Minimalist Contact Flow → Contact-active validation → 根据实验结果决定 Stage 2。**

本文件记录的是当前实验假设和计划，不代表这些结论已经成立。
