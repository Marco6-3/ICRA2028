# ICRA2028 · Minimalist Contact Flow 与流式触觉记忆

更新：2026-10-07。状态：Stage 1 已根据离线负结果重新划定胜利边界；不再死磕完整静态 11D，下一步优先复用现有 LeFlexiTac / TaF 数据验证 Minimalist Contact Flow。

## 当前科学判断

此前结果已经足够支持一次方向收缩，而不是继续给 11D 调参：

- LeFlexiTac 平滑动作预测中，Physics11D 没有显示稳定增量。
- TaF 独立 ATI 标签上，Scalar+centroid 已经达到或超过完整 11D；二阶矩/方向没有显示可靠额外收益。
- 窗口内变化诊断中，完整 11D 没有证明动态追踪能力。

这些负结果保留，不追溯性改写。它们意味着：**完整 11D 降级为历史 ablation / failure boundary；高阶静态空间矩不再是默认贡献。**

详见 [离线成果与决策](research/OFFLINE_RESULTS.md)。

## 当前 Stage 1：Minimalist Contact Flow

默认紧凑表示改为：

```text
z_t  = [f_N, CoP_x, CoP_y, A]
Δz_t = z_t - z_{t-1}
MCF_t = [z_t, Δz_t] ∈ R^8
```

即 **4D contact state + 4D causal first difference = 8D**。不再笼统写“6–8D”；只有后续消融真的删除某些差分项时才改变维数。

核心假设变为：

> **在 contact-active 时段，一阶空间位置（CoP）及其动态演化，可能比高阶静态空间矩更接近真正有用的紧凑触觉状态。**

这仍是待验证假设，不是既定结论。

完整协议见 [Minimalist Contact Flow](research/MINIMALIST_CONTACT_FLOW.md)。

## Stage 1 不再看什么

不再把“全轨迹、平滑开环 Action MSE”作为主要胜负指标。触觉在大量平稳轨迹中本来可能冗余，容易稀释真正的接触事件。

第一轮直接复用现有数据，构造 **Contact-active Subsets**，优先验证：

1. **Slip onset / contact-instability precursor**：只有存在独立 slip / object-motion / tangential-force 真值时才称 slip；否则称 contact-instability / redistribution event。
2. **Torque delta / eccentric-load change**：优先使用 TaF 同步 ATI 独立力矩信号，预测短 horizon 的力矩变化。

事件筛选阈值只允许从 training split 冻结；由 tactile 自己生成的事件标签不能再作为独立真值证明自己。

## Stage 1 固定比较

第一轮固定：

| 表示 | 角色 |
| --- | --- |
| Scalar `[f_N, Δf_N]` | 低维 force baseline |
| Static first-order state `[f_N, CoP_x, CoP_y, A]` | 检查空间位置本身 |
| **Minimalist Contact Flow 8D** | 当前主方法候选 |
| Dense Raw / matched tiny encoder | 高维信息参照 |
| Physics11D | 仅历史 ablation，不再要求获胜 |

所有比较固定 split、标签、因果可见范围和 readout capacity。

## Stage 1 新的通过标志

Stage 1 通过不再等价于“11D 显著最好”。需要：

- MCF 在至少一个**独立标注的 contact-active target** 上稳定优于 Scalar；
- 增量可归因于 CoP / dynamic terms，而不是未来泄漏、阈值或更大模型；
- MCF 相对 Dense Raw 达到**事先冻结的 non-inferiority 标准**，同时显著更紧凑/低成本。

若 dense comparison 证据不足，只能宣布“相对 Scalar 的增量成立”，不能提前写“与 Dense Raw 相当”。

理想情况下，论文第一观点才可以写成：

> *For contact-active prediction, first-order spatial contact location and its causal evolution preserve most of the useful compact signal, while higher-order static moments add little under the tested conditions.*

## 长期主线

新的证据链是：

**Dense tactile field → Minimalist Contact Flow → Streaming contact memory → Closed-loop recovery**

Stage 2 才研究 current / short-window MCF 与 TacMamba/Mamba-style streaming memory；Stage 3 再回答这种状态与记忆是否真的改善接触扰动后的闭环 recovery。

TacMamba = long tactile memory baseline；LeFlexiTac = dense tactile baseline；RDP = fast tactile-reactive control baseline。T-Rex 与 TacForcing继续作为多速率与 execution-time tactile conditioning 的 supporting work。

## 现在先做什么

1. **不采新数据**：先用已有 LeFlexiTac / TaF split 重提取 8D MCF。
2. 用 training split 冻结 contact-active window 定义。
3. 优先做 TaF torque-delta / eccentric-load-change，因为它有独立 ATI 信号。
4. 若数据确有独立 slip/object-motion 标签，再做 slip-onset；没有就不要制造“滑移真值”。
5. 固定 Scalar / Static / MCF / Dense Raw 四组 matched readout；Physics11D 只作历史消融。
6. test 前冻结 dense non-inferiority margin、预测 horizon、主指标和统计单位。
7. 只有 Stage 1 通过后，才把 MCF 接入 streaming temporal model。

## 导航

| 文档 | 用途 |
| --- | --- |
| [Minimalist Contact Flow](research/MINIMALIST_CONTACT_FLOW.md) | **当前 Stage 1 主协议、通过条件与失败分支** |
| [离线成果与决策](research/OFFLINE_RESULTS.md) | 11D 负结果与历史证据 |
| [研究主线](research/CORE.md) | representation → memory → closed-loop 总问题 |
| [实验方案](research/EXPERIMENTS.md) | 长期公平比较与因果纪律 |
| [Core Baselines](research/BASELINES.md) | TacMamba / LeFlexiTac / RDP |
| [后续验证路线](research/VALIDATION_ROUTES.md) | 备选消融，不得覆盖当前 Stage 1 主协议 |
| [研究路线](ROADMAP.md) | 2027 投稿推进顺序 |
| [协作规则](AGENTS.md) | AI/Agent 执行时必须遵守的证据纪律 |

**当前仓库的首要任务不是证明某个预设 descriptor，而是找出最小、因果、对 contact-active prediction 真正有用的 tactile stream。**
