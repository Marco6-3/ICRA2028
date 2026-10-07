# Stage 1 收尾：Minimalist Contact Flow

状态：当前最高优先级离线验证。更新：2026-10-07。

## 为什么改 Stage 1

此前离线结果已经给出足够清楚的边界：完整 Physics11D（尤其二阶矩/主方向）在稳态读出中没有可靠超过更小的 Scalar+centroid，窗口内动态变化甚至没有超过静态参照。因此不再把“证明完整静态 11D 必要”作为 Stage 1 的胜利条件。

这不是删除负结果，而是把它固定为一个科学结论候选：

> 高阶空间矩在当前稳态数据/任务上增量信噪比较低；后续优先验证一阶空间位置及其动态演化是否已经捕获主要 control-relevant contact cue。

完整 11D 保留为历史 ablation / negative boundary，不再是默认方法。

## 默认表示：4D state + 4D causal delta = 8D

从每个 tactile frame 提取：

```text
z_t = [f_N, CoP_x, CoP_y, A]
Δz_t = z_t - z_{t-1}
m_t = [z_t, Δz_t] ∈ R^8
```

其中：

- `f_N`：total normal contact intensity / force proxy；没有绝对标定时不得写成真实牛顿值。
- `CoP_x, CoP_y`：归一化阵列坐标中的接触质心。
- `A`：归一化 active contact area。
- `Δ`：严格因果的一阶差分；只使用 t 与 t-1，episode 开头 reset，不跨 episode。
- 若差分噪声过高，可在训练/验证协议冻结后使用因果 EMA 或短后向 slope；必须报告额外延迟。

默认版本是 **8D**。只有经过预注册消融后删除某些差分项，才称 6D/7D 版本，避免把维数写模糊。

## 数据：先不采新数据

优先复用仓库已经使用的 LeFlexiTac / TaF 离线数据和现有 split。第一轮不得因为结果不好重新挑 source / episode。

新采真机数据属于后续 Stage 2/closed-loop 或独立确认，不是完成本轮 Stage 1 的前置条件。

## Contact-active subset

不再以全程平滑 action MSE 作为 Stage 1 主指标。先从现有序列构造 **contact-active windows**，只研究发生接触变化的局部时段。

候选事件分数仅用于筛选窗口，不作为最终真值：

```text
E_t = robust_norm(|Δf_N|) +
      robust_norm(||ΔCoP||_2) +
      robust_norm(|ΔA|)
```

阈值必须只由 training split 确定，然后冻结到 validation/test。窗口建议覆盖事件前若干帧与事件后若干帧，确保可以测试“前兆”而不是只识别已经发生的事件。

禁止把由 tactile feature 自己阈值生成的事件标签再作为“证明该 feature 有预测力”的独立 ground truth。

## 两个 Stage 1 任务

### Task A — Slip-onset / contact-instability precursor

只有数据中存在独立 slip / object-motion / tangential-force / pose 标签时才使用“slip onset”名称。否则统一称 **contact-instability / redistribution event prediction**。

目标是在事件真正发生前的固定 horizon 预测二分类或 time-to-event。

主指标：
- AUROC + AUPRC；
- event-level recall at fixed false-positive rate；
- lead time（若标签时间精度允许）。

### Task B — Torque-delta / eccentric-load change

优先在 TaF 使用同步 ATI 独立力/力矩作为标签，预测下一短 horizon 的 torque delta / normalized torque change，而不是再次拟合平滑 operator action。

主指标：
- source-level RMSE / MAE；
- direction/sign accuracy（若物理上有意义）；
- 按完整 source/trial bootstrap 的 paired uncertainty。

标签定义、坐标系与单位必须按数据实际元数据记录；不能把未核实的 ATI 比值直接叫毫米 CoP。

## 必须比较的表示

第一轮固定四组：

| 表示 | 目的 |
| --- | --- |
| Scalar: `[f_N, Δf_N]` | TacMamba-style low-dimensional force baseline |
| CoP-static: `[f_N, CoP_x, CoP_y, A]` | 判断空间一阶量本身的价值 |
| **Minimalist Contact Flow (8D)** | 判断“CoP + 动态演化”的核心假设 |
| Dense Raw / matched tiny encoder | 信息上界/高维参照 |

Physics11D 只作为历史 ablation 加入，不再承担“必须获胜”的主线角色。

所有比较使用相同 split、相同标签、相同 causal visibility、相同 readout capacity；先用 linear / tiny MLP 或 matched tiny encoder，不做大规模 architecture search。

## Stage 1 的通过条件

Stage 1 不再要求完整 11D 优于所有表示。新的通过条件是：

1. 在至少一个**独立标注的 contact-active target**上，Minimalist Contact Flow 相对 Scalar 有可重复、source/trial-level 的增量；
2. 该增量主要来自 CoP / dynamic terms，而不是测试集阈值、未来信息或模型容量；
3. 相对 Dense Raw，8D 表示在主指标上达到预注册的“近似相当”标准，同时显著更紧凑/低成本。

“表现相当”必须在运行 test 前冻结 non-inferiority margin；不得看完结果后再定义。若现有数据不足以支持严格 dense comparison，则只允许写“Scalar superiority established; dense equivalence unresolved”。

### 理想结论

若以上成立，可形成第一核心观点：

> **For contact-active prediction, first-order spatial contact location and its causal evolution carry most of the useful compact signal; higher-order static spatial moments add little under the tested conditions.**

注意这是待验证 claim，不预先写成论文结论。

## 失败也有明确去向

- MCF ≈ Scalar：说明现有任务/标签不需要空间信息；停止包装 CoP 为贡献，转向真正 spatially ambiguous 的受控任务。
- MCF > Scalar，但明显 < Dense Raw：compact flow 有价值但信息不充分；进入 compact + sparse/event-triggered rich tactile 的 hybrid 路线。
- MCF ≈ Dense Raw，且 > Scalar：Stage 1 通过，进入 temporal memory / Mamba，研究更长 history 是否进一步提高 early prediction 与闭环纠偏。
- Physics11D > MCF：重新检查究竟是哪一个高阶项贡献增量，只恢复被证明有用的项，不恢复“完整 11D”故事。

## 与 Stage 2 的接口

Stage 1 输出不再是固定 11D，而是一个经过验证的 compact stream：

```text
m_1, m_2, ..., m_t
```

Stage 2 再回答：current MCF、short-window MCF 与 streaming memory（TacMamba/Mamba-style）相比，谁能更早、更稳地识别接触状态变化，并最终改善闭环 recovery。

因此论文主线从“静态 11D 压缩”改成：

**Dense tactile field → Minimalist Contact Flow → Streaming contact memory → Closed-loop recovery.**
