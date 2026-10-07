# ICRA2028 研究路线

更新：2026-10-07。

**最新用户指令：先完成 Mamba 离线验证。** 在当前 Now-casting 管线上进行两层 Mamba 与 GRU 的 T=50/100 对照，保留 Scalar / MCF、FP32和种子协议，验证侧向力矩RMSE与batch1延迟。见 [Mamba实验](research/STAGE2P_MAMBA.md)。

**最新推进决定：Stage 1 Conditional Pass → 封存 → Stage 2P Now-casting。** 首轮未来预测实验保留；当前按用户纠偏，采用同参数量两层 GRU，以 T=1/20/50 触觉历史重构当前 ATI 剪切力和侧向力矩，比较 Scalar / MCF，并比较各自长历史与 T=1。详见 [当前重构实验](research/STAGE2P_NOWCAST.md)。不回调 Stage 1。下文保留原始阶段安排，以上最新决策优先。

## 当前路线

长期关注触觉表示、时序信息与接触丰富操作。

当前先完成 Stage 1：

**Static 11D 收尾 → Minimalist Contact Flow → Contact-active validation → 根据实验结果决定下一步**

## Stage 1：Minimalist Contact Flow

候选表示：

```text
s_t = [f_N, CoP_x, CoP_y, A]
delta_s_t = s_t - s_(t-1)
MCF_t = [s_t, delta_s_t]
```

第一轮直接复用已有 LeFlexiTac / TaF 数据。

不再把平滑全轨迹 Action MSE 作为主要验证目标，而是筛选 Contact-active Subsets，研究：

- 任务 A（TaF）：预测外力矩变化量 `Δτ_ext`（Torque Delta）；
- 任务 B（LeFlexiTac）：接触状态跳变检测，检验 MCF 对突然切换（微滑移或失稳前兆）的识别能力。

切片过滤剔除完全空载段和静态死锁且无外力扰动的平淡段；保留 `abs(Δf_N) > ε_f`、`norm(ΔCoP, 2) > ε_p`、`abs(ΔA) > ε_A` 中任一成立的片段。保留事件优先，避免误删脱开瞬间或仅面积变化的片段。

任务 A 预期判定：Scalar 丢失空间偏心信息，预测力矩变化误差必然显著偏高；MCF 依靠 CoP 及其差分，以 8 维捕捉力矩偏置，误差显著低于 Scalar，且逼近 Dense Tokens。保留该假设，实验验证后再修正。

核心比较：

**Scalar vs Minimalist Contact Flow vs Dense Raw tactile representation**

Static CoP、Physics11D 等可以作为额外消融。

当前希望通过实验回答：

> CoP + 差分是否显著优于纯 Scalar，并且能否与 Dense Raw Tokens 表现相当？

若实验支持，则进一步讨论“高阶空间矩是否冗余、核心是否主要来自 CoP 及其动态演化”。若不支持，则根据结果重新分析，不预先规定结论。

详细方案见 [research/MINIMALIST_CONTACT_FLOW.md](research/MINIMALIST_CONTACT_FLOW.md)。

## 后续 Stage 2

Stage 1 得到结果后，再决定如何研究 temporal memory。

可能包括把经过验证的 compact tactile representation 接入 Mamba / streaming memory，测试更长历史信息是否有额外作用。

具体架构、实验任务和 claim 等待 Stage 1 结果后确定，不在当前阶段锁死。

## Stage 3

若前面的 representation 与 temporal hypothesis 得到支持，再进入真实闭环 manipulation / recovery 实验。

具体任务同样根据前序实验结果选择。

## 时间目标

仍以 2027 年形成完整研究成果并准备 ICRA 2028 投稿为长期项目目标。内部可继续以 2027-07-31 作为完整初稿的项目管理参考日期；正式投稿要求以未来官方 CFP 为准。

## 原则

路线图描述的是当前计划，不代表结果。每一阶段完成后根据真实实验结果更新下一阶段，而不是提前把整篇论文的结论锁死。
