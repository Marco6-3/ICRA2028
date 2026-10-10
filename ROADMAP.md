# ICRA2028 研究路线

> **2026-10-10 当前范围冻结**：以 [最小论文方向冻结稿](research/ICRA2028_MINIMAL_PAPER_FREEZE.md) 为最新执行准则。优先 **E1 仿真物理可观测性检查 / 正负切向与偏心加载 → FlexiTac 实测标定 → E2 Scalar/MCF/Dense 与历史消融 → E3 仿真 pilot 和 Stage 3A 真机抗倾覆闭环**。下文 2026-10-07 的“直接进入 Stage 3A”作为先前规划保留；现将仿真可观测性与真机标定设为受控闭环的前置验证。Anti-Slip、Physics-Residual、Mamba 机理和 VLA 暂不作为并行主线。


## 当前最高优先级：Stage 3A 真机闭环

离线 Stage 2P 已完成当前 GRU / Mamba 对照。下一步不并行联调 VLA，而是进入受控真机闭环：

**Stage 3A-0 bring-up → Stage 3A-1 受控扰动采集 → Stage 3A-2 matched Scalar / MCF residual controller → Stage 3A-3 冻结闭环评测。**

优先任务为 **Anti-Tilt / eccentric disturbance**，其次为 **Anti-Slip / pull disturbance**。主 baseline 为 Fixed、Scalar Reflex、MCF Reflex。第一轮 residual 只控制夹爪，不同时控制末端 XYZ / 姿态。

现有 Stage 2P 模型用于时序诊断，不能直接当作已训练好的控制器。只有 Stage 3A 给出可信闭环增量后，才进入 Stage 3B，将同一 tactile reflex 接到 scripted / BC / Diffusion / VLA nominal policy 上做 plug-and-play 展示。

详见 [research/STAGE3A_REAL_ROBOT.md](research/STAGE3A_REAL_ROBOT.md)。


更新：2026-10-07。

**已完成的 Stage 2P 记录：Mamba 离线验证。** 在当前 Now-casting 管线上进行两层 Mamba 与 GRU 的 T=50/100 对照，保留 Scalar / MCF、FP32和种子协议，验证侧向力矩RMSE与batch1延迟。见 [Mamba实验](research/STAGE2P_MAMBA.md)。

**历史推进记录：Stage 1 Conditional Pass → 封存 → Stage 2P Now-casting。** 首轮未来预测实验保留；当前按用户纠偏，采用同参数量两层 GRU，以 T=1/20/50 触觉历史重构当前 ATI 剪切力和侧向力矩，比较 Scalar / MCF，并比较各自长历史与 T=1。详见 [当前重构实验](research/STAGE2P_NOWCAST.md)。不回调 Stage 1。下文保留原始阶段安排，以上最新决策优先。

## 已完成 / 归档阶段

长期关注触觉表示、时序信息与接触丰富操作。此前阶段顺序为：

**Static 11D 收尾 → Minimalist Contact Flow → Contact-active validation → Stage 2P temporal probe**

这些阶段保留作为当前 Stage 3A 的离线依据与历史记录；不再作为当前执行入口。

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
