# ICRA2028 research collaboration

用户目标：围绕 compact tactile representation、streaming memory 与 contact-rich closed-loop recovery，在 2027 年形成可投稿 ICRA 2028 的证据链。用户后续明确指令优先。

## 当前主线（2026-10-07）

**Dense tactile field → Minimalist Contact Flow → Streaming contact memory → Closed-loop recovery**

Stage 1 已根据离线负结果 pivot。不要再把“证明完整静态 Physics11D 必要”当作当前目标。

默认 Minimalist Contact Flow：

```text
z_t = [f_N, CoP_x, CoP_y, A]
MCF_t = [z_t, z_t-z_{t-1}] ∈ R^8
```

完整 11D（含二阶矩/主方向）保留为历史 ablation / negative boundary。除非新实验明确证明某个高阶项有独立增量，否则不得恢复“完整 11D 是核心方法”的叙事。

## Stage 1 执行规则

- 第一轮**不要求新采数据**；优先复用已有 LeFlexiTac / TaF 固定 split。
- 不再以全轨迹平滑 action MSE 作为主要胜负指标。
- 主分析聚焦 **contact-active windows**。
- 优先任务：
  1. TaF 独立 ATI 标签上的 torque-delta / eccentric-load-change prediction；
  2. contact-instability / redistribution event prediction；
  3. 只有存在独立 slip / object-motion / tangential-force 真值时才称 slip-onset prediction。
- 禁止用 tactile feature 自己阈值生成标签，再把该标签作为“该 tactile feature 有预测力”的独立证据。
- contact-active 筛选阈值、预测 horizon、filter、normalization 只能用 training/validation 冻结；test 不得参与。
- 默认固定比较：
  - Scalar = [f_N, Δf_N]
  - Static first-order = [f_N, CoP_x, CoP_y, A]
  - MCF 8D
  - Dense Raw / matched tiny encoder
  - Physics11D 仅作历史 ablation
- 比较必须固定 split、标签、causal visibility、readout capacity、训练预算与统计单位。
- 第一轮优先 linear / tiny MLP / matched tiny encoder；不要用大模型或 architecture search 把表示问题变成调参竞赛。
- 与 Dense Raw “相当”必须在 test 前冻结 non-inferiority margin。没有冻结 margin 时只能报告差距，不能事后宣布 equivalent。

## Stage 1 通过/失败判读

Stage 1 通过需要：
1. MCF 在至少一个独立标注 contact-active target 上可靠优于 Scalar；
2. 消融支持收益来自 CoP / causal dynamics；
3. 相对 Dense Raw 达到预注册 non-inferiority，或明确标注 dense equivalence unresolved；
4. compactness / latency / compute 优势有实际测量。

分支：
- MCF ≈ Scalar → 停止在当前数据刷 descriptor，转真正 spatially ambiguous 的受控任务。
- MCF > Scalar but < Dense Raw → hybrid compact flow + sparse/event-triggered rich tactile。
- MCF ≈ Dense Raw and > Scalar → Stage 1 通过，进入 streaming memory。
- Physics11D > MCF → 只恢复被消融证明有效的高阶项，不恢复完整 11D 故事。

## Stage 2 规则

Stage 2 研究 current / short-window / streaming memory 的增量，而不是“证明 Mamba”。

- 至少保留简单 causal memory baseline（如 GRU/TCN）与 TacMamba/Mamba-style streaming model 的公平比较。
- 固定 history visibility、parameter/hidden budget、训练数据、decision rate 和 latency accounting。
- 重点指标包括 early prediction / lead time、history-dependent ambiguity resolution、source/trial-level uncertainty。
- 若 short window 已达到 streaming memory，接受结果并缩小论文 claim，不为了 Mamba 更换任务或无限调参。

## Stage 3 规则

若要声称 tactile-reactive manipulation 改善，必须进入真实闭环并报告 physical event → action effect 的端到端 P50/P95 latency，而不是只报 descriptor/network forward。

优先闭环问题：eccentric-load/contact redistribution recovery、grasp disturbance recovery、history-dependent insertion/search。

## 证据纪律

- 论文事实、作者报告、项目推断、待验证假设和本项目结果分开记录。
- 未运行不能声称复现；未确认不能声称 first/SOTA/必然录用。
- CoP 漂移只能默认称 contact redistribution cue；没有独立真值不能直接称 slip。
- 未绝对标定的 taxel sum 称 contact intensity / force proxy，不冒称 N/Pa。
- preprocessing 必须严格因果；episode reset；不得使用 test future statistics。
- 数据划分按完整 source/trial/episode，避免随机帧泄漏。
- 负结果必须保留；不得追溯性修改原假设让实验看似“成功”。
- 新实验必须记录：数据版本、split、标签来源、预测 horizon、filter/threshold、模型预算、seed、统计单位、代码提交、全部失败与排除。
- 本仓库管理研究问题和证据，不扩张成通用 infra。

## 核心文献角色

- TacMamba：long tactile streaming memory baseline。
- LeFlexiTac：dense FlexiTac representation / policy interface baseline。
- RDP：fast tactile-reactive closed-loop baseline。
- T-Rex：多速率 temporal/spatial tactile design reference。
- TacForcing：execution-time tactile conditioning reference。

不要把“FlexiTac + Mamba”或“physics + Mamba”本身当 novelty。真正的 novelty 必须由经过验证的 representation/memory/control gap 决定。

当前 Stage 1 的权威协议为 [research/MINIMALIST_CONTACT_FLOW.md](research/MINIMALIST_CONTACT_FLOW.md)。若其它旧文档仍保留 11D-first 表述，以该文件与最新 README 为准，直到旧文档完成归档/同步。
