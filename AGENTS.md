# ICRA2028 research collaboration

用户后续明确指令优先。

## 当前 Stage 1

当前不继续围绕完整静态 11D 反复优化。候选表示改为 Minimalist Contact Flow：

```text
s_t = [f_N, CoP_x, CoP_y, A]
delta_s_t = s_t - s_(t-1)
MCF_t = [s_t, delta_s_t]
```

四个状态量及其一阶差分全部保留时为 8D。

当前核心假设是：CoP + 差分能否在 Contact-active Subsets 上优于纯 Scalar，并与 Dense Raw tactile representation 表现相当。

该内容是待验证假设。最终结论由实验决定。

## 当前实验

- 第一轮复用已有 LeFlexiTac / TaF 数据，不要求先采新数据。
- 重新提取 MCF。
- 不再主要使用平滑全轨迹 Action MSE。
- 筛选 Contact-active Subsets，包括接触突变、受力不均、CoP 移动、面积变化和方向微调。
- 验证 Slip Onset / 接触微滑移前兆与 Torque Delta / 倾覆力矩变化。
- 若现有数据不足以定义 slip onset，则记录限制，并优先完成 TaF Torque Delta。
- 核心比较为 Scalar、MCF、Dense Raw tactile representation。
- Static CoP 与 Physics11D 可以作为额外消融。
- 具体阈值、预测 horizon、模型和指标在实验实现时根据数据设计并完整记录。

## 已有结果

此前完整 11D 的离线结果继续保留。它们用于说明为什么现在尝试更精简的动态表示，但不代表 MCF 已经得到验证。

## 后续

Stage 1 完成后，根据实际结果再决定是否以及如何进入 Mamba / streaming temporal memory。

当前顺序：

**Static 11D 收尾 → Minimalist Contact Flow → Contact-active validation → 根据实验结果决定下一步。**

## 协作原则

- 区分论文事实、已有结果、当前假设和未来想法。
- 未运行的实验不写成结果。
- 保留负结果。
- 用户定义的核心研究目标优先。
- 新的研究判断或额外限制先作为建议讨论，确认后再写成项目约束。
- 本仓库管理研究问题与实验，不扩张成通用 infra。
