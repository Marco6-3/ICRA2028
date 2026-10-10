# ICRA2028：最小论文方向冻结稿（Research Scope Freeze）

> **状态：2026-10-10 研究范围暂时冻结；科学假设尚待检验，非论文贡献已成立。**
>
> 本文件是后续计划与 AI Agent 执行的**优先研究入口**。冻结的是第一篇 ICRA 论文的**研究问题、最小任务和验证顺序**，不是实验结果、超参数或方法优劣；后续如需扩大 scope，由研究负责人明确决定并记录修订。原 Stage 1、Stage 2P 的结果、假设、负结果和协议全部保留，不溯及改写。

## 1. 第一篇论文唯一核心问题

**Can compact spatial tactile states improve eccentric disturbance rejection over scalar tactile feedback under normal-only sensing?**

在仅有法向响应的柔性触觉阵列条件下，紧凑的空间接触状态，能否比标量触觉反馈更有效地抵抗偏心扰动，同时维持可接受的夹持代价和端到端时延？

候选论文题目（暂定，**不是定稿**）：

*Minimalist Contact Flow: Physics-Structured Tactile Feedback for Eccentric Disturbance Rejection*

**第一任务：Eccentric Disturbance / Anti-Tilt。** Anti-Slip 为第二验证任务，非第一篇论文不可缺少的并行主线。

## 2. 当前候选方法：保持简单，暂不增加架构复杂度

- 主要传感输入：法向触觉阵列的**实际测量响应**；**TaF 12×12 数据规格不等于 FlexiTac 真实传感器规格**，真机网格、坐标系、单位、标定、帧率在 bring-up 时记录。
- 主表示为 **8D MCF**：`s_t=[f_N,CoP_x,CoP_y,A]`，`MCF_t=[s_t,s_t-s_{t-1}]`。目前 `f_N` 是法向强度 proxy；未经校准不以牛顿或牛顿·米表述。
- 时序骨干优先 **GRU**：基于已有 Stage 2P 的工程比较减少变量；**GRU 优于 Mamba、显式差分必要、MCF 最小充分**均不作为既定理论结论。
- 第一轮实际动作权限：`u_final=u_nominal+Δg`，**仅调整夹爪宽度/位置/力样式命令**（根据硬件安全接口确认）。不能把单自由度夹持调整称为任意方向的多轴力矩主动控制。
- `Physics-Residual` 力矩估计、直接学动作 vs 显式 wrench observer、Mamba 机制比较均为**候选消融/后续延伸**；不作为最小论文必须同时成功的创新点。
- **现有 Stage 2P GRU/Mamba checkpoint 是 `[Fx,Fy,Mx,My]` 的当前状态重构，不是已训练好的 residual controller**。真机控制器需另行开发、训练和冻结。

## 3. 物理前提：先可证伪，不预写答案

研究假设 H-Obs：在指定柔性层、接触几何、边界条件和加载协议下，正负切向扰动和偏心力矩可能经柔性接触耦合，诱发可重复、可区分的法向压力空间重分布（例如 CoP/面积变化）；这可能为闭环纠偏提供 Scalar 看不到的信息。

但**法向压力一般不能唯一决定切向力**；CoP 偏移也可能由载荷位置、姿态、材料迟滞、局部滚动等造成。符号及大小必须标定和验证，不能在没有独立真值时将 `ΔCoP` 写成剪切方向、微滑移或失稳标签。

在一致参考系且接触点近似处于 `z=h` 平面时：

```text
M_x = Σ_i (y_i f_z,i − h f_y,i)
M_y = Σ_i (h f_x,i − x_i f_z,i)
```

上述是接触力矩分解，**不是** `CoP → Fx/Fy` 一一可逆的证明。外部支撑力矩、接触点高度变化、装配偏置与参考系变换会影响实际解释。仅当压力单元已力学校准、网格物理坐标已知时，`F_N·CoP` 才可作为相应法向载荷力矩分量；未标定前不得称完整力矩真值。

## 4. 三个最小实验（按顺序推进）

### E1 — 物理可观测性：Why Spatial?

**先仿真检查机制与实验设计，再真实标定。**

- 在尽量相同的法向预载下施加 `+Fx/-Fx`、不同偏心距/方向的已知扰动；同步记录 raw pressure、MCF、法向载荷、独立 F/T、物体姿态/相对位移。
- 仿真应改变柔性层刚度/厚度、摩擦、预紧与加载约束，检查可观测性的适用范围。**禁止手工规定“切向符号 → CoP 符号”再用模型证明该关系**；若仿真接触模型不包含柔性层的法向-切向耦合，只能得出该模型的边界，不能用作 FlexiTac 机理证据。
- 真机优先验证传感器坐标、漂移、迟滞、重复性、信噪比和扰动方向可区分性。按 trial / 独立 session 分组，留出反向扰动及未见载荷条件。
- 报告正负扰动分类、方向混淆、与独立 F/T 的关系、置信区间和失败条件；不要只展示挑选的 CoP 曲线。

### E2 — 表征与时序：Why MCF + Memory?

固定样本、目标、数据划分和时序模型族；比较：

1. Scalar + matched history（基础线）；
2. Static MCF / no memory；
3. **MCF 8D + GRU history**（主方法）；
4. Raw/Dense spatial encoder（合理 CNN/MLP）+ matched temporal backbone（强参照）。

补充消融：`s_t` vs `[s_t,Δs_t]`、`T=1` vs `T>1`、CoP/面积去除；Physics-Residual 仅在独立真值、坐标标定和物理模型有效时尝试。对比要控制时序长度、训练数据、调参与参数/推理预算，并单独报告编码成本和时序成本。

**注意：**当历史帧全部可见时，`Δs_t` 不增加信息论上的独立观测，只可能改善学习的归纳偏置；不预设“时序一阶必要”或“8D 最小充分”。比较分布内误差、独立扰动方向与跨载荷/材料条件下的泛化，而非只看训练分数。

### E3 — 闭环偏心抗扰：Does It Matter for Control?

先在经 E1 检查的仿真环境进行受控 pilot / residual controller 研发，再进入 Stage 3A 真机；**仿真不替代真实 FlexiTac 机理与闭环验证**。

- 相同名义轨迹、物体、夹爪、可用动作权限、观测率、控制频率及安全约束，对比 **Fixed/Nominal、Scalar Reflex、MCF Reflex**；在主结果可信后加入合理 **Dense Reflex** 和固定高握力（high-grip）对照。
- 可以额外报告仿真 privileged-wrench `Oracle` 作为参考；**不自动称其为理论性能上界**（还取决于控制器类别和优化）。
- 主指标：最大倾角、最大相对滑移、恢复时间、存活/掉落率、峰值/积分夹持努力、物体损伤/饱和以及真实 **sensor→actuator** 延迟。按独立 trial 报告，不以帧数量冒充独立重复次数。
- 传感观测频率与执行器命令频率独立记录；若命令更快，标明 ZOH/异步更新/过期帧处理。GPU-resident 单步推理小于 1 ms **不等于**真实闭环亚毫秒。
- 残差限幅、变化率限制、无接触门控、watchdog 和急停必须先完成硬件安全检查。

## 5. 最小论文贡献：只保留两项**候选**，按结果决定

1. **Contact-State Representation / Observability**：提出并定量研究法向阵列上紧凑物理结构化空间-时间状态，明确可观测性适用条件与信息丢失边界。
2. **Disturbance-Reactive Control**：在匹配动作和算力条件下验证它是否在真实偏心扰动中提供相对 Scalar 的闭环增量，同时核查握力与延迟代价。

跨材质/摩擦 OOD、Physics-Residual 优势、Explicit Wrench Observer 优于 Direct Policy、Mamba/GRU 非光滑机理等，只有独立证据出现后才考虑升格为论文贡献。

**禁止提前写成事实的 claim：**“法向阵列必然可辨认正负剪切”“MCF 不可替代/最小充分”“虚拟六维真实传感器”“首个高频触觉反射”“首个 GRU-Mamba 机理对决”“无需标定的牛顿/牛米”“已达亚毫秒端到端闭环”“已实现多轴力矩主动补偿”。

## 6. 项目现状与 Go / No-Go

- **已完成并保留**：Stage 1 的 11D/MCF 离线诊断；Stage 2P 的当前 ATI `[Fx,Fy,Mx,My]` Now-casting，以及 GRU/Mamba 对比。这些并未独立证明本文件的 E1/E3。
- **下一个实验入口**：E1 仿真可观测性和受控加载设计；之后 FlexiTac 真机标定。与 [Stage 3A](STAGE3A_REAL_ROBOT.md) 和 [真机协议](../experiments/real_robot_reflex/PROTOCOL.md) 协同，不推翻现有步骤。
- E1 若无法在真实传感器上区分关心的扰动方向，必须如实记录观测边界，不靠加深 GRU “补出”不可观测信息；评估改用真实可观测量或修订传感/接触设计。
- E2 若 MCF 不优于 Scalar 或 Dense，也不预设成功；分别检验任务是否需要空间信息、编码成本是否构成真正收益。
- E3 若提高握力即可同样奏效，或 MCF 只降低离线 RMSE 而不改善独立闭环 trial，不宣称已证明控制贡献。
- 正式定量阈值、物理加载范围、真机频率、试验次数、训练监督/奖励和主统计指标，**在 pilot 后、正式冻结测试集前写入协议并锁定**；不凭空填写。
- 首次 pilot 后可按实测修正科学解释，保留早期假设、实验和负结果，禁止事后重写为预先预测。

## 7. 明确暂不开展的工作

本轮**不并行**：VLA / π0.5 系统集成、复杂 Diffusion 控制器改造、大规模 Mamba 调参/新模型搜索、灵巧手多自由度力矩补偿、统一世界模型、未经真机验证的复杂剪切逆解。

需要进一步工作的优先顺序：**E1（仿真→真机标定）→ E2（表征与记忆对照）→ E3（仿真 pilot→真机闭环）→ 依据实验证据再评估扩展**。

## 8. 参考入口

- [现有离线结果](OFFLINE_RESULTS.md)
- [Stage 2P Now-casting](STAGE2P_NOWCAST.md)
- [GRU/Mamba 对照](STAGE2P_MAMBA.md)
- [Stage 3A 真机闭环计划](STAGE3A_REAL_ROBOT.md)
- [完整基线定位](BASELINES.md)
- [Real-Robot Protocol](../experiments/real_robot_reflex/PROTOCOL.md)

**修订原则：**这是可撤销的研究范围冻结，而非科学真理冻结。除用户明确批准的范围修改外，新想法先进入候选/讨论，不静默扩展第一篇论文 scope。
