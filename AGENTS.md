# ICRA2028 research collaboration

## 当前最高优先级：Stage 3A 真机受控闭环

2026-10-07 用户决定进入真机实验，并采用“先把小脑走深，再做 VLA 即插即用展示”的递进路线。

执行约束：
- 当前不并行做 VLA 联调。
- 第一真机主任务：Anti-Tilt / eccentric disturbance；第二任务：Anti-Slip。
- 主 baseline：Fixed、Scalar Reflex、MCF Reflex。
- 第一轮 residual 只允许修改 gripper command；不同时学习 XYZ / orientation。
- Stage 2P 的 GRU/Mamba now-casting checkpoint 输出接触相关状态，不是直接可部署的 residual controller；必须先完成受控数据采集和 matched controller development。
- GRU 可作为第一版 temporal backbone 以减少变量，但“GRU/Mamba”本身不是当前核心 claim。
- 必须测真实 sensor→actuator 闭环延迟，不得把 GPU-resident inference latency 等同于 robot reaction time。
- 必须记录 grip effort / high-grip alternative，排除“只是夹得更紧”的替代解释。
- MCF 当前只称 compact spatial contact dynamics；没有独立真值前，不写成直接 shear / torque sensing。
- Stage 3B 只有 Stage 3A 的真实闭环结果支持后才启动；VLA 是后续外部有效性与 plug-and-play 展示。

执行入口：
- `research/STAGE3A_REAL_ROBOT.md`
- `experiments/real_robot_reflex/PROTOCOL.md`
- `experiments/real_robot_reflex/EXPERIMENT_MATRIX.csv`


用户后续明确指令优先。

## 当前最高优先级：Stage 2P

2026-10-07 最新用户指令：在已完成的 Now-casting 流程上离线比较 Mamba / GRU，Scalar / MCF，T=50/100，约45k–50k参数，三种子和FP32协议不变；实测batch1携带状态延迟。当前入口 `run_stage2p_mamba.py`，说明见 `research/STAGE2P_MAMBA.md`。T100公共样本须重跑GRU，不与旧T50异样本结果直接混比。

2026-10-07 用户决定：**Stage 1 条件性通过（Conditional Pass）并封存，停止单帧 Stage 1 的局部微调，正式进入 Stage 2P（Matched Temporal Probe）。**

- 不再调 Stage 1 的阈值、单帧模型或视频小样本指标来追求更大提升；保留原始脚本、数据、模型和负结果。
- 当前任务按用户纠偏改为 **now-casting**：使用 T=1/20/50 的触觉历史重构当前 ATI `[Fx,Fy,Mx,My]`，比较 Scalar 与 MCF，骨干和参数量一致。未来力矩预测实验保留为历史失败模式分析，不再用它判断时序触觉价值。
- 历史窗口保持连续，接触活跃筛选只作用于窗口末端，不能删帧后拼接伪历史。
- 15%–30% 时序增益和 10–20 倍吞吐/显存优势是用户提出的待检验假设，先运行再判断。
- Stage 1 条件性通过是已确认的项目推进决定；已有实验数值与未验证机理分开记录。
- 当前方案见 `research/STAGE2P_NOWCAST.md`；首轮预测记录保留在 `research/STAGE2P.md`。

## 已封存的 Stage 1

当前不继续围绕完整静态 11D 反复优化。候选表示改为 Minimalist Contact Flow：

```text
s_t = [f_N, CoP_x, CoP_y, A]
delta_s_t = s_t - s_(t-1)
MCF_t = [s_t, delta_s_t]
```

四个状态量及其一阶差分全部保留时为 8D。

当前核心假设是：CoP + 差分能否在 Contact-active Subsets 上优于纯 Scalar，并与 Dense Raw tactile representation 表现相当。

该内容是待验证假设。最终结论由实验决定。

## Stage 1 归档方案

- 第一轮复用已有 LeFlexiTac / TaF 数据，不要求先采新数据。
- 重新提取 MCF。
- 不再主要使用平滑全轨迹 Action MSE。
- 筛选 Contact-active Subsets：剔除完全空载段、静态死锁且无外力扰动的平淡段。
- 保留接触 / 脱开（`abs(Δf_N) > ε_f`）、CoP 移动 / 倾覆（`norm(ΔCoP, 2) > ε_p`）、面积收缩 / 膨胀（`abs(ΔA) > ε_A`），三个条件按 OR 组合；保留事件优先，避免误删脱开瞬间及仅面积变化的片段。
- 任务 A 使用 TaF 预测外力矩变化量 `Δτ_ext`（Torque Delta）。
- 任务 B 使用 LeFlexiTac 检测接触状态跳变，检验 MCF 对突然切换（微滑移或失稳前兆）的识别能力。
- 保留任务 A 原始预期：Scalar 丢失空间偏心信息，误差必然显著偏高；MCF 以 8 维 CoP 及其差分捕捉力矩偏置，误差显著低于 Scalar，且逼近 Dense Tokens。先实验验证，再修正科学表述。
- 核心比较为 Scalar、MCF、Dense Raw tactile representation。
- Static CoP 与 Physics11D 可以作为额外消融。
- 具体阈值、预测 horizon、模型和指标在实验实现时根据数据设计并完整记录。

## 已有结果

此前完整 11D 的离线结果继续保留。它们用于说明为什么现在尝试更精简的动态表示，但不代表 MCF 已经得到验证。

## 后续

Stage 1 已条件性通过，当前进入 Stage 2P 流式时序探针；后续 Mamba / streaming memory 路线根据时序实验推进。

当前顺序：

**Static 11D 收尾 → Minimalist Contact Flow → Contact-active validation → 根据实验结果决定下一步。**

## 协作原则

- 区分论文事实、已有结果、当前假设和未来想法。
- 未运行的实验不写成结果。
- 保留负结果。
- 用户定义的核心研究目标优先。
- 新的研究判断或额外限制先作为建议讨论，确认后再写成项目约束。
- 本仓库管理研究问题与实验，不扩张成通用 infra。
