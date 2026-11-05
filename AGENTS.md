# ICRA2028 research collaboration

## 2026-11-05 当前优先级：UMI + FlexiTac 数据与真实触觉闭环（暂不开展 RL）

> **用户最新决定优先于下方 2026-11-03 的历史执行顺序。** 当前研究应先完成 UMI + 双指 FlexiTac 的传感集成、固定 UMI 可控载荷/跨 Session 数据实验、Raw/Scalar/MCF/BPI/CMQ 匹配学习评估，以及真实机械臂 Anti-Tilt / Anti-Slip 闭环。**本轮不启动 PPO/SAC/在线残差 RL，也不接 VLA。** 不删除已有 Stage 4 / V2.2 的代码、代理 RL 结果和二次审计。
>
> 当前可执行依据：[UMI + FlexiTac 研究计划](research/REAL_ROBOT_UMI_FLEXITAC_PLAN_20261105.md) + [真机执行协议](experiments/real_robot_reflex/PROTOCOL.md)。旧 `AGENTS` 中“先 PPO/SAC → 再 VLM → 再真机”的段落视作 **2026-11-03 的历史计划**，不再是当前任务依赖。
>
> 执行原则：
> - 真实法向阵列的空间变化须先由受控独立载荷/位姿测量验证，不预设可唯一估计切向力；
> - UMI 自由手持示教与固定加载数据分开管理；自由示教不自动具有真实外力标签或机器人最优纠偏动作；
> - 真实 Raw Taxels 基线必须使用全阵列而非历史 `Raw_State4_16D` 摘要；
> - 持续加载、反转后的瞬时正确率、长期漂移是与静默稳态同等重要的验证目标；正视 CMQ 现有反转后负结果；
> - 所有测试须独立 Session/trial 分组，且训练集外的标准化、重调参、阈值选择不可污染最终测试；
> - 优先简单安全的 1D 闭环，然后按硬件安全与公平要求扩展 3-DoF；记录夹持代价、独立真值与端到端延迟。

---

## 2026-11-03 研究范围升级与最新执行准则（Stage 3S–V2.1，当前最高优先级）

优先阅读 [research/ICRA2028_MINIMAL_PAPER_FREEZE.md](research/ICRA2028_MINIMAL_PAPER_FREEZE.md)、[research/STAGE3S_V2_1_BPI_MCF_3DOF_REFLEX_20261103.md](research/STAGE3S_V2_1_BPI_MCF_3DOF_REFLEX_20261103.md) 与 [experiments/v2_1_bpi_mcf_3dof_reflex/README.md](experiments/v2_1_bpi_mcf_3dof_reflex/README.md)。旧版 Stage 1、Stage 2P、Stage 3S（`stage3s_isaac` 与 `stage3s_alignment`）的全部数据、源码、哈希与负结果原样保留，严禁覆写。

- **主表征升级**：由滑动窗口 8D `MCF` 升级为 **32D 参考帧锁定双指预紧不变流形接触场（`BPI-MCF`）**（初始接触帧锁定 $k_{\text{base}}=5$ + 单指预紧归一化 $f_{i,0}$ + 双指对称/反对称正交分解 + 空间偏心力矩偶极），保持 `extract_bpi_mcf` 特征算子冻结。
- **多模态防坍缩自监督**：允许引入视觉位姿/场景输入，但必须配备 **跨模态接触力螺旋与剪切方向自监督（`aux_dir_head + aux_wrench_head + vis_wrench_head`）+ 计划性视觉模态丢弃**，并通过单步反向传播梯度审计（`all_positive = True`）与公平冻结隐空间线性探针（`FrozenProbeDirAcc`）核验。
- **动作权限与因果门控升级**：解除早期“仅允许 1D $\Delta g$”限制，升级为 **因果偶极新息能量门控 3-DoF 空间反射策略（$[\Delta x_{\text{wrist}}, \Delta y_{\text{wrist}}, \Delta g_{\text{grip}}]$）**。闭环评估必须严格区分 **0% `Stage 3S` 混杂的纯 V2.1 策略** 与混合策略，并同时报告受载滑移、倾角与预紧静默段抖动（`PreloadJitter`）。
- **后续系统推进顺序**：**Step 1 Isaac Lab Warm-Start 残差强化学习（Residual PPO/SAC）$\to$ Step 2 2×A100 慢频 VLM（`Qwen2.5-VL`）微调与 DGX Spark 128GB 双频推理部署 $\to$ Step 3 真机 Sim-to-Real 轻量标定与受控闭环验证**。

---

## 历史记录：2026-10-10 第一篇论文范围冻结（已于 2026-11-03 修订升级）

优先阅读 [research/ICRA2028_MINIMAL_PAPER_FREEZE.md](research/ICRA2028_MINIMAL_PAPER_FREEZE.md)。它覆盖下方历史性的阶段优先级描述，但**不改写** Stage 1/2P 已保存的假设、实验结果和负结果。

- 唯一主问题：**法向阵列的紧凑空间触觉状态能否相对 Scalar 改善偏心扰动闭环稳定性？** 首要任务 Anti-Tilt。
- 按 **E1 物理可观测性（先仿真检查、后真机标定）→ E2 matched representation × memory → E3 仿真控制 pilot / 真机闭环** 推进。现有 Stage 3A 仍为真机执行协议，但在闭环前先补可观测性验证。
- 保持 MCF 8D + GRU 为主要候选；Physics-Residual、Direct Policy vs Observer、GRU-Mamba 机制、Anti-Slip、VLA 均不是此阶段必须成立的主贡献，不擅自扩大 scope。
- **不能**把 CoP 与剪切方向的映射、8D 充分性、亚毫秒物理闭环、六维真实力感知或多轴独立力矩补偿预先写成结论。除非用户明确要求，不自动把待证假设收缩或改写为负结论；所有结论由真实实验决定。
- 第一轮夹爪 residual 仅 `Δg`；严格保持动作权限、频率、传感器信息、握力代价和数据划分公平。仿真不得硬编码期望的剪切/CoP 符号关系再验证自身假设。
- 用户明确批准后才改变冻结范围；科学假设、模型优劣、任务细节仍可由实验推动修订，记录原因并保留完整历史。


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

## 已完成的 Stage 2P 历史执行规则

2026-10-07 已执行：在 Now-casting 流程上离线比较 Mamba / GRU，Scalar / MCF，T=50/100，约45k–50k参数，三种子和FP32协议不变；实测batch1携带状态延迟。当前入口 `run_stage2p_mamba.py`，说明见 `research/STAGE2P_MAMBA.md`。T100公共样本须重跑GRU，不与旧T50异样本结果直接混比。

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
