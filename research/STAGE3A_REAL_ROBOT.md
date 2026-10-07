# Stage 3A：真机受控触觉反射实验

状态：下一阶段实验协议（待执行）。更新：2026-10-07。

## 1. 当前目标

Stage 1 与 Stage 2P 的离线实验已经完成阶段性验证与归档。下一步不直接接 VLA，而是先在真实机械臂/夹爪上回答一个可否证问题：

> **Does a high-frequency compact tactile residual controller improve disturbance rejection over fixed-force and scalar tactile feedback?**

中文表述：

> 在相同 nominal controller、相同硬件和扰动条件下，高频 MCF 时序触觉反馈是否能够相对固定握持与标量触觉反馈，减少滑移、倾覆或掉落，并保持合理的夹持 effort 与端到端延迟？

这是待验证假设，不预先规定实验结果。

## 2. 与 Stage 2P 的关系

Stage 2P 的 GRU / Mamba 模型当前输出的是 ATI 当前剪切力和侧向力矩重构，不是可直接部署的 gripper residual controller。

因此禁止把现有 now-casting checkpoint 直接解释为“已经训练好的小脑控制器”。

Stage 3A 必须显式拆成：

1. **硬件与扰动采集**：建立可重复的真机接触与扰动数据；
2. **Residual controller 构造 / 训练**：Scalar 与 MCF 使用匹配的控制接口、动作边界和训练预算；
3. **冻结 controller 后闭环评测**：在独立 trial 上测物理响应。

GRU 当前作为第一版 temporal backbone 的工程候选，原因是本仓库 Stage 2P 中同预算对照下它比当前 Mamba 配置更低 RMSE、更低 CUDA Graph 延迟。该事实只用于减少真机变量，不把 GRU 本身写成最终 novelty。

## 3. 控制接口

所有方法共享同一个 nominal command：

```text
u_final(t) = u_nominal(t) + Δu_reflex(t)
```

第一轮只允许 residual 修改夹爪控制量：

```text
Δu_reflex(t) = Δg_t
```

其中 `Δg_t` 可以对应平台可安全支持的 gripper position / width / force-like command，具体单位在硬件 bring-up 后冻结。

第一轮不同时学习或控制 XYZ、末端姿态和夹爪，以免失败原因不可归因。

所有 residual 必须具备：
- amplitude clamp；
- slew-rate clamp；
- emergency stop；
- no-contact gate；
- watchdog / stale-frame handling。

## 4. 第一轮 baseline

固定三组主比较：

| 方法 | 触觉输入 | 控制形式 |
| --- | --- | --- |
| Fixed / Nominal | 无 tactile residual | 只执行冻结的 nominal grip |
| Scalar Reflex | `f_N` 及其必要的因果历史 / 差分 | 有界 `Δg_t` |
| MCF Reflex | `[f_N, CoP_x, CoP_y, A]` 及一阶差分与连续历史 | 有界 `Δg_t` |

Scalar 与 MCF 必须：
- 使用相同控制频率；
- 使用相同可见历史长度；
- 使用匹配的 temporal backbone / 参数预算或显式记录差异；
- 使用相同动作 clamp；
- 使用相同训练数据、优化预算与选模规则；
- 不允许 MCF 额外看到外部 F/T、视觉、未来帧或测试条件标签。

Dense / raw tactile policy 暂不作为第一轮必须 baseline；只有在 Fixed / Scalar / MCF 的闭环机制结果可信后，再作为 richer tactile reference 加入。

## 5. 优先实验：Anti-Tilt / Eccentric Disturbance

### 5.1 为什么优先

仅垂直抗拔可能主要由 total normal load 解释，Scalar baseline 可能已经足够。偏心扰动更直接测试空间接触状态是否带来闭环增量。

### 5.2 任务

夹持刚性细长物体或适合安全挂载偏心载荷的 test object。机械臂保持固定姿态或执行冻结的低速 scripted motion。

使用可重复装置在物体不同 lateral offset `r` 位置施加相同或标定后的外载 `F`：

```text
τ_ext ≈ F · r
```

改变 offset / direction，同时尽可能控制总载荷水平，使部分条件下 total-load cue 接近，而空间接触重分布不同。

### 5.3 待观察量

- CoP trajectory；
- active area trajectory；
- total load proxy；
- object tilt angle；
- slip / relative displacement；
- gripper residual command；
- actuator response。

本实验检验的是 **compact spatial contact dynamics** 的控制价值。

当前 MCF 不直接测量真实剪切力或真实外力矩；在有独立 F/T 真值前，不将 MCF 描述为“直接 shear / torque sensing”。

## 6. 第二实验：Anti-Slip / Pull Disturbance

夹爪夹持圆柱或方块，使用滑轮、砝码、弹簧加载器或其他可重复装置产生近似 step / ramp pull disturbance。

优先避免把“实验员手拉”作为正式主评测；手拉可用于最终 demo，但正式 trial 应尽量有可追溯的加载条件。

变量可包括：
- disturbance magnitude；
- pull direction；
- ramp / step profile；
- object surface / mass（第二阶段再扩展）。

第一轮只选择少量、可稳定复现的条件。

## 7. High-grip 替代解释

必须加入一个强工程对照或至少明确报告：

> 是否通过简单提高固定握力就能获得同等稳定性？

因此记录：
- nominal / peak grip command；
- time-integrated grip effort；
- 若有独立力传感则记录真实 grip force；
- object damage / saturation / safety limit。

MCF 的收益若存在，不能只来自“最终夹得更用力”。

## 8. 主指标

闭环 trial 至少记录：

1. **最大滑移距离** `d_max`；
2. **最大倾角** `theta_max`；
3. **恢复时间** `T_recover`；
4. **disturbance survival / drop rate**；
5. **peak grip command / grip effort**；
6. **sensor → actuator 真实闭环延迟**。

端到端时间链建议记录：

```text
t_disturb
→ t_sensor
→ t_host_receive
→ t_feature_ready
→ t_model_done
→ t_command_sent
→ t_actuator_response
```

Stage 2P 中 0.07–0.15 ms 量级的 CUDA Graph 数值仅是 GPU-resident 模型推理，不得等同于真实机器人反射时间。

## 9. 试验分层

### Stage 3A-0 — Bring-up
目标：只验证硬件、同步、FlexiTac 数据稳定性、夹爪 residual interface 与安全 clamp。

不比较模型优劣。

### Stage 3A-1 — Passive disturbance acquisition
关闭 learned residual，执行 Fixed / nominal grip，采集受控偏心扰动与抗滑扰动的完整连续序列。

用途：
- 确认扰动确实产生可重复接触变化；
- 冻结输入、标签、动作边界；
- 构造 Scalar / MCF matched controller 的训练与开发数据。

### Stage 3A-2 — Controller development
在训练 / 开发 trial 上构造 Scalar 与 MCF residual controller。

训练目标、teacher / supervision / reward 生成方式必须在读取正式测试结果前写入 protocol revision，不允许使用 test trial 反向设计 controller。

### Stage 3A-3 — Frozen closed-loop evaluation
冻结：
- feature extraction；
- history length；
- model；
- thresholds；
- action limits；
- nominal controller。

再执行独立 closed-loop trials，并按 trial-level 报告置信区间与失败案例。

## 10. Go / No-Go

只有在至少一个受控真机任务上，MCF Reflex 相对 Scalar / Fixed 出现可信、可复现的闭环增量，且不是更高 grip effort、不同 latency budget、额外传感器或测试后调参造成，才进入 Stage 3B。

如果 MCF ≈ Scalar：
- 接受结果；
- 检查任务是否真正需要空间状态；
- 不通过扩大网络或直接接 VLA 掩盖该结果。

如果 Fixed high-grip 已达到相同效果：
- 把问题转向安全 grip effort、易损物体、偏心姿态或需要姿态修正的任务；
- 不把“防掉落”本身写成充分贡献。

## 11. Stage 3B：后续 plug-and-play 展示

Stage 3B 不是当前并行任务。

只有 Stage 3A 通过后，保持 tactile reflex 的接口与权重冻结或只做预先声明的最小适配，将 nominal controller 依次替换为：

```text
scripted trajectory
→ BC / Diffusion Policy
→ open VLA / general manipulation policy
```

目标问题：

> **Can the same tactile reflex augment a slower high-level manipulation policy without retraining that policy?**

VLA 是系统级展示与外部有效性验证，不替代 Stage 3A 的受控物理证据。

## 12. 当前执行顺序

```text
Stage 1 representation
→ Stage 2P temporal probe
→ Stage 3A-0 hardware bring-up
→ Stage 3A-1 controlled passive perturbation acquisition
→ Stage 3A-2 matched Scalar / MCF controller development
→ Stage 3A-3 frozen closed-loop Anti-Tilt / Anti-Slip
→ only if supported: Stage 3B plug-and-play policy / VLA integration
```

论文事实、仓库已有结果、当前假设和后续展示始终分开记录。
