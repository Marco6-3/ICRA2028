# 实验路线：从 Physics Bottleneck 到 Temporal Memory

## Stage 3A — Controlled Real-Robot Tactile Reflex（当前最高优先级）

当前不直接接 VLA。先在真实机械臂/夹爪上建立受控 disturbance benchmark，并比较 Fixed / Scalar Reflex / MCF Reflex。

实验优先级：
1. Anti-Tilt / eccentric disturbance：更直接隔离 spatial contact state 的增量价值；
2. Anti-Slip / pull disturbance：验证扰动抑制与恢复。

第一轮 residual 只修改 gripper command。必须记录 max slip、max tilt、recovery time、survival/drop、grip effort 与完整 sensor→actuator latency chain。

现有 Stage 2P GRU/Mamba checkpoint 是 now-casting / state reconstruction 模型，不直接作为 residual policy。先采集受控扰动，再冻结 controller supervision / reward、动作边界、训练协议，最后在独立 trial 上做闭环评测。

如果 MCF 没有稳定优于 Scalar，不通过直接接 VLA 掩盖结果。只有 Stage 3A 获得可信闭环增量后，进入 Stage 3B 的 scripted / learned policy / VLA plug-and-play 系统展示。

完整协议见 [STAGE3A_REAL_ROBOT.md](STAGE3A_REAL_ROBOT.md) 与 [real_robot_reflex/PROTOCOL.md](../experiments/real_robot_reflex/PROTOCOL.md)。


状态：实验路线方案；已补充数学检查与公开数据探索性离线结果，真实硬件 Stage 0 和闭环尚未完成。更新：2026-10-06。

## 已执行的离线实验

[成果与决策总览](OFFLINE_RESULTS.md) 汇总 LeFlexiTac 动作拟合与 TaF 空间诊断。具体协议、全部失败／排除、验证调参、逐源误差、阈值敏感性、图表和脚本见 [实验归档](../experiments/contact_operator/README.md)。当前结果没有可靠支持完整 11D 的增量价值，不能据此宣布 Stage 1 通过。

下文保留长期实验标准；已经完成的探索性实验不等于所有标准已满足。此次没有运行 dense tactile token、Mamba 或 VLA；TaF 留出的是完整源序列，未核实为物体／材质 OOD。神经小样本效率与实机闭环仍需后续证据。

## 后续候选改进与运行登记

[VALIDATION_ROUTES.md](VALIDATION_ROUTES.md) 收录 11 条尚未运行的验证路线，固定当前目标为“紧凑触觉表示的接触状态估计”，不把任务阶段识别、Mamba 或纠偏控制写成已有结果。先做数据／时间接口 Q0，再按条件选择参数化、差分、面积、质量或读出的小消融；原 Stage-2P 四路对照继续保留。

使用 [候选矩阵](../experiments/validation_routes/MATRIX.csv) 登记进度，实际运行前复制 [记录模板](../experiments/validation_routes/RUN_TEMPLATE.md) 冻结数据、指标、实用差异和预算。当前矩阵全部为 candidate_not_run。新增路线不授权一次性大规模扫描，不覆盖旧协议；根据开发结果选定的组合须使用独立确认数据。

## 总原则

当前路线不先比较一堆时序网络，而先回答两个问题：

1. 低维 physics contact state 是否是一个有价值的 tactile bottleneck？
2. 长历史 temporal memory 是否会暴露单帧/短时读出没有显示出来的 representation × memory 交互价值？正式 Stage 2 仍需 Stage 1 支持，但允许一个低成本 Stage-2P probe 提前回答这一诊断问题。

所有阶段都必须把**表示、时序、控制**分开验证，避免同时改多个轴后只报告最终成功率。

---

## Stage 0 — Zero-Network Physics Sanity Check

候选硬件优先使用 FlexiTac 一类二维法向触觉阵列。固定真实硬件、固件、采样配置与记录格式后，采集重复序列：

no-contact → contact → load → redistribution / roll / tilt → release

第一阶段不训练神经网络。

### 必测量

- raw sampling rate 与 frame timestamp；
- transport latency；
- baseline update / threshold / filtering 设置；
- descriptor compute latency；
- P50 / P95 event-to-descriptor delay；
- repeated-trial noise、drift 和 repeatability；
- sustained load 是否被 baseline tracking 错误抹掉；
- descriptor 是否在 contact / release / redistribution 等事件处出现可重复变化。

### Phase-plane visualization

可视化包括但不限于：

- $(P,\Delta P)$；
- $(c_x,\Delta c_x)$、$(c_y,\Delta c_y)$；
- $(A,\sigma_1/\sigma_2)$；
- orientation-confidence trajectory。

这些图只用于形成 hypothesis。结构化轨迹不能替代后续定量比较。

### 继续条件

只有当 descriptor 在重复实验中：

- 数值稳定；
- 不依赖测试期未来信息；
- 事件响应具有可重复性；
- 不主要由 threshold / filter artifact 产生；

才进入 Stage 1。

### Stage 0 预注册门槛与 timebox

- task-relevant descriptor 的 repeated-trial ICC 目标 ≥ 0.90；ICC 不适用的维度必须在看测试结果前声明绝对误差界；
- 固定稳定载荷下 total-load proxy 的 CV 目标 ≤ 5%；
- preprocessing + descriptor compute P95 ≤ 一个 tactile frame interval 的 20%；
- event-to-descriptor P95 ≤ 一个 tactile frame interval；
- 硬件、固件与记录格式冻结后，以 4 周主动实验或 3 个独立固定协议采集批次（先到者）作为一次 timebox；仍不稳定则强制进入 descriptor / calibration / hardware pivot 讨论。

这些阈值是项目决策门槛，不是领域通用常数；若真实设备显示阈值不合理，只能基于训练/标定数据和工程约束前瞻性修订，并记录原因，不能看测试成绩后追溯修改。

---

## Stage 1 — Representation Bottleneck

比较四类表示：

1. **Raw**：原始 tactile map；
2. **LeFlexiTac-style dense tactile tokens**：保留 dense spatial tactile information 的现代 policy / VLA 接口；
3. **Physics**：约 11D physically interpretable contact state；
4. **Learned**：从 Raw 学得、维度与 Physics 匹配的 compact latent。

推荐的最小读出：

| 输入 | 读出 / 模型 | 目的 |
| --- | --- | --- |
| Physics 11D | Linear / tiny MLP | 检查物理 bottleneck 的低样本可分性 |
| Raw | Linear | 检查高维 raw 的直接可分性 |
| Raw | Tiny CNN | learned spatial baseline |
| Learned 11D | Tiny encoder + linear head | 与 Physics 同维度比较 inductive bias |
| LeFlexiTac-style dense tokens | matched policy head / available official path | 检查直接 dense token 化是否已经足够，记录 training / inference cost |

不要预设 Physics 必须击败 Raw。Physics 的潜在价值是 **compactness、data efficiency、OOD robustness、latency 与 interpretability**，不是信息量更大。

### 数据效率

至少报告：

1%、5%、10%、25%、50%、100% training data

下的主要任务指标与方差。

### OOD

按完整 trial / object / acquisition batch 分割，优先测试：

- 未见物体；
- 未见接触位置；
- 未见刚度 / 表面条件；
- 必要时新的 indenter geometry。

不能随机打散帧后声称 OOD。

### 对抗性信息丢失测试

#### A. Moment-matched but locally different

不要人工强行同时匹配 $P,CoP,A,\Sigma$。从真实数据中跨类别搜索：

$$
(i,j)^*=\arg\min_{y_i\neq y_j}\|s_i-s_j\|_2
$$

同时要求 raw tactile map 差异较大。

目标：找到 physics descriptor 近似相同、局部 pressure pattern 不同的真实样本，量化 many-to-one compression 的信息损失。

#### B. Different contact dynamics

至少覆盖两类：

- click / bistable event：$\Delta P$ 大，CoP 变化相对小；
- rolling / tilting / redistribution：总压力变化较缓，CoP / shape 持续变化。

目标不是让 Physics “碾压” Raw，而是判断 compact physical state 是否以更低成本显式暴露控制相关动力学。

### Stage 1 主要指标

- classification / regression task metric；
- sample efficiency curve；
- IID → OOD performance drop；
- representation dimensionality；
- preprocessing / encoder P50、P95 latency；
- CPU / GPU / memory footprint；
- failure examples。

### 继续条件

Physics bottleneck 至少需要在以下一个或多个维度体现清晰价值，且没有不可接受的信息损失：

- data efficiency；
- OOD robustness；
- latency / compute；
- interpretability / diagnostic value。

若 Physics 与 learned compact latent 相比没有明显价值，或丢失的信息正是目标闭环任务所必需，则应停止、修改 descriptor 或转向 learned representation，而不是直接进入 Mamba。

---

## Stage 2P — Low-cost Temporal Interaction Probe

Stage 1 当前尚未通过，但允许一个与其并行的低成本诊断 probe，专门检验：

> **Does temporal memory reveal value that instantaneous readout missed?**

该 probe **不是**进入正式 Stage 2，也不能据此宣称 11D 已成立。第一轮固定比较：

| Representation | Temporal backbone | 目的 |
| --- | --- | --- |
| Scalar / total load | same Mamba-style streaming model | TacMamba-style 1D memory baseline |
| Scalar + centroid (3D) | same model | 当前最强的简化 spatial baseline |
| Physics11D | same model | 检验 richer structured state 是否在长历史中出现增量 |
| Learned11D | same model | 排除“只是低维 latent”这一替代解释 |

必须固定：相同数据划分、相同可见历史、相同 hidden/parameter budget、相同训练步数/优化协议、相同 decision frequency。history length 可以做一个**预注册的小 sweep**，但不得看 test 后追加长度或网络规模。

首轮 probe 不接 π0.5/VLA，不以最终 manipulation success 为目标，也不要求先复现 TacMamba 的完整系统；只借用其 streaming-memory 范式做 matched comparison。

### Stage-2P 判读

- 若 Physics11D + memory > Scalar+centroid + memory，且优势随有效历史出现而非 current-only 就存在：支持 richer structured spatial state × memory 交互，值得进入正式 Stage 2；
- 若 Physics11D ≈ Scalar+centroid：优先缩减 descriptor，3D/更小 state 可能已经足够，不维护“11D 必要”的 claim；
- 若 spatial variants 都不优于 Scalar：削弱 spatial-memory 主线，不进入 VLA 来掩盖结果；
- Learned11D 若明显更优，则转向 learned/hybrid bottleneck，而不是强保 physics。

Task 1 使用 non-inferiority 判断：Physics+memory 相对 Scalar+memory 的主要指标默认允许 **3% 相对退化 margin**。Task 2/3 的 superiority claim 必须使用预先冻结的主指标与 source/trial-level paired uncertainty；不能只凭单 seed 均值。

---

## Stage 2 — Formal Temporal Memory

只有 Stage 0 / 1 通过，或 Stage-2P 给出足以重新支持 representation×memory 假设的预注册证据后，才扩大研究：

$$
s_{1:t}\rightarrow h_t.
$$

强候选包括 TacMamba / Mamba 式递推 SSM，但不预设骨干。

建议比较：

| Representation | Temporal model / interface | 所回答的问题 |
| --- | --- | --- |
| scalar total force | **matched TacMamba-style streaming memory** | 最关键 memory baseline；隔离 spatial state 的增量价值 |
| Physics state | streaming memory | 物理结构是否是更好的长历史接口 |
| Learned compact latent | matched temporal model | 收益是否来自 physics inductive bias |
| Raw / rich tactile | temporal encoder | 信息更丰富但成本更高的参照 |

需要固定：

- 历史可见范围；
- hidden-state size / parameter budget；
- sensor update rate；
- training data；
- decision frequency；
- end-to-end latency budget。

不能只比较模型 forward time。

---

## 三个诊断任务

不追求“很多任务 SOTA”，而是逐层检验三个 scientific claims。

### Task 1 — Blind bistable / click counting

检验 Physics state 是否保留 scalar-force memory 擅长的时间事件与长时计数能力。

关键比较：scalar force + matched memory vs Physics state + matched memory。visual-only policy 仅作为 observability negative control，不预设其具体成功率。

### Task 2 — Contact Redistribution Recovery under Eccentric Loading

检验总接触强度近似不变时，空间 contact geometry 是否提供 scalar force 看不到的控制信息。尽量构造 $P_t$ 近似稳定而 $c_t,\Sigma_t$ 明显变化的扰动。

关键比较必须使用 same temporal model，避免把收益归因于更强 backbone。CoP movement 只解释为 contact redistribution cue，不能未经验证称为 incipient slip。

### Task 3 — History-dependent tactile search / insertion

检验当前 tactile state 存在歧义时，contact history 是否真正有用。任务应尽量满足：

$
s_t^A \approx s_t^B,\qquad s_{t-H:t}^A \neq s_{t-H:t}^B.
$

关键消融：Physics state without memory vs Physics state + memory，并与 Learned compact latent / dense tactile baseline 比较。

详细逻辑见 [BASELINES.md](BASELINES.md)。

---

## Stage 3 — Closed-Loop Control

Representation 与 memory 都只是假设链条中的中间变量。最终若要声称“改善 tactile-reactive manipulation”，必须进入真实闭环任务。

优先从上述三个诊断任务中选择能直接支撑当前 claim 的任务；若前期证据否定某个任务假设，可以替换，而不是为了保持故事强行执行。额外候选包括：

- sustained contact with disturbance；
- rolling / contact redistribution；
- insertion / snap / click-like transitions；
- 对局部 pattern 有要求的 negative-control task。

至少报告：

- task success / continuous quality；
- physical event → action effect 的端到端 P50 / P95 latency；
- dangerous / excessive-contact events；
- unseen-object / unseen-contact-condition performance；
- failure taxonomy。

---

## 因果性与数据纪律

- 每个传感器记录 sampling timestamp 与 arrival timestamp；
- 动作记录 command timestamp 与实际执行时间；
- 所有在线输入必须在决策时已经到达；
- episode 开始重置 memory state；
- baseline、normalization、threshold 与 calibration 不得使用测试 episode 的未来数据；
- no-contact baseline 若在线更新，接触时必须按协议冻结；
- 训练 / 验证 / 测试按完整轨迹、对象和采集批次划分；
- 仿真 privileged friction / object state 默认只用于锁定模型后的只读诊断。

## 投稿前证据门槛

- Physics representation 的收益和失败边界都有定量证据；
- LeFlexiTac-style dense tactile path 是认真实现或明确说明适配差异的 baseline；
- TacMamba-style scalar memory 使用 matched temporal model；
- RDP / nearest slow-fast controller 在 closed-loop claim 需要时作为 reactive baseline；
- 同维 learned latent 是认真调优的 baseline；
- temporal model 的收益不能只是更多参数、更多历史或不同刷新率；
- 报告完整端到端 latency，不只报 descriptor 或网络 forward；
- 若主张真实 contact-rich manipulation，必须有真实机器人闭环证据；
- 负结果与停止条件保留，不能追溯性重写假设。

详细 descriptor 与底层预处理定义见 [PHYSICS_BOTTLENECK.md](PHYSICS_BOTTLENECK.md)。
