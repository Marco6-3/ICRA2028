# Phase 1 Protocol — Controlled Spatial Contact Identification

状态：预注册采集协议 v1。适用于第一轮 FlexiTac 自采数据。  
研究阶段：Why Spatial? / Stage 0 + Stage 1 controlled experiment。  
本轮不训练 Mamba，不接 π0.5，不做闭环控制。

## 1. Scientific question

核心问题：

> 在总接触载荷近似相同的条件下，不同接触位置 / 倾角是否能够被二维触觉空间状态稳定地区分？最小需要多少 spatial state？

要隔离的变量：

- 空间位置：`x`
- 接触倾角：`theta`
- 总接触载荷水平：`load_level`

第一轮只改变 x、theta、load。材质、压头形状、摩擦表面、速度等保持固定。

## 2. Hypotheses

H0-spatial:
在 matched-load 条件下，Scalar total-load proxy 已足够解释位置 / 倾角标签；空间信息没有明确增量。

H1-spatial:
在 matched-load 条件下，至少一种 spatial representation（3D centroid state、Physics11D、Learned11D 或 Raw encoder）比 Scalar 更稳定地恢复位置 / 倾角。

本实验不预注册“11D 必须最好”。若 `[P,cx,cy]` 与 Physics11D 相当，则缩小 descriptor 是正确结论。

## 3. Hardware setup

最低配置：

- 1 个 FlexiTac，固定在刚性底座上；
- 1 个形状固定的压头 / indenter；
- 能够重复给出 x 位置和倾角 theta 的机械臂或刚性定位工装；
- 主机记录 FlexiTac 原始帧和时间戳。

推荐但非强制：

- 独立六维 F/T 或单轴 load cell；
- 外部相机，用于事后核查接触位置与滑移；
- 机械臂控制器时间戳。

### 3.1 坐标定义

将 FlexiTac 有效区域归一化：

```
x_norm = -1 ... +1
y_norm = -1 ... +1
```

本轮只沿 x 方向改变位置，固定 `y_norm = 0`。

正式位置：

```
x_norm ∈ {-0.50, -0.25, 0.00, +0.25, +0.50}
```

若边缘位置导致压头超出有效区域，应在采集前统一缩放全部位置，例如改为 `{-0.40,-0.20,0,+0.20,+0.40}`，并把变更记录进 session metadata；不得采到一半只改单个条件。

倾角：

```
theta ∈ {-10°, -5°, 0°, +5°, +10°}
```

倾角定义为绕 y 轴旋转，使主要接触变化沿 x 方向发生。

## 4. Load-level definition

本轮不预先写死牛顿值，避免在没有独立 F/T 标定时制造虚假绝对力。

在 pilot 开始前，用中心位置、theta=0° 做安全标定，冻结三个稳定接触水平：

- L = low
- M = medium
- H = high

如果有独立 F/T：用 Fz 设定 L/M/H，并同时记录 FlexiTac P。

如果没有 F/T：用 FlexiTac total-response proxy P 做闭环匹配，但必须明确：
- P 只用于构造 matched-load 条件；
- x / theta 标签仍来自机械臂 / 工装；
- 不能用“P 匹配”本身证明空间表示正确。

matched-load 容差：

```
|P_trial - P_target| / P_target <= 5%
```

若无法稳定进入 ±5%，该 trial 标为 invalid，不通过增大容差偷偷纳入。

## 5. One-trial timing

每条 trial 保存完整连续序列：

1. 2 s no-contact：估计 / 检查 baseline；
2. 1 s approach：接近并建立接触；
3. 2 s settle / load-match：达到指定位置、角度与目标载荷；
4. 2 s stable hold：正式稳定评价窗口；
5. 1 s release；
6. 1 s no-contact：检查 release 与 baseline recovery。

总时长约 9 s。

如果机械臂运动速度导致 1 s approach 不合理，可以统一调整，但必须在 pilot 前冻结，整批保持一致。

## 6. Pre-run calibration

每次实验 session 开始：

1. 固定传感器、压头、夹具，拍照并记录硬件版本。
2. 空载 10 s，检查坏点、漂移和丢帧。
3. 中心位置 theta=0° 做 5 次重复接触。
4. 选定并冻结 L/M/H load setpoint。
5. 检查稳定保持窗口没有饱和 taxel。
6. 核对 FlexiTac timestamp、host receive timestamp；若有 robot/F-T，同步记录。
7. 运行 `PILOT_MATRIX.csv`，不要直接跑正式 108 条。

## 7. Pilot — 24 trials

Pilot 只检查三件事：

### P0 Baseline
3 条 no-contact trial。

检查：
- drift；
- 丢帧；
- baseline recovery。

### P1 Repeatability
中心 x=0、theta=0，在 L/M/H 各重复 3 次，共 9 条。

检查：
- total-load proxy CV；
- centroid / descriptor repeated-trial repeatability；
- 是否存在明显 hysteresis / baseline artifact。

### P2 Minimal Why-Spatial
x = left / center / right，即 `{-0.5,0,+0.5}`，theta=0，load=M，各重复 4 次，共 12 条。

检查：
- 三个位置的 P 是否满足 ±5% matched-load；
- raw tactile map 是否发生可重复空间移动；
- centroid 是否随位置呈一致变化；
- 是否存在边缘饱和 / 接触超出阵列。

Pilot 未通过时，不跑 108 条正式矩阵。

## 8. Formal matrix — 108 trials

### S0 Baseline drift — 5 trials
无接触，每条 9 s。

### S1 Repeatability — 15 trials
中心位置、theta=0：
- L × 5
- M × 5
- H × 5

### S2 Spatial position — 25 trials
theta=0、load=M：

```
x_norm = -0.50, -0.25, 0, +0.25, +0.50
```

每个位置 5 次。

这是第一核心任务：在 matched-load 下检验 Why Spatial。

### S3 Position × load — 27 trials
位置：

```
x_norm = -0.50, 0, +0.50
```

载荷：

```
L, M, H
```

每个组合 3 次。

目的：检查 spatial readout 是否只是偷用了 load correlation。

### S4 Tilt — 20 trials
x=0、load=M：

```
theta = -10°, -5°, 0°, +5°, +10°
```

每个角度 4 次。

目的：检查接触形状 / 二阶矩 / orientation descriptor 是否真正带来信息。

### S5 Position + tilt — 16 trials
组合：

```
(-0.5, -5°)
(-0.5, +5°)
(+0.5, -5°)
(+0.5, +5°)
```

load=M，每个组合 4 次。

目的：检查位置与倾角耦合时 descriptor 是否仍可解释。

## 9. Randomization

不要按照“从左到右、从小角度到大角度”连续采完整数据。

规则：
- 同一 block 内随机 trial 顺序；
- 不连续执行同一 x/theta 超过 3 次；
- 每 20–30 条插入 1 条 no-contact drift check；
- session_id 和真实执行顺序必须保存。

CSV 中的 `planned_order` 是默认顺序；实际顺序另存 `executed_order`，允许随机化但不允许丢失映射。

## 10. What to save per trial

必须保存 raw sequence，而不是只保存平均 descriptor。

至少包括：

- raw tactile frame
- processed tactile frame
- sensor timestamp
- host receive timestamp
- target x / y / theta / load level
- actual robot pose / fixture position
- total-load proxy P
- centroid cx, cy
- active area
- sigma1, sigma2
- orientation ox, oy
- delta_P, delta_cx, delta_cy
- contact_flag
- valid_flag
- invalid_reason
- session_id / trial_id

若有独立 F/T：
- Fx, Fy, Fz, Tx, Ty, Tz
- F/T timestamp

## 11. Stable evaluation window

主分析默认只使用第 4 阶段 stable hold 的最后 1.0 s。

原因：
- 避免 approach / settling 过程污染静态 spatial identifiability；
- 同时完整 sequence 仍被保存，为后续 temporal probe 保留可能性。

窗口定义在看模型结果前冻结。不得为了提高某个 representation 的分数单独换窗口。

## 12. Analysis plan

使用 trial-level split，不随机打散 frame。

第一层：重复性
- P 的 CV；
- task-relevant descriptor 的 ICC 或等价 repeated-trial 指标；
- saturation / dead taxel / dropped-frame rate。

第二层：位置 x 预测
比较：
- Scalar P
- [P,cx,cy]
- Physics11D
- Learned11D
- Raw + reasonable spatial encoder

主指标：
- MAE(x_norm)
- R²
- trial-level paired uncertainty

第三层：theta 预测
同上，主指标：
- MAE(theta)
- R²

第四层：data efficiency
只在正式数据通过采集 QC 后再做 10% / 25% / 50% / 100% 训练数据比较。

## 13. Go / No-Go

### Gate A — acquisition
满足后才能正式分析：
- 没有系统性饱和 / 大面积坏点；
- timestamp 与 trial metadata 可追溯；
- matched-load 条件大多数 trial 能进入 ±5%；
- repeated-trial descriptor 达到项目预注册重复性要求，或明确记录不满足的维度。

### Gate B — Why Spatial
至少一种 spatial representation 在 matched-load 的 x / theta 任务上相对 Scalar 给出稳定增量，且不是由 future leakage 或 load mismatch 解释。

如果 Scalar 已经和 spatial state 一样好：
- 不进入“空间必要性”叙事；
- 检查任务设计是否真正消除了 load cue；
- 若任务本身确实不需要空间信息，则换 scientific task，而不是调模型刷结果。

### Gate C — minimal state
重点比较：

```
[P,cx,cy]  vs  Physics11D
```

如果相当：
- 下一阶段优先 3D / 更小 state；
- 不维护“完整 11D 必要”的 claim。

如果 11D 只有在 tilt / shape task 上出现增量：
- 将 11D claim 限定为需要 shape/orientation 的任务。

## 14. After Phase 1

只有 Phase 1 证明存在可信的 spatial observability gap 后，才设计 Why Memory 数据：

```
current_state(A) ≈ current_state(B)
history(A) != history(B)
future / required action(A) != future / required action(B)
```

然后进入 Stage-2P：

```
Scalar
Scalar+centroid
Physics11D
Learned11D
    -> same streaming temporal backbone
```

此时再决定 Mamba 是否只是方便的 probe backbone，还是值得成为最终方法组成部分。
