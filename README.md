# 机器人夹爪 × VLA — IROS 2027

> **暂定论文题目：** *Same Brain, Different Fingers: Task-Aligned Morphology and Embodiment Shift in Vision-Language-Action Manipulation*
>
> **目标会议：** IROS 2027  
> **目标截稿：** 2027-03-01

本仓库研究一个受控的机器人操作问题：

> **在机器人、执行器、传感器、动作空间、训练流程和策略检查点保持一致时，只改变末端手指形态，不同任务是否存在不同的最优 morphology？进一步地，换手指是否会给同一个 VLA 带来可测量的 embodiment / deployment distribution shift，而显式形态条件能否缓解这种变化？**

详细实验协议、统计方案、Gate 和逐阶段 TODO 见 [`IROS_2027_TODO.md`](./IROS_2027_TODO.md)。

---

## 1. 研究故事

项目不再只把问题表述为“8 mm、14 mm、25 mm 哪个成功率高”，而是按三层展开：

```text
Layer 1 — Mechanical Effect
不同任务存在不同的最优 finger morphology

Layer 2 — Embodiment Shift
换 finger 会改变 contact geometry、可达空间、视觉外观和 action consequences

Layer 3 — Morphology-Aware Policy
显式 morphology conditioning 是否帮助同一个 VLA 更好适应不同末端形态
```

可以把整体关系概括为：

```text
Morphology
    ↓
Physical Affordance / Contact Geometry
    ↓
Execution Distribution Shift
    ↓
Policy Performance
```

第一版实验只主动改变 **末端有效宽度**，因此主要形态变量优先写成：

```text
m = tip_width
```

而不是为了模型复杂度人为增加没有实际变化的参数。

---

## 2. 两个必须完成的贡献

### Contribution 1：Task–Morphology Interaction

建立一个可控的三指形 Piper 实验平台，验证：

> **不存在一个在所有任务上都最优的 finger morphology。**

理想趋势是：

```text
精细 / 受限空间任务：8 mm 更有优势
稳定 / 大物体搬运任务：25 mm 更有优势
14 mm 作为中性折中基线
```

真正需要证明的是稳定、可重复的：

```text
task_type × morphology interaction
```

而不是某个夹爪总体平均成功率最高。

### Contribution 2：Morphology-Aware Shared Policy

在同一训练池和同一共享策略框架下比较：

```text
A. π(a | image, state, language)

B. π(a | image, state, language, gripper_id)

C. π(a | image, state, language, tip_width)
```

研究显式 morphology awareness 是否能让共享 VLA 更好利用不同的末端物理特性，并减轻换指形后出现的 execution distribution shift。

主实验必须保持：

> **Same Brain, Different Fingers.**

不得用三个独立训练模型代替共享策略结果。

---

## 3. 三种冻结的末端宽度

| 编号 | 定位 | 末端宽度 |
|---|---|---:|
| `G_P` | 精细型 | **8 mm** |
| `G_N` | 中性基线 | **14 mm** |
| `G_W` | 稳定型 | **25 mm** |

第一轮尽量只改变末端宽度，并控制：

- 手指长度；
- 根部结构；
- 安装孔位；
- TCP；
- 材料；
- 接触表面；
- 最大夹持力；
- 驱动器；
- 运动速度与加速度。

打印后必须记录：

- 实际宽度；
- 单指质量；
- 整套末端质量；
- 材料；
- 打印参数；
- 安装偏差；
- TCP / 接触中心。

除非出现明确机械不可行问题，否则不再根据实验结果反复修改 8 / 14 / 25 mm。

---

## 4. 四个主任务

### P1：Type-C 刚性插头插入

**任务类型：精细定位 / 插接**

主要观察：

- 成功率；
- 周边碰撞；
- 对准失败；
- 插入时间；
- 最大接触力 / 腕部力矩；
- 最终插入深度。

要求 25 mm 手指“更困难但仍可完成”，避免任务退化为简单尺寸可行性判断。

### P2：窄盒 / 窄槽内部取物

**任务类型：狭窄进入 / 受限空间抓取**

主要观察：

- 成功率；
- 手指与侧壁碰撞；
- 是否能到达有效抓取位姿；
- 重抓次数；
- 完成时间。

### W1：宽瓶抓取与动态搬运

**任务类型：稳定抓取 / 抗滑移 / 抗转动**

流程：

```text
抓取
  ↓
抬升
  ↓
横向移动
  ↓
加速 / 减速
  ↓
放置
```

主要观察：

- 成功率；
- 滑移；
- 物体旋转；
- 掉落；
- 重抓次数；
- 搬运时间。

### W2：宽盒抓取、搬运与放置

**任务类型：大接触面稳定搬运**

主要观察：

- 成功率；
- 姿态稳定性；
- 滑移 / 转动；
- 重抓次数；
- 最终放置误差。

---

## 5. 为什么不能只看 success / fail

宽手指的假设不能简单写成：

```text
接触面积更大 → 摩擦力一定更大
```

因为在最简单 Coulomb friction 模型中，摩擦上限主要由摩擦系数和法向力决定。

真正值得验证的机制包括：

- 接触区域和压力分布变化；
- 对姿态误差的容忍度；
- 对物体转动的抵抗能力；
- contact wrench 能力变化；
- 狭窄空间中的碰撞和可达性变化。

因此正式实验除 success rate 外还记录：

```text
slip
rotation Δθ
collision
regrasp
pose error
contact force / wrist torque
```

希望最终能够形成：

```text
morphology
    ↓
contact / access mechanism
    ↓
slip / rotation / collision / regrasp
    ↓
success
```

而不是只报告相关性。

---

## 6. 正式主实验矩阵

```text
3 种手指
× 4 个任务
× 3 个未见物体
× 20 次配对测试
= 720 次真实机器人试验
```

要求：

- 训练集和测试集按 **物体实例** 划分，不能按轨迹随机划分；
- 各 morphology 使用相同初始条件编号、位置、姿态扰动种子；
- 正式 trial 顺序随机化；
- 先完成 720 次主证据，再运行额外扩展。

主统计包括：

- 每个物体等权成功率；
- `gripper × task_type` interaction；
- `Delta_match`；
- 最差物体成功率；
- 跨物体置信区间；
- failure taxonomy。

---

## 7. 机械 Pre-Pilot 先于大规模 VLA

在正式训练前，先用固定脚本 / 状态控制验证：

> **不依赖学习策略时，8 / 14 / 25 mm 是否已经产生可解释的 task-dependent mechanical effect？**

推荐第一轮：

```text
3 种手指 × 4 个任务 × 5 次
= 60 次 Pre-Pilot
```

重点检查：

1. 是否所有手指都接近 100%，任务太简单；
2. 是否所有手指都接近 0%，任务太难；
3. 8 mm 与 14 mm 是否可区分；
4. 25 mm 在精细任务中是否仍然可做；
5. 25 mm 在稳定任务中是否确有稳定性优势；
6. 差异是否来自进入空间、碰撞、滑移、转动等可解释机制。

如果机械层本身没有任何可辨识交互趋势，不应直接投入大量 VLA 数据采集。

---

## 8. Learning PoC：先跑通最短链路

第一版只做：

```text
G_N = 14 mm
+
W1 宽瓶搬运
```

目标不是论文级成功率，而是跑通：

```text
observation
    ↓
dataset
    ↓
training
    ↓
checkpoint
    ↓
policy inference
    ↓
Piper rollout
    ↓
automatic logging
```

W1 比 Type-C 插接更适合作为第一条 learning pipeline，因为接触精度要求更低，能减少“机械精度问题”和“训练管线问题”的混淆。

---

## 9. 第一个 morphology-shift 实验

当 `G_N = 14 mm + W1` 的策略可以运行后，直接冻结 policy：

```text
同一个 checkpoint
        ↓
8 mm
14 mm
25 mm
```

不重新训练。

这个 early pilot 用来观察：

- 换手指后 performance 是否下降；
- 哪些失败来自纯机械限制；
- 哪些失败来自视觉外观变化；
- 哪些失败来自同一个 action 在不同 morphology 下产生不同物理结果。

从研究角度，可以把它理解为：

```text
P_test(m = 8)
P_test(m = 14)
P_test(m = 25)
```

之间是否存在可测量的 execution distribution shift。

---

## 10. 一个必须重视的消融：视觉形态泄漏

如果腕部相机能直接看到 finger，模型即使没有显式 `tip_width`，也可能从图像中推断当前 morphology。

因此“无 morphology 输入”不能自动等价于“模型不知道 morphology”。

建议至少预注册以下 2 × 2 消融：

| | width token OFF | width token ON |
|---|---:|---:|
| finger visible | A | B |
| finger masked / 仅第三视角 | C | D |

它可以回答：

1. VLA 是否会隐式识别自己的 embodiment；
2. 显式几何参数是否在视觉已经可见时仍有增益；
3. 当 morphology 外观被遮罩后，连续 width conditioning 是否更加重要。

---

## 11. Unseen Morphology：真正的跨形态泛化实验

三种训练宽度全部被模型见过时，连续 `tip_width` 可能只是另一种 ID 编码。

因此如果主线进度健康，可额外打印一个：

```text
G_U = unseen width
```

它：

- 不参与训练；
- 不参与模型选择；
- 只在主实验完成后用于测试。

此时才能真正问：

> **Can a geometry-conditioned shared policy generalize to an unseen end-effector morphology?**

`G_U` 的结果单独报告，不与 8 / 14 / 25 mm 主矩阵混合统计。

---

## 12. 少量纠正数据的 morphology adaptation

另一个高级扩展是参考 DAgger：

```text
已训练 morphology
      ↓
切换到另一 morphology
      ↓
performance drop
      ↓
人工纠正少量失败状态
      ↓
0 / 5 / 10 / 20 corrections
      ↓
性能恢复曲线
```

研究问题：

> **How much corrective experience is required for a VLA to adapt to a new end-effector morphology?**

这可以把项目进一步连接到 embodiment adaptation / robot generalization，但它是高级扩展，不能阻塞 720 次主实验。

---

## 13. Kai0 / openpi 对本项目的定位

[`OpenDriveLab/kai0`](https://github.com/OpenDriveLab/kai0) 对本项目最有价值的是 **工程基础设施和 train–deploy alignment 思想**，而不是把它的全部算法搬进论文。

优先参考 / 复用：

- openpi / π₀.₅ fine-tuning；
- LeRobot dataset pipeline；
- policy server / client inference；
- Piper + RealSense 部署经验；
- temporal smoothing / ensembling；
- DAgger 数据采集逻辑；
- 数据转换与 augmentation 工具。

暂不进入主线：

### Model Arithmetic

可以研究：

```text
π8 + π14 + π25 → πmerged
```

但这需要 morphology-specific checkpoints，会破坏主实验的：

> **Same Brain, Different Fingers**

因此只允许作为 appendix / future work，不代替 shared checkpoint 主结果。

### Stage Advantage

更适合长程、多 semantic stage 的操作。当前 P1/P2/W1/W2 不为使用该方法而人为复杂化。

原则：

> **用 Kai0 帮我们少造基础设施，不让 Kai0 改写我们的研究问题。**

---

## 14. Scope Control

### 主线必须完成

```text
8 / 14 / 25 mm 硬件验证
        ↓
Mechanical Pre-Pilot
        ↓
Learning PoC
        ↓
Frozen-policy morphology-shift pilot
        ↓
共享策略：none / ID / continuous width
        ↓
720 次正式实验
        ↓
视觉泄漏与 morphology conditioning 消融
```

### 高价值扩展

- unseen morphology `G_U`；
- DAgger few-correction adaptation；
- morphology selector；
- 更多任务。

### 不得阻塞主线

- Model Arithmetic；
- 自动换爪；
- 复杂 morphology generation；
- 大量新增传感器；
- Cosmos 深度集成；
- architecture-level 大改。

---

## 15. 当前时间线

| 日期 | Gate / 目标 |
|---|---|
| **2026-09-06** | Gate 0：三手指 + 四任务 Prototype + logging + Mechanical Pre-Pilot；尽量完成 W1 Learning PoC |
| **2026-09-28** | Gate 1：至少一个精细任务和一个稳定任务出现方向相反、可重复的 morphology 优势 |
| **2026-10-07** | 国庆 Sprint：至少两个任务跑通完整 policy 闭环 |
| **2026-11-02** | Gate 2：共享策略 pipeline 稳定，并完成 frozen-policy 跨形态 pilot |
| **2026-12-14** | Gate 3：硬件、任务、测试对象、模型、指标和主假设冻结 |
| **2027-01-24** | 所有必须依赖真机的主实验原则上结束 |
| **2027-02-07** | 全文 v1 |
| **2027-02-14** | 内部审稿版 |
| **2027-02-21** | submission-ready |
| **2027-03-01** | IROS 2027 目标截稿 |

---

## 16. 当前立即执行

### 2026-08-24：打印与装机

- [ ] 打印 8 / 14 / 25 mm 三种 finger；
- [ ] 记录实际宽度、质量、材料和打印参数；
- [ ] 逐一安装到 Piper；
- [ ] 检查张开、闭合、左右对称和机械干涉；
- [ ] 检查接触中心和 TCP 一致性；
- [ ] 拍摄固定机位装机记录；
- [ ] 记录腕部相机中 finger 是否明显可见。

### 2026-08-25 至 08-30：机械冒烟测试

- [ ] 完成 3 种 finger × 当前 3 个对象的 3–5 次冒烟测试；
- [ ] 记录 success / fail；
- [ ] 记录 collision；
- [ ] 记录 slip；
- [ ] 记录 rotation；
- [ ] 记录 drop；
- [ ] 记录 cannot-enter；
- [ ] 记录 regrasp；
- [ ] 确定四任务夹具与难度；
- [ ] 定义 trial 开始 / 成功 / 失败 / 超时标准；
- [ ] 建立统一日志和视频命名。

本阶段不做：

- 不因为某几次结果不好就改 8 / 14 / 25 mm；
- 不在机械效应没有被确认前采大量 VLA 数据；
- 不把现有三个物体直接当最终测试集；
- 不训练三个独立 policy 冒充 shared policy；
- 不为了使用 Kai0 的全部模块而改变论文问题。

---

## 17. 仓库文档

- [`README.md`](./README.md)：研究问题、主实验、Kai0 启示和当前执行路线；
- [`IROS_2027_TODO.md`](./IROS_2027_TODO.md)：完整实验协议、统计方案、Gate、扩展实验与投稿决策。

项目原则：

> **早闭环、早发现失败、早冻结变量；先证明物理效应，再增加学习复杂度。**
