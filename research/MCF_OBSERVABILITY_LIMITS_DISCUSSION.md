# MCF 的物理可观测性边界：平面外倾覆与平面内旋转滑移

> **状态：2026-10-18，论文 Discussion / Limitations 写作备稿（analytic discussion only）。**
>
> **用途与权限边界：**本文是对 MCF 8D 表征的**理想化数学辨析**，不是新增实验方案、实验通过/失败结论，也不构成对第一篇论文冻结范围的修改。**不要求新增纯 Mz 实验，不切换 9D，不变更模型、Stage 3S、E1/E2/E3、Stage 3A、指标或已有数据。**如未来论文正文需要，可择要改写引用。
>
> 与 [最小论文方向冻结稿](ICRA2028_MINIMAL_PAPER_FREEZE.md)、[MCF 定义](MINIMALIST_CONTACT_FLOW.md)、[Stage 2P Now-casting](STAGE2P_NOWCAST.md)、[Stage 3S 机制对照](../experiments/stage3s_isaac/isaac_windows/e1s/PHASE_PROTOCOL.md) 配套阅读；以上文件及其原结论保持原样。

## 1. 为什么要在 Discussion 中讨论这个边界

第一篇论文的核心研究问题是：**仅有法向响应的触觉阵列中，紧凑空间状态是否能比 Scalar 更有效地支持偏心扰动下的闭环抗扰？** 主任务为 **Anti-Tilt**，当前主候选为逐指 MCF 8D + GRU，首轮控制权限仅为夹爪闭合增量 Δg。

一种需要避免的概念扩大是：将“部分偏心扰动可以引起法向压力重分布”理解为“任意切向外力或力矩都能由 CoP/MCF 唯一识别”。本讨论给出条件明确的**表示不可辨识反例**，用来解释方法的适用范围，而**不宣称已经测得真实 FlexiTac 对该运动失效**。

该辨析适合论文后部的 *Physical Interpretation / Observability Limits* 或 *Limitations*；不宜将其包装为新的实验证明、普适接触力学定理或第三项已验证贡献。

## 2. 三维偏心外力应先按力矩轴分解

固定指尖局部坐标系：传感器法向为 z，指尖平面为 x–y。外力作用点相对于选定参考点为 r=(r_x,r_y,r_z)，外力为 F=(F_x,F_y,F_z)。由 M=r×F：

$$
M_x=r_yF_z-r_zF_y,\qquad
M_y=r_zF_x-r_xF_z,\qquad
M_z=r_xF_y-r_yF_x.
$$

因此，即使外力都在指尖平面内，也至少需要区分：

- **平面外倾覆（out-of-plane tilt，绕 x/y 轴）**：例如切向力作用点有法向偏置 r_z，形成 M_x 或 M_y。物体与柔性层的相对倾角可能引起两侧法向压力不对称，进而改变 CoP、接触面积或总响应。但变化幅度和符号并非普适保证。
- **平面内偏转/扭转滑移（in-plane pivoting / torsional slip，绕 z 轴）**：例如切向力作用点有平面内偏置 (r_x,r_y)，形成 M_z。物体可能在指尖平面内平移、旋转，或二者结合；法向压力矩是否响应，取决于接触几何、柔顺性、摩擦、边界与载荷。

**仓库已有实验的准确覆盖范围：**[Stage 2P Now-casting](STAGE2P_NOWCAST.md) 的预测标签是 [F_x,F_y,M_x,M_y]，不包含 M_z。[Stage 3S 纯力/纯力矩/组合载荷协议](../experiments/stage3s_isaac/isaac_windows/e1s/PHASE_PROTOCOL.md) 的纯力矩对照明确采用 M_x/M_y（如 +F_x 对应 +M_y），**不是纯 M_z 扭转实验**。[相应报告](../experiments/stage3s_isaac/isaac_windows/e1s/outputs/phase_report_20261014_01/REPORT.md) 也不能被援引为纯 M_z 边界已经实测。Stage 3S 使用刚体接触 + SDF 触觉读出代理；真实柔性 FlexiTac 的机理仍不能由此直接宣布成立。

## 3. 条件性数学反例：零阶/一阶量不能区分中心纯旋转

把**理想化、连续、非负的法向压力场**记作 p(x)，定义域为完整容纳接触斑块的平面区域。为方便证明，这里先把响应看作经过理想校准的压力；真实离散阵列的 f_N 仍按仓库约定称为 *normal response / force proxy*，不自动当作 N。

在非零接触时，令：

$$
N=\int p(\mathbf{x})\,d\mathbf{x},\qquad
\mathbf c=\frac{1}{N}\int \mathbf{x}\,p(\mathbf{x})\,d\mathbf{x},\qquad
A=\operatorname{area}\{\mathbf{x}:p(\mathbf{x})>\tau\},
$$

其中 τ 为固定压力阈值；假设运动不造成边界裁剪、像素化误差或阈值变化。

考虑接触斑块围绕**自身当前 CoP c** 作平面内刚性旋转、压强随斑块一同转动的理想构造：

$$
p_\phi(\mathbf{x})
=p_0\!\left(\mathbf c+R_{-\phi}(\mathbf{x}-\mathbf c)\right),
\qquad R_\phi\in SO(2).
$$

对积分变量作刚性旋转换元，Jacobian 行列式为 1，于是：

$$
N_\phi=N_0,\qquad \mathbf c_\phi=\mathbf c_0,\qquad A_\phi=A_0.
$$

故对于仓库的逐指状态：

$$
s_t=[f_N,c_x,c_y,A],\qquad
\mathrm{MCF}_t=[s_t,s_t-s_{t-1}],
$$

若整段观测恰好符合上述变换、且 f_N 与 N 采用相容的理想读出，则：

$$
s_t=s_0,\quad \Delta s_t=\mathbf 0,\quad
\mathrm{MCF}_t=\text{constant},
$$

即使旋转角 φ(t) 随时间改变。

**严格含义：**这个反例证明从任意长的**相同 MCF 历史**中无法唯一恢复这类理想旋转角（或其变化率）；GRU/Mamba 增加时序记忆不能凭空恢复已被特征映射丢弃的信息。它**不证明**真实扭转一定产生这样的压力场，也不证明实际多指输入在一切旋转情况下都完全不可区分。若外部动作、视觉、机器人状态或其他指尖提供不同信息，结论需要重新限定。

尤其不能把压力中心（CoP）和物体的瞬时旋转中心（ICR）等同：CoP 是**法向压力加权中心**，ICR 是**刚体平面运动的运动学量**。接触斑块或物体发生平移，不意味着 CoP 必须同幅移动；CoP 移动也不意味着发生了同幅实际滑移。

## 4. 二阶矩何时含有额外信息，何时也无效

定义理想压力场的二阶中心矩：

$$
C=\frac{1}{N}\int
(\mathbf{x}-\mathbf c)(\mathbf{x}-\mathbf c)^\top
p(\mathbf{x})\,d\mathbf{x}.
$$

在上述**旋转压力场构造**中：

$$
C_\phi=R_\phi C_0 R_\phi^\top.
$$

若两个特征值不同（λ_1≠λ_2），主轴方向可能随 φ 改变：

$$
\theta=\tfrac12\operatorname{atan2}(2C_{xy},C_{xx}-C_{yy}).
$$

因此，**8D MCF 虽可能保持常量，非对称原始压力图或二阶方向信息仍可能区别这些构造状态**。这解释了为什么“高阶矩在某一任务中没有明显收益”不能推出“对所有接触运动，高阶矩都是冗余的”。

必须同时保留两个边界：

1. **主轴退化：**λ_1≈λ_2 时方向不稳定，完全各向同性时无唯一主轴；即使不退化，主轴角具有 π 周期，需要正确处理等价方向。Δθ 也不等于独立测得的真实 yaw。
2. **传感退化：**若圆对称压力场满足 p_\phi=p_0，则**连完整法向压力图**在理想模型中也没有发生可检测的改变。增加网络容量、时间窗口或二阶矩都不能从相同的法向观测恢复角度。实际扭转中出现的微观剪切形变、粘滑、边缘效应或柔性层法向—切向耦合，可能打破此退化；但其存在与可辨识程度需要由相应传感器和条件决定。

另需注意：MCF 并非对**固定坐标原点的任意旋转**都不变——(c_x,c_y) 是随坐标系旋转而变换的向量。本反例仅讨论**围绕自身 CoP 的压力场旋转且零阶/一阶不变**。

## 5. 接触物理与表征损失必须区分

存在两种概念上不同的“不可观测”：

- **表示层面的信息丢失（representation ambiguity）：**两张原始压力图可能不同，但经过 [f_N,CoP_x,CoP_y,A] 变成同一个状态；上述各向异性斑块纯旋转是解析构造。
- **法向测量层面的信息缺失（sensor-observation ambiguity）：**不同物理运动可产生相同的法向压力观测；理想各向同性压力场是一个有条件的例子。在这种情形下，Dense Raw 也不能保证恢复运动。

此外，固定夹爪间距 g 并不等于完整接触几何锁死：柔性层可形变，夹爪/物体并非必然绝对刚性，外部支撑和接触位置可能变化。切向外力与柔性层厚度 h 可在力矩平衡中形成耦合项，但**不能由 F_t h 这个力臂直接推出 CoP 必然按外力方向漂移**。其方向与大小依赖真实材料、摩擦及边界条件。经典摩擦滑动模型中关于 ICR、limit surface 的运动学直觉，也不能代替真实法向压力场测量。

## 6. 对本论文的含义：解释范围，而非扩大工作量

这段辨析能支持一种**有边界的结论**：

> 低维 MCF 的优势来自它保留了部分任务相关的压力空间信息，而不是它对所有接触运动都充分。对于指定 Anti-Tilt / 偏心扰动任务，某些载荷与几何可诱发可用的法向压力重分布；而对理想中心旋转，零阶和一阶描述可能保持不变。

**不能据此写成实验结论：**

- “真实 FlexiTac 已经被证明无法感知 M_z”；
- “当前 Stage 3S 已发现 M_z 引起的盲区”；
- “MCF 8D 能完美观测 M_x/M_y 或切向力方向”；
- “增加 Δθ 一定修复真实旋转滑移问题”；
- “二阶矩/Raw 一定能感知所有扭转”。

**当前无需采取的行动：**不安排额外 M_z 实验、不增加独立任务、不修改 MCF 输入、不增加 9D 训练、不改变阶段验收与论文主问题。把这一理想反例作为 Discussion / Limitations 中的**物理解释与适用边界**即可。若论文篇幅紧张，保留第 3 节的积分不变量与第 4 节的一句二阶矩说明就足够；无需把它升级成单独贡献。

## 7. 可用于论文的英文备稿（须随最终实测结果调整）

### A. Discussion — Physical Interpretation and Observability Limits

> **Observability limits of compact tactile statistics.** Our representation summarizes each normal-only tactile frame using the integrated response, pressure centroid, and contact area, together with their temporal differences. These descriptors retain spatial information potentially useful for rejecting eccentric disturbances, but they are not sufficient statistics for arbitrary contact motion. Consider an idealized planar pressure distribution that rotates rigidly about its own pressure centroid without changing its magnitude, support area, or contact boundaries. A change of variables shows that the integrated response, centroid, and area remain unchanged throughout this motion. Consequently, the MCF sequence is identical for different rotation angles, and temporal memory alone cannot recover the discarded orientation information. In contrast, for an anisotropic pressure patch, its second central moment transforms with the rotation and may retain orientation information. For a rotationally symmetric pressure distribution, even the complete normal-pressure image can remain unchanged. These are analytical counterexamples at the representation and measurement levels, **not empirical evidence that a physical tactile sensor cannot detect torsional slip**. In practice, shear-induced deformation, contact transitions, and material compliance may introduce additional observable changes. Thus, the utility of compact tactile states should be interpreted relative to the contact geometry, sensing physics, and control objective considered here.

### B. Short Limitations paragraph

> **Limitations.** MCF is designed to retain task-relevant changes in normal tactile responses rather than to reconstruct an arbitrary six-dimensional contact wrench. Under an idealized in-plane rotation of a pressure patch about its own centroid, the integrated response, centroid, and contact area can remain unchanged even when the patch orientation changes. This provides a representation-level non-identifiability example for MCF. It does not establish that actual torsional slip is unobservable with FlexiTac, since real contact deformation may alter the normal response. Accordingly, the present findings should not be generalized from the studied eccentric anti-tilt conditions to all rotational-slip regimes.

### C. One-sentence takeaway

> **The adequacy of a compact tactile representation is task- and contact-dependent, rather than determined by its dimensionality alone.**

## 8. 写作时的证据检查

- 理想压力场的换元与矩变换：**数学推导**，可以在清楚写出假设后成立；不需要另做实验才能提出反例。
- 现有 Stage 2P 目标与 Stage 3S 载荷方向：**仓库已记录的事实**，对应当前研究覆盖范围，不自动证明 M_z 性能。
- “MCF 相对 Scalar 的控制增益”：**只能采用最终冻结且匹配条件的实测结果表述**，不能由本讨论补强或替代。
- 真实 FlexiTac 的剪切耦合、迟滞和纯扭转信号：**本讨论未实测、也未证实其不存在**。
- 如以后研究问题变化，可在新的工作中研究法向特征与原始触觉 / 剪切敏感传感的互补性；**不是当前论文的新增 TODO**。

---

**归档声明：**本文件仅提供理论辨析与论文写作素材；不修改冻结稿，不新增实验义务，不预写实验成功或失败，也不替代 [Stage 3S 当前结论](../experiments/stage3s_alignment/README.md)。
