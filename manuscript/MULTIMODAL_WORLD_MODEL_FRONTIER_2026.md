# 2025–2026 多模态物理感知与机器人世界模型前沿调研与切入评估

> **调研目的**：系统梳理 2025–2026 年全球顶级具身智能实验室与工业前沿机构（含 Meta FAIR、TU Dresden LASR、Stanford REAL、SJTU/清华 TEA-Lab/SHAILab、Columbia RoboPIL、UIUC PLAN-Lab/GeWu-Lab、Specialist AI、NVIDIA GEAR/Cosmos、Skild AI、Generalist AI、TRI、Physical Intelligence、Berkeley RAIL 等）在**多模态物理感知、视触觉世界模型与柔顺力控**方向上的最新论文与开源 GitHub 仓库；在代码机制层面对比 `TacForeSight`、`VTAM`、`OmniVTA`、`UMI-FT` 与 `PAEC` 的差异；并据此评估“强物理/多模态感知 + 轻重分离世界模型 + 超越人类感官操作”的红海与蓝海切入点，形成 PAEC 下一阶段（PAEC v2）的可执行研发路线。

---

## 一、2025–2026 全球顶级实验室与前沿机构 GitHub 版图

> 注：本节仅收录已在仓库外网调研中核验的直达 GitHub 仓库与 `github.io` 前沿成果，按研究阵营与技术主线归类。

### 1. Meta FAIR & TU Dresden LASR（动作条件潜空间世界模型 & 多感官触觉预训练）
- **V-JEPA 2 & V-JEPA 2.1**（Meta FAIR, 2025–2026.03）：[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) — 在海量无标注视频预训练后，仅用约 62 小时 `DROID` 机器人交互数据微调动作条件预测头（`V-JEPA 2-AC`），通过潜空间 MPC 实现新环境零样本抓取与放置。代表视频潜空间动作条件世界模型的基准范式，但高度依赖外视角 RGB，在遮挡与接触力控细节上存在天然盲区。
- **Sparsh & Sparsh-X**（Meta FAIR）：[facebookresearch/sparsh](https://github.com/facebookresearch/sparsh) / [facebookresearch/sparsh-multisensory-touch](https://github.com/facebookresearch/sparsh-multisensory-touch) — 面向高分辨率视触觉（DIGIT / GelSight）及 **Digit 360 多感官指尖**（`Sparsh-X`：触觉图像 + 音频 + IMU + 气压）与 `Sparsh-Skin` 的自监督掩码潜空间预训练家族。
- **OpenTouch & RobotControlStack**（TU Dresden LASR, ICRA 2026）：[lasr-lab/opentouch-interface](https://github.com/lasr-lab/opentouch-interface) / [RobotControlStack/robot-control-stack](https://github.com/RobotControlStack/robot-control-stack) — 统一真机多模态触觉采样接口与实时控制工程栈。

### 2. Stanford REAL Lab（Shuran Song 团队：高频力觉 + 柔顺动作空间）
- **UMI-FT**（2026）：[real-stanford/UMI-FT](https://github.com/real-stanford/UMI-FT) — 双指指尖集成 **360 Hz `CoinFT` 六维力传感**，并提出自适应柔顺策略（Adaptive Compliance Policy），**动作空间直接输出位姿 + 内抓力（internal grasp force）+ 目标刚度**，突破纯位置 chunking 对力控接触的限制。
- **DexUMI / ManiWav / Minimalist Compliance Control**：[real-stanford/DexUMI](https://github.com/real-stanford/DexUMI)（灵巧手无本体迁移）、[real-stanford/maniwav](https://github.com/real-stanford/maniwav)（指尖音频接触波形学习）、[real-stanford/minimalist_compliance_control](https://github.com/real-stanford/minimalist_compliance_control)（轻量真机柔顺底环）。

### 3. SJTU / 清华 TEA-Lab / SHAILab（双频慢-快策略与隐式力反馈）
- **Reactive Diffusion Policy (RDP)**（RSS 2025）：[xiaoxiaoxh/reactive_diffusion_policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) — 确立“慢速视觉扩散规划 + 快速因果力/触觉反馈修正”的双频分层基准（本仓库离线跨库测试即以其 `peel_v3 / wipe_v1.5 / lift_v2` 为目标域）。
- **ImplicitRDP**（RA-L 2026）：[Chen-Wendi/ImplicitRDP](https://github.com/Chen-Wendi/ImplicitRDP) — 将高频力/触觉闭环隐式嵌入端到端结构，减少双频模块人为割裂。
- **FoAR / ForceMimic / TEA-Lab**：[Alan-Heoooh/FoAR](https://github.com/Alan-Heoooh/FoAR)（RA-L/IROS 2025）、[ForceMimic](https://github.com/ForceMimic)、[TEA-Lab](https://github.com/TEA-Lab) — 力-视多模态接触示教采集与自适应接触策略生态。

### 4. Columbia RoboPIL（Yunzhu Li 团队：物理孪生与粒子动力学）
- **PhysTwin & 可变性物理建模**（ICCV 2025 + 2026 Boba/Deform360）：[Hanxiao-Jiang/PhysTwin](https://github.com/Hanxiao-Jiang/PhysTwin) / [RoboPIL](https://github.com/RoboPIL) / [YunzhuLi](https://github.com/YunzhuLi) — 将交互视频与物理粒子/弹簧-质点系统耦合进行在线系统辨识，为物理显式世界模型提供连续形变与接触参数标定参照。

### 5. UIUC RoboTouch / PLAN-Lab / GeWu-Lab / UMD PRG（通用触觉表征与无预训练代理）
- **AnyTouch & AnyTouch 2**（ICLR 2025 / ICLR 2026）：[GeWu-Lab/AnyTouch](https://github.com/GeWu-Lab/AnyTouch) / [GeWu-Lab/AnyTouch2](https://github.com/GeWu-Lab/AnyTouch2) — 跨异构视触觉传感器（GelSight/DIGIT/TacTip 等）统一表征对齐。
- **VTAM**（PLAN-Lab, *without Tactile Pretraining*）：[PLAN-Lab/VTAM](https://github.com/PLAN-Lab/VTAM) — 跳过独立触觉预训练，直接复用冻结视频 VAE/DiT 主干 + 单步动作专家，并用**光流-力代理（flow-force proxy）**估计凝胶受力。反衬出仅靠视觉形变流代理在**零形变硬接触**与**传感器零点漂移**下存在天然盲区。
- **FeelAnyForce & 光学触觉仿真**：[prgumd/FeelAnyForce](https://github.com/prgumd/FeelAnyForce) / [Robo-Touch/tactile_optical_simulation](https://github.com/Robo-Touch/tactile_optical_simulation) — 面向视触觉凝胶力学与光学反演的高保真仿真与力估计工具链。

### 6. 2026 视触觉/力觉世界模型专门项目与高频策略潮
2026 年年中至秋季出现了一波密集的“视觉 + 触觉/力觉世界模型”工作：
- **TacForeSight / TacForceWM**（*arXiv:2606.11184*, 2026.06）：[TacForeSight/TacForeSight](https://github.com/TacForeSight/TacForeSight) — 使用 **120 Hz 腕部 F/T 经空洞因果 1D CNN** 编码后，驱动 **30 Hz 双指触觉潜空间世界模型 `TacForceWM`** 并衔接流匹配动作策略。
- **OmniVTA**（2026.03 / 2026.08）：[MrSecant/OmniVTA](https://github.com/MrSecant/OmniVTA) — 依托超 `21,000` 条轨迹的 `OmniViTac` 数据集，构建 **30 Hz 双流视触觉世界模型（VTWM）+ 60 Hz 反射式潜空间触觉控制器（RLTC）**，由潜空间预测误差 $z_{\text{pred}} - z_{\text{obs}}$ 直接触发高频补偿。
- **Dexterous-Tactile-World-Model (DTWM)**（2026.09）：[Adonis-galaxy/Dexterous-Tactile-World-Model](https://github.com/Adonis-galaxy/Dexterous-Tactile-World-Model) — 外穿戴触觉手套驱动的灵巧手扩散世界模型。
- **同期未全开源或处于 `github.io` / arXiv 阶段的前沿工作**：`VT-WAM`（非对称 MoT + 流匹配联合生成视/触/动 + 接触门控 AVTAG，2026.07）、`ViTacWorld`（真机+仿真大规模动作条件视触动联合，2026.09）、`DexTacWAM`（[dextacwam.github.io](https://dextacwam.github.io/), *arXiv:2609.24976*, 2026.09）、`$N_0$-TWAM`、`TACO`（触觉世界模型做后训练自纠正器，2026.09）、`ACTS`（人类手持传感器摩擦动力学迁移，2026.10）、`VT-WM`（Cosmos + Sparsh 对齐）、`TacMamba`（Mamba 压缩触觉历史衔接慢速 VLA，2026.03）、`T-Rex`（[tactile-reactive-dexterous.github.io](https://tactile-reactive-dexterous.github.io/), 2026.06）、`TacForcing`（2026.08）、`LeFlexiTac`（[tna001-ai.github.io/LeFlexiTac/](https://tna001-ai.github.io/LeFlexiTac/), 2026）与 `RoboJEPA / JEPAGeneralist`（2026）。

### 7. Specialist AI 与硅谷/工业具身前沿实验室
- **Specialist AI（UC Berkeley 团队）**：[specialist-ai](https://github.com/specialist-ai)（含 `specialist`、`specialist_description`、`specialist_bc`、`specialist_sim`、`specialist_sdk`、`specialist_lerobot`）— 围绕低成本手持夹爪与全景 Insta360 X3 打造高精 6-DoF 度量级采集-训练-仿真-部署全栈。该栈目前高度视觉中心化，**在腕部高频力觉抗漂移与物理内部缺陷无损读出上仍属空白**，可直接作为真机数据/部署底座的参照。
- **NVIDIA GEAR & Cosmos**：[NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)、[NVIDIA/GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams)、[nvidia-cosmos](https://github.com/nvidia-cosmos)（Cosmos 2.5 / Cosmos 3 全模态世界基础模型，偏重海量视频生成与合成轨迹幻想，边缘端 `>100 Hz` 实时力闭环与物理残差探针并非其主打）。
- **Skild AI & Generalist AI**：[skild-ai/steel](https://github.com/skild-ai/steel) / [kyegomez/Open-Gen](https://github.com/kyegomez/Open-Gen)（GEN-0/1/1.5 社区复现）。
- **Toyota Research Institute (TRI)**：[ToyotaResearchInstitute/lbm_eval](https://github.com/ToyotaResearchInstitute/lbm_eval)（LBM / Drake / Punyo 软触觉气囊 / TACTIC）。
- **Physical Intelligence / Berkeley RAIL / Stanford IPRL / UT Austin RPL**：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)、[rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl)（人类在环残差强化学习突破纯 BC 上限）、[rail-berkeley/serl](https://github.com/rail-berkeley/serl)、[stanford-iprl-lab/fact](https://github.com/stanford-iprl-lab/fact)、[UT-Austin-RPL/deoxys_control](https://github.com/UT-Austin-RPL/deoxys_control)（[UT-Austin-RPL](https://github.com/UT-Austin-RPL)）。
- **持续追踪聚合导航库**：[NTUMARS/Awesome-Tactile-Force-aware-Robot-Learning](https://github.com/NTUMARS/Awesome-Tactile-Force-aware-Robot-Learning)、[OpenHelix-Team/Awesome-Force-Tactile-VLA](https://github.com/OpenHelix-Team/Awesome-Force-Tactile-VLA)、[aitofound/awesome-robot-world-model](https://github.com/aitofound/awesome-robot-world-model)。

### 8. “超越人类感官”的非传统物理模态早期探索
相比上述拥挤的“RGB + GelSight”赛道，将非人类物理传感接入操作或空间智能的工作仍处于极早期：
- **The MOTIF Hand**（GitHub 2025）：在机器人灵巧手上集成**热成像（Thermal）+ 惯性（IMU）+ 力觉**多模态传感用于精细接触操作。
- **RoBustLab Thermal-IR & Multi-Spectral Robotics**（ICRA/ICCV 2025）：将长波红外热成像与视觉融合用于烟雾/弱光/极端工业现场空间感知。
- **mmHSense**（2026）与毫米波感知开源生态：分布式毫米波雷达点云用于穿透遮挡与空间交互。
- **DOVE**（*Nature Sensors*）：仿生多模态触觉融合（力、振动、温度等）感知框架。

---

## 二、四大代表性架构机制对比剖析（vs. PAEC）

为明确 PAEC 在 2026 前沿版图中的不可替代定位及下一步演化抓手，下表将 4 个代码逻辑最清晰的代表性框架与 PAEC 进行机制级拆解对照：

| 维度 | **TacForeSight** (`TacForeSight`) | **VTAM** (`PLAN-Lab/VTAM`) | **OmniVTA** (`MrSecant/OmniVTA`) | **UMI-FT** (`real-stanford/UMI-FT`) | **PAEC (本仓库 V4.4–V4.8)** |
|---|---|---|---|---|---|
| **物理输入与频率** | 120 Hz 腕部 F/T + 30 Hz 双指视触觉 | 仅原始视触觉图像（跳过触觉预训练） | 30 Hz 视觉 + 触觉图像（`OmniViTac` 21k+ 轨迹） | 双指 **360 Hz `CoinFT`** 6-DoF 力觉 + 腕部视觉 | 20 Hz 局部因果 6D F/T（绝对力/差分）+ 已执行动作 |
| **核心骨干与复杂度** | 空洞因果 1D CNN 压缩 F/T $\to$ 潜触觉世界模型 + Flow Matching | 冻结视频 VAE/DiT + 单步动作专家 + 光流-力代理 | 双流 30 Hz VTWM + 60 Hz 反射式控制器 RLTC | 360 Hz 高频自适应柔顺策略（无世界模型） | 2 层因果 MLP-GRU（`~902k`），**精确 $O(1)$ 递推**（CPU P99 `10.93 ms`） |
| **物理显式解耦** | 无 $b_\phi + g_\phi$ 零动作分解 | 无显式力学预测，依赖视觉形变流代理 | 无 $b_\phi + g_\phi$ 零动作分解（隐空间混合） | 直接策略监督映射，无前向世界模型 | **$\widehat{\Delta y} = b_\phi(z) + g_\phi(z, a)$**，严格 $g_\phi(z, \mathbf{0}) \equiv \mathbf{0}$ |
| **传感器漂移鲁棒性** | 默认传感标定恒定，未检验零漂 | 对无形变刚体接触及热/零漂脆弱 | 依赖视觉-触觉同域一致性，未设零漂隔离 | 需高频硬件标定与清零，无编码器抗漂移 | **V4.8 实证解耦**：揭示不变性-预压权衡；`PureDiff/WideDrift` 抗静态与阶跃零漂 |
| **刚度/缺陷读出** | 需预测未来触觉帧辅助规划，无虚拟探针 | 无内部刚度/硬接触定量读出 | 有潜空间惊奇度 $z_{\text{pred}} - z_{\text{obs}}$ 误差驱动 | 输出动作含刚度指令，**不能反向测算未知环境刚度** | **零抖动 18D/1D 虚拟刚度探针**（$r = +0.9255$, `battery_scan` `AUROC = 0.9767`）+ 物理惊奇度 $r_t$ |
| **动作空间与闭环层** | Flow Matching 动作块 | 单步流匹配动作专家 | **60 Hz 潜误差驱动反射式残差修正** | **位姿 + 内抓力 + 目标刚度**（柔顺动作空间） | 接入 `ACT`（位置块）或 4 阶段 `FSM` 原语 |
| **对 PAEC v2 的关键启发** | **多速率空洞因果 1D CNN** 提升至 `120–360 Hz` 而不爆显存 | 反证“重型预训练 vs 物理锚定轻量核”在硬/脆接触场景的价值 | **直接借鉴其误差驱动快反射环路**，将 PAEC $r_t$ 接成高频残差修正器 | **直接升级下游 ACT 动作空间**为“位姿+力/刚度柔顺指令” | 守住 $O(1)$、$b_\phi + g_\phi$、零漂不变性与零抖动虚拟探针护城河 |

---

## 三、冷静判断：哪里是红海，哪里是高价值蓝海？

### ❌ 1. 极度拥挤的红海（不建议硬卷）
- **单纯拼“RGB + GelSight + 大型流匹配/视频扩散世界模型”**：2026 下半年已有 `VT-WAM`、`ViTacWorld`、`DexTacWAM`、`N0-TWAM`、`VT-WM`、`OmniVTA` 同台扎堆。这类工作极其依赖海量双目+视触觉真机采集（数万条轨迹）和百卡级生成模型训练，算力与数据性价比极低，且容易被审稿人质疑与现有 `WAM` 系列同质化。

### ✅ 2. 三个真正稀缺、且能从 PAEC 自然长出来的蓝海切入点

1. **方向一：非人类无损检测传感（Active NDT）+ 动作条件物理世界模型（最贴合“超越人类感官”）**
   - **科学空白**：现有工业无损检测（超声测厚/裂纹、红外热激励、涡流、接触声发射/根部微应变）大多是**静态被动扫描**；而现有机器人世界模型只看 RGB 和指尖凝胶形变。
   - **切入方式**：延续 PAEC 的“动作激发物理响应”核心思想——控制探头以特定接触压力与微滑移轨迹扫过工件，用 `Action-Conditioned` 世界模型联合预测**力觉 + 超声耦合回波衰减 / 热传导瞬态 / 根部高频应变变化**。人类即便戴着手套也感知不到电芯内部析锂鼓包或焊缝微裂纹，但机器人的超人类传感通道 + 动作条件物理编码器能直接将其显式化。

2. **方向二：跨设备标定与传感器零漂不变的“抗漂移世界模型”**
   - **科学空白**：检索可见，2026 年这批 `VT-WAM / TacForeSight / ViTacWorld / OmniVTA` 几乎**全部默认传感读数无零点漂移、且跨工况标定恒定**。而我们在 PAEC V4.8 已经实证：哪怕仅有 `+0.5σ` 温漂，无漂移解耦的端到端力觉时序模型就会从 `59.7%–69.4%` 暴跌至 `4.2%–9.7%`；同时还存在**不变性与稳态预压可观测性的内在权衡**。
   - **切入方式**：研究如何在多模态（力觉 + 视触觉/超声/热像 + 根部应变）下通过**交叉模态物理约束打破这一权衡**（例如用对温度零漂不敏感的光学凝胶微形变或高频应变脉冲作为物理锚点，在线自标定 `F/T` 传感器的静态零漂），这是审稿人极买账的控制-感知交叉硬问题。

3. **方向三：训练期重型多模态世界模型 $\to$ 部署期 $O(1)$ 物理核 + 惊奇度触发虚拟探针**
   - **科学空白**：`VT-WAM` / `DexTacWAM` / `Cosmos` 虽能生成未来帧，但难以在边缘 CPU 上以 `>100 Hz` 稳定控制；而单纯删掉世界模型又会失去如 V4.6/V4.8 验证过的零抖动刚度诊断能力。
   - **切入方式**：在训练期利用多模态物理世界模型生成反事实故障交互（极端偏心、高刚度撞击、粘滑断裂）并蒸馏至轻量 `PAEC` 物理状态核；真机部署期常规阶段走 `O(1)` 策略直出，仅当物理残差惊奇度 $r_t$ 超阈或进入关键质检段时唤醒虚拟探针做主动安全降级与缺陷诊断。

---

## 四、PAEC v2 可执行研发路线（从当前成果到下一代突破）

### 1. 当前第一优先级：收口现有 `PAEC` 主线投稿
- 严格保留已完成 V4.1–V4.8 全套审计的 PAEC 论文（[PAEC_T_ASE_DRAFT.md](PAEC_T_ASE_DRAFT.md)）边界与负结果，优先在物理机械臂上补齐核心真机闭环验证并正式投出 `IEEE T-ASE` / `T-MECH`（不在本篇强塞超声/红外等未做实验的新模态，仅在 Section VII 展望）。

### 2. 下一代 `PAEC v2` 四项关键架构升级（直接吸收 2026 前沿精华）
1. **多速率因果压缩（借鉴 `TacForeSight`）**：用轻量空洞因果 1D CNN + $O(1)$ 状态更新将 PAEC 从 `20 Hz` 拓展至 `120–360 Hz`（兼容 `CoinFT` / 高频腕部 F/T / 根部应变）。
2. **误差驱动的快反射闭环（借鉴 `OmniVTA` RLTC）**：保留慢层视觉-触觉 ACT/FSM（`10–20 Hz`）输出宏观名义轨迹，在其下游并联一个 `60–200 Hz` 轻量残差反射环，直接以 PAEC 物理残差惊奇度 $r_t$ 与虚拟刚度 $k^{\text{virt}}_z$ 调制退让与阻尼，彻底根治 `auto_connector` 下的浅层早退与位置 chunking 滞后。
3. **柔顺力-刚度动作空间（借鉴 `UMI-FT`）**：将下游控制输出从纯位置增量升级为“目标位姿 + 期望法向力/内抓力 + 柔顺刚度”。
4. **用 `HIL-SERL` 残差 RL 突破“人类示教上限”（针对纯标注多模态+ACT 的天花板）**：
   - 单纯用“标注过的超人类多模态数据集训练大脑 + 下游接纯 ACT 行为克隆”会撞上**示教者感官与反应延迟上限**（人类看不到超声波回波或 5 ms 微滑移，示教轨迹里就没有对应的超人类微动作，BC 策略必然忽略或无法超越人类）。
   - **解法**：仅用 ACT/FSM 做安全冷启动，随后引入类似 `rail-berkeley/hil-serl` 的在线残差强化学习，**直接将超人类感知读出量（如超声耦合信噪比最大化、接触微观滑移趋零、内部应力峰值受限）写入 RL 奖励与安全约束**，让策略自主学会人类示教者自身也做不出的微米级物理探触动作。
