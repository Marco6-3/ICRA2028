# E1-S 本轮交付与验收记录

2026-10-13 完成扰动加载、四指数据采集和开发 pilot。它是工程链路交付，完整 E1 科学可观测性、E2 和 E3 均未宣告通过。

## 已运行与可复核证据

| 运行 | 状态 | 规模 | 用途 |
|---|---|---|---|
| `outputs/pilot_tangent_02` | manifest completed，analysis validated | 45 条件 × 3 恢复重复，12150 记录物理步 | 切向轴、偏心、预载指令、摩擦、曲线与 SDF 读出增益 pilot |
| `outputs/coordinate_check_01` | manifest completed，analysis validated | 24 试次，2160 记录物理步 | 世界/物体/旋转参考系四方向各两次 |
| `outputs/smoke_02` | manifest completed | 3 试次，270 记录物理步 | 初始接触和力加载 smoke；早于完整校验版本 |
| `outputs/smoke_01` | failed | 0 完成试次 | 位姿字段读取错误，保留记录 |
| `outputs/pilot_world_01` | failed | 0 完成试次 | 偏心力矩动态校验发现重置 COM 缓存问题，已修正 |
| `outputs/pilot_tangent_01` | failed | 0 完成试次 | float32 参考轴未满足严格正交检查，改为 float64 正交化 |

正确版本的 runner/disturbance 文件哈希与主 pilot provenance 完全相符。原作者 `aloha_tactile_env.py` 和 SDF sensor 源文件哈希未改变；原仓库已有的两项 Windows 兼容修改保留。原 starter 与回放采集文件未覆盖；GitHub 冻结稿只读下载到 references，没有推送修改。

## 工程验收

| 检查项 | 本轮状态 | 证据 |
|---|---|---|
| 独立坐标加载正负 Fx/Fy 与偏心矩 | 通过 | 24 个坐标系物理试次；质量/速度及角动量独立校验 |
| 四指 12×32 Raw 与逐指 MCF | 通过 | 每 trial `[90,4,12,32]` / `[90,4,8]`；保存初始 Raw，因果重算误差 0 |
| 外部力/COM 力矩、物体状态与同步时间 | 通过（仿真激励真值） | 每步保存加载时刻与观测时刻、COM、物体速度/位姿、网格几何、PhysX 接触合力 |
| 固定条件重复与加载一致性 | 通过工程重放检查 | 主 pilot 三次恢复重复；载荷/物体位置/Raw 在保存精度下完全一致 |
| 方向混淆、重构误差与失败案例 | 通过输出检查 | 全部逐指/联合混淆、留出因素、反向时段、ridge 激励误差、150 条误判、零扰动对照 |

135 个主数据 NPZ 的 SHA-256、有限值、形状、差分、同步时间以及加载数学全部通过。另有 24 个坐标试次通过相同核验。四项加载数学测试通过；原 starter 四项 MCF 测试在其目录下通过。此前从工作区根调用 starter 测试造成 import 路径失败，已改正确工作目录重跑，不计入算法失败。

工程检查 5/5 不表示科学假设成立。外部 wrench 是加载器施加的激励，校验来自独立物体动力学；逐指净接触力另存，未拥有真实六轴 F/T 或完整接触力矩真值。

## pilot 发现及解释边界

基准阶跃条件，Scalar / 静态状态 / MCF / 归一化 Raw 的方向准确率为 66.7% / 66.7% / 75.0% / 83.3%。只有 12 个诊断试次，且与训练轨迹完全重复，不能把这些数值或 bootstrap 区间作为独立泛化或 MCF 显著优势的证据。

留出条件中，MCF 在预载 / 摩擦 / SDF 读出增益 / 渐变 / 反向诊断上分别为 50.0% / 37.5% / 100.0% / 0% / 50.0%。这反映当前固定均值探针对条件与时序的敏感性；不等于 GRU 的泛化结论，更不等于正常柔性传感器的物理结论。预载指令更松的一组出现某目标指净法向接触投影均值为 0，而 SDF 仍非零，两个模型读出不能互相替代。

四指 MCF 拼接为 32D，Raw 拼接为 1536D。MCF 的 fN 是 SDF proxy，CoP 轴为网格轴；切向激励轴由指尖网格平面冻结到世界坐标。世界 X 在当前保持姿态接近法向，因此主 pilot 没有误把它称作切向。

本场景包含桌面与另一个物体支撑；材料仍是刚体接触，SDF stiffness sweep 只改读出增益。不同几何入口已实现，但本轮没有做几何 sweep，也没有实现真实柔性层刚度、层厚和迟滞的物理 sweep。完整 H-Obs 必须继续用独立 FlexiTac 标定检验。

## 下一阶段保留

[PROTOCOL.md](PROTOCOL.md) 完整保留 E1 正负/偏心/预载/曲线/材料/几何条件、真实漂移迟滞与独立会话，E2 四组主比较和 representation × memory 交互，E3 Fixed/Scalar/MCF、High-Grip/Dense/Oracle、Δg 专家训练及 effort/延迟指标。正式实验样本量、载荷范围、阈值与统计成功标准仍待下一版冻结。本轮没有训练或部署控制器。

主要报告：[主 pilot](outputs/pilot_tangent_02/REPORT.md)。机器可复核结果：[analysis.json](outputs/pilot_tangent_02/analysis.json)。原始 trial 与同步几何/物理状态保留在同目录的 NPZ 中。
