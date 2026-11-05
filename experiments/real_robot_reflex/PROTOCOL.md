# Real-Robot Reflex Protocol v0

状态：执行前协议草案。对应 [Stage 3A](../../research/STAGE3A_REAL_ROBOT.md)。

> **2026-11-05 执行补充（v0.1，待 Pilot 冻结）**：本轮暂不开展 RL / PPO / SAC，不接 VLA；新增 **UMI + 双指 FlexiTac 数据采集**与 **BPI / CMQ 跨 Session 表征验证**，并保留真实机械臂 Anti-Tilt / Anti-Slip 闭环。详细方案见 [UMI + FlexiTac 研究计划](../../research/REAL_ROBOT_UMI_FLEXITAC_PLAN_20261105.md)。下方原 v0 流程继续有效，尚未冻结的载荷、频率、样本量及阈值须在 Pilot 后冻结。

## 1. 实验目标

第一轮真机不接 VLA，不比较大量网络，只建立可重复的受控扰动 benchmark，并验证 Fixed / Scalar Reflex / MCF Reflex 的闭环差异。

优先顺序：

1. Anti-Tilt / eccentric disturbance；
2. Anti-Slip / pull disturbance。

## 2. Bring-up checklist

每个实验 session 开始前记录：

- robot / gripper 型号与 firmware；
- FlexiTac 版本、安装方向、采样率；
- host / GPU / controller machine；
- tactile timestamp 与 host receive timestamp；
- gripper command frequency；
- nominal grip setpoint；
- residual clamp 与 slew-rate clamp；
- emergency stop；
- test object ID；
- disturbance fixture / weights；
- 外部 F/T、相机或位移测量设备（若有）。

必须保存一张 setup photo 和 session metadata。

## 3. Trial structure

默认单 trial：

1. no-contact / baseline；
2. establish grasp；
3. stable hold；
4. disturbance onset；
5. disturbance hold / evolution；
6. recovery；
7. release。

具体秒数在首次 bring-up 后冻结，整批保持一致。

## 4. Anti-Tilt matrix

最小 pilot 建议：

- offset: left / center / right；
- load: 1 个安全的 medium level；
- controller: Fixed / Scalar / MCF；
- 每条件至少 3 次开发重复。

正式评测前再根据 pilot 的可重复性冻结 offset、load 与重复数，不在测试过程中追加“更容易出效果”的条件。

## 5. Anti-Slip matrix

最小 pilot 建议：

- pull magnitude: low / medium；
- controller: Fixed / Scalar / MCF；
- 每条件至少 3 次开发重复。

正式评测优先使用机械/砝码式可重复扰动。人工手拉只作为非定量 demo。

## 6. Controller fairness

Scalar 与 MCF 的第一版 controller 必须共享：

- nominal trajectory；
- decision rate；
- action bounds；
- history length；
- backbone family / 参数量级；
- optimization budget；
- training trial IDs；
- early stopping / model selection protocol。

如果某一项不能匹配，必须在结果表显式记录。

## 7. Data to save

每个 tactile frame / controller tick 至少保存：

- session_id / trial_id / condition；
- sensor timestamp；
- host receive timestamp；
- raw tactile frame；
- MCF features；
- scalar features；
- gripper state；
- nominal command；
- residual command；
- final command；
- model start / end timestamps；
- command-send timestamp；
- actuator state timestamp；
- contact flag；
- valid flag / invalid reason。

若有外部真值：
- object pose / tilt；
- slip displacement；
- Fx/Fy/Fz/Tx/Ty/Tz；
- disturbance onset trigger。

## 8. Safety

- 先空夹爪验证 residual sign；
- 再软物体 / 轻载；
- 再正式刚性 test object；
- 所有 residual 都有限幅和 rate limit；
- stale tactile frame 自动退回 nominal；
- no-contact 时禁止持续加紧；
- 明确硬件最大安全 command，不因追求成功率突破。

## 9. Primary analysis

正式测试集上按 trial-level 分析：

- max slip；
- max tilt；
- recovery time；
- survival / drop；
- peak / integrated grip effort；
- end-to-end latency。

不以单帧随机拆分代替 trial-level 统计。

## 10. 尚未冻结的关键项

以下内容必须在首次真实平台 bring-up 后补全，而不是现在凭空填写：

- gripper command 的物理单位；
- controller frequency；
- FlexiTac 实测稳定 sampling rate；
- residual amplitude / slew limits；
- test object 几何与质量；
- disturbance magnitude；
- tilt / slip ground-truth 测量方式；
- Scalar / MCF controller 的 supervision 或 reward 定义。

这些项目未冻结前，v0 不能被称为正式预注册 protocol。

## 11. 2026-11-05 执行分层（UMI + FlexiTac）

- **P0 集成标定**：双指 raw 阵列、实际采样率/丢帧、taxel 物理坐标和方向、接触/松开、预紧参考帧稳定性；相机、UMI 位置/宽度、tactile 的独立时间戳及同步误差。建议 Pilot 10–20 trials（预算非结果）。
- **P1 固定 UMI 受控数据**：刚性夹具固定 UMI，机械/砝码/滑轮受控加载；无扰动、±Fx、±Fy，改变预紧和独立 Session；专门记录长时保持、反向加载与卸载恢复。候选基础矩阵 `3 sessions × 3 preloads × 5 load conditions × 6 repeats = 270 trials`，额外反转/长保持另计；正式数量在 Pilot 后冻结。
- **P2 自由 UMI 示教**：采集原始双指触觉、视频、UMI 位姿/宽度与人类动作轨迹，建议试点 60–120 trials。它提供自然操作接触数据，**不自动提供真实外部力或机器人正确纠偏动作真值**。
- **P3 机器人闭环**：同一真实机械臂/触觉安装下对比 Fixed、Scalar、BPI、CMQ；Standard MCF、Raw Taxels CNN、High-Grip 补充。先检验安全且匹配权限的 1D gripper 控制，经过独立安全与公平验证后才考虑统一的 3-DoF。

### 11.1 离线表征与样本效率分析

- 输入依次为：真实 **Raw Taxels**（全阵列 + 合理 CNN/MLP/GRU）、Scalar、Standard Sliding-Window MCF、BPI-MCF、CMQ-BPI-MCF。历史 `Raw_State4_16D` 不能冒充 Raw Taxels。
- 从**同一原始 trial**计算各表征，匹配时间窗、训练/开发数据、超参数搜索预算及预测目标；分别报告模型参数、编码耗时、推理耗时。
- 按完整 `session_id` / `trial_id` / 工况划分，封存新 Session 测试集；训练内部 10% / 25% / 50% / 100% 数据比例及多次抽样/训练种子。不能对完整 test 归一化或拟合类中心后再宣称 OOD。
- 重点指标：四向即时方向/混淆矩阵、持续载荷保留率、反向载荷识别与延迟、跨 Session 表征稳定性及下游性能退化、受载/卸载事件检测、学习曲线。无外部 F/T 时不得报告外力/力矩监督精度。
- t-SNE / UMAP 只用于可视化；主结论建立在 trial-level 定量统计及置信区间上。

### 11.2 数据记录扩展

两端统一使用 `session_id, trial_id, platform, schema_version, split, sensor_timestamp, host_receive_timestamp, tactile_left_raw, tactile_right_raw, gripper_width, camera_timestamp`；可选 `pose, gripper_state, load_direction, load_magnitude, preload_level, external_wrench, object_pose, slip, tilt`；机器人端另记 `nominal_command, residual_command, final_command, t_command_sent, t_actuator_response`；标注缺失、坏点、饱和、丢帧、时间戳时钟域与无效原因。

原始数据只读；`BPI/CMQ` 输出须记录算法版本/参数与校准依赖。UMI 与机器人指尖刚度、接触表面、传感器朝向、夹爪宽度/预紧量不一致时，不把跨平台误差直接归于表征。

### 11.3 Go / No-Go 与安全

1. P0 不能稳定读数/同步 → 停止性能比较，先修硬件。
2. P1 同一外力下空间信号无法可靠区分或 CMQ 丢失反转 → 如实记录，调整物理条件/表征，再考虑训练。
3. P2 低样本不优于 Raw → 记录负结果，不宣称“10% 超过 Raw 100%”；分析部署开销和适用边界。
4. P3 如 Fixed/High-Grip 同样稳定，或只是更高夹持代价 → 不宣称控制收益；记录落差与风险。
5. 真机部署前固定残差幅值/变化率限制、无接触门、过期帧回退、watchdog、急停；手持 UMI 不运行机器人自主控制命令。

**历史研究边界**：Stage 4 / V2.2 的 score-function 代理精调不是已完成 Isaac Lab 动作驱动 PPO/SAC；本轮不以它作为真机实验的必要前提。此前仿真数据、审计与负结果保持原样。
