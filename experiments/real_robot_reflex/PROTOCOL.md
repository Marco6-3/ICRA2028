# Real-Robot Reflex Protocol v0

状态：执行前协议草案。对应 [Stage 3A](../../research/STAGE3A_REAL_ROBOT.md)。

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
