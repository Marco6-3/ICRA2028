# Metadata and File Schema

## Directory layout

建议：

```
experiments/flexitac_spatial/data/
  session_YYYYMMDD_HHMM/
    session.json
    metadata.csv
    raw/
      <trial_id>.npz
    processed/
      <trial_id>.npz
    calibration/
      baseline.npz
      load_setpoints.json
```

原始数据目录默认不提交 Git；只提交 protocol、schema、分析脚本和小型汇总结果。大文件按项目实际存储方案管理。

## Trial ID

格式：

```
<block>_x<signed-x>_th<signed-theta>_L<level>_R<repeat>
```

例如：

```
S2_x-050_th+000_LM_R03
S4_x+000_th-010_LM_R02
```

x 使用归一化位置乘 100，例如 -0.50 -> -050。

## metadata.csv required columns

- trial_id
- session_id
- block
- planned_order
- executed_order
- repeat
- x_norm_target
- y_norm_target
- theta_deg_target
- load_level
- load_target_value
- load_match_tolerance
- indenter_id
- surface_id
- sensor_id
- firmware_version
- sampling_rate_nominal_hz
- robot_or_fixture_id
- start_time_host
- valid
- invalid_reason
- operator_note

若有独立 F/T，再加：
- ft_sensor_id
- ft_calibration_id

## raw NPZ

建议数组：

- tactile_raw: [T,H,W]
- tactile_processed: [T,H,W]
- sensor_timestamp: [T]
- host_receive_timestamp: [T]
- robot_timestamp: [Tr]（若有）
- robot_pose: [Tr,D]（若有）
- ft_timestamp: [Tf]（若有）
- ft_wrench: [Tf,6]（若有）

不要把不同采样率信号提前粗暴 resample 后覆盖原始流。原始时间戳必须保留。

## processed NPZ

可派生：

- P
- cx
- cy
- active_area
- sigma1
- sigma2
- orientation_x
- orientation_y
- delta_P
- delta_cx
- delta_cy
- contact_flag
- saturation_fraction
- dropped_frame_flag

所有 preprocessing 参数必须写入同一 session 的 config / json，不允许只存在代码默认值里。

## session.json

至少：

- date
- git_commit
- protocol_version
- sensor hardware / firmware
- active taxel shape
- effective width / height
- indenter geometry
- fixture geometry
- coordinate convention
- load setpoints L/M/H
- baseline rule
- threshold rule
- filtering rule
- operator
- environment notes
- known anomalies

## Invalid-trial rule

invalid 只能由预先定义的采集 QC 触发，例如：

- load mismatch > 5%
- target pose not reached
- sensor disconnect / packet loss
- saturation above frozen threshold
- indenter leaves active area
- synchronization failure

不能因为“模型预测错了”把 trial 标 invalid。
