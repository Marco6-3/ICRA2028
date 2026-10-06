# FlexiTac Spatial Calibration Experiment

本目录是 ICRA2028 项目的第一套自采数据协议，目标不是直接训练 VLA，也不是证明 11D 一定优于所有表示，而是建立一个可重复的 **Why Spatial?** 受控实验。

核心问题：

> 在总接触载荷近似匹配时，二维触觉空间信息是否能够区分不同接触位置 / 倾角？最小需要多少 spatial state？

当前比较对象：

- Scalar: total-load proxy `P`
- Scalar + centroid: `[P, cx, cy]`
- Physics11D
- Learned11D
- Raw tactile map / reasonable spatial encoder

文件：

- `PROTOCOL.md`：完整实验协议、变量、步骤、Go / No-Go。
- `PILOT_MATRIX.csv`：第一天先跑的 24 条试验。
- `EXPERIMENT_MATRIX.csv`：pilot 通过后正式采集的 108 条试验。
- `METADATA_SCHEMA.md`：每条 trial 必须保存的数据字段与目录约定。

原则：

1. 每条 trial 保存完整时间序列，不只保存稳定帧。
2. 位置 / 倾角真值必须来自机器人或固定工装，不从 FlexiTac 自己生成。
3. 载荷匹配只用于构造受控条件，不能把同一传感器计算出的标签当作“独立真值”证明自己。
4. 在 test 结果出来前冻结 preprocessing、load tolerance、有效窗口和排除规则。
5. pilot 失败时先修数据采集和同步，不进入 Mamba。
