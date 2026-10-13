# 支撑/预紧与夹持初态检查

| 组 | 加载试次 | 初始化拒绝 | 满足初态标准的记录 |
|---|---:|---:|---:|
| removed_normal-force | 0 | 18 | 0 |
| removed_open | 18 | 0 | 0 |
| removed_proxy | 0 | 18 | 0 |
| table_normal-force | 18 | 0 | 0 |
| table_open | 18 | 0 | 0 |
| table_proxy | 0 | 18 | 0 |

初态标准同时要求目标两指的 SDF/PhysX 接触、零环境支撑法向力、相对位移≤5 mm。该标准用于开发夹持初态；加载后的满足数不是正式 Anti-Tilt 控制成功率。

SDF proxy 反馈可能达到接触门限但达不到目标响应，在闭合上限处仍被拒绝；这与“完全无接触”不同，分别记录 ever_contact_gate 和 feedback_target_reached。

过滤接触力只包含相应法向接触通道，不是完整含摩擦的六轴 wrench；另保存 table collider 关闭状态。未接触的试次或掉到接物地面不计为纯夹持。

改变初始放置与闭合量是物理场景初始化，没有写入任何预期压力/CoP 响应。数据、哈希、逐试次失败和全部门限历史均保留。
