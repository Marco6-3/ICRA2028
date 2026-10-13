# E1-S 偏心扰动加载、四指采集与 pilot

入口协议：[PROTOCOL.md](PROTOCOL.md)。本目录保留原 ALOHA + WarpSdfTactileSensor，新增物理外力/力矩，不修改触觉响应来匹配预期结论。研究依据保存在 `references/`，原作者仓库提交为 `dc2e1c6607a1b1a0e4b196bb4e2d4ccdc1afdcb0`。

本轮运行及工程验收：[DELIVERY.md](DELIVERY.md)。主 pilot 完成 135 试次 / 12150 步，另有 24 个坐标系物理检查试次；报告同时保留条件泛化失败与确定性重复的限制。

后续仿真/离线开发交付：[NEXT_DELIVERY.md](NEXT_DELIVERY.md)。包含不重跑旧仿真的因果探针、支撑与预紧对照、无支撑初态/加载检查，以及 33 次 E2 开发训练、3 次历史权限校正和3 次 Static MCF 输入对照；全部负结果保留。

## 后续开发入口

在工作区根目录运行，所有 `--out` 必须采用新目录：

```powershell
& 'C:\isaacsim_env\Scripts\python.exe' 'isaac_windows\e1s\temporal_probe.py' --source 'isaac_windows\e1s\outputs\pilot_tangent_02' --out 'isaac_windows\e1s\outputs\new_temporal'
& 'C:\isaacsim_env\Scripts\python.exe' 'isaac_windows\e1s\e2_dev.py' --source 'isaac_windows\e1s\outputs\pilot_tangent_02' --out 'isaac_windows\e1s\outputs\new_e2'
```

无支撑加载的可复核命令（先确认原作者环境可用并接受其 EULA）：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
& 'C:\isaacsim_env\Scripts\python.exe' 'isaac_windows\e1s\contact_pilot.py' --out 'isaac_windows\e1s\outputs\new_pure_loading' --fixture-npz 'isaac_windows\e1s\outputs\contact_01\table_open\c000_r00.npz' --matrix 'isaac_windows\e1s\contact_matrix.json' --support removed --preload-mode open --initial-close-rad .003 --load-frame reference --reference-from-finger 2 --headless --device cpu --kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false
& 'C:\isaacsim_env\Scripts\python.exe' 'isaac_windows\e1s\analyze_contact.py' 'isaac_windows\e1s\outputs\new_pure_loading'
```

这是固定夹持开发加载，不是 E3 反射控制。原 `run_pilot.py` 仍是原 pilot 的复现入口；`contact_pilot.py` 单独包含支撑/初始化干预。旧版本脚本快照用于对照来源哈希，运行入口优先上述当前脚本。

## 运行

在工作区根目录的 PowerShell 中：

简化入口：`& '.\isaac_windows\e1s\launch_e1s.ps1' -Mode pilot`。它创建新输出目录，保存日志，并检查 manifest/analysis 后才报告通过。`-Mode coordinates` 运行坐标系专项；`-Mode analyze -OutputDirectory <已有目录>` 只分析。运行数据不能仅靠进程退出码判断。

```powershell
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:PYTHONUNBUFFERED = '1'
& 'C:\isaacsim_env\Scripts\python.exe' 'isaac_windows\e1s\run_pilot.py' --out 'isaac_windows\e1s\outputs\new_pilot' --load-frame reference --reference-from-finger 2 --headless --device cpu --kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false
& 'C:\isaacsim_env\Scripts\python.exe' 'isaac_windows\e1s\analyze_pilot.py' 'isaac_windows\e1s\outputs\new_pilot'
```

输出目录必须是新目录。原始 replay/starter 不覆盖。每步 1/120 s，共 135 个开发试次，各 90 步；一次启动内完成，使用这台机器已验证的 CPU 物理与 Warp 计算设备。

`--matrix path.json` 可提供 Trial 字段列表，修改幅度、曲线、偏心位置、预载、摩擦或读出增益；矩阵在启动前写入输出。`--load-frame world/object/reference` 指定加载坐标，`--reference-wxyz w x y z` 指定独立参考旋转；`--reference-from-finger` 则从当前保持姿态的网格平面冻结坐标。`--plug-scale` 与 `--normal-offset` 是不同场景几何入口，改变它们后需重新验证接触；normal offset 是采样面偏移，不等于模拟真实层厚。

输出每 trial 的 NPZ/JSON、`matrix.json`、`provenance.json`、独立引擎载荷校验、`manifest.json`。分析生成 `analysis.json`、中文 `REPORT.md` 和 `diagnostics.png`。退出码不能代替完成核验：Isaac App 关闭可能掩盖异常，必须检查 manifest 和 analysis 状态。

## 证据边界

每指 MCF 是 8D，四指拼接为 32D。Raw 为四指各 12×32。SDF 为几何穿透的归一化读出，读出 stiffness 不等于柔性材料接触刚度；未模拟真实迟滞。外力/COM 力矩是独立加载激励，不是阵列推断的接触 F/T。原装配场景还有桌面与物体支撑，pilot 不构成真实 FlexiTac 剪切机理或 Anti-Tilt 闭环结论。

原 E1 交付没有 GRU 控制器。当前追加的 Reflex 原型已经包含专家采集、MCF/Scalar 学生训练和实际仿真闭环，见下面的新入口。完整 E2 四组比较及真实 FlexiTac E1 仍按原协议推进。

## 归一化修复与 Reflex 闭环原型

当前交付见 [REFLEX_DELIVERY.md](REFLEX_DELIVERY.md)。0.18 N 四方向偏心载荷下，MCF Reflex 最坏相对滑移 1.013 mm，相比 High-Grip 平均法向力积分降低 33.9%；0.20 N 的未达标条件也完整记录。

- `bounded_scaler.py`：训练集 median/IQR、各通道量纲下限和 [-8,8] 截断，避免近常量训练通道将正常变化放大数千倍。
- `bounded_probe.py`：复用 pure_loading_01，不重新模拟；同预算比较修复前后 Fx/Fy 误差与方向准确率。
- `reflex_sim.py`：保留原 ALOHA 和 WarpSdf 模型，新增仅作用于抓握的 Δg，记录独立物理位姿、力、输入、教师标签与实际动作。
- `train_reflex.py` / `train_reflex_stream.py`：滚动窗口第一版和持续 GRU hidden 第二版；均包含 MCF 与 Scalar，同数据、三 seeds、独立条件验证。
- `report_reflex.py`：成对初态/载荷核验、因果输入核验、逐条件结果及滑移/倾角/法向力/动作曲线。
- `REFLEX_PROTOCOL.md` / `REFLEX_STREAM_PROTOCOL.md`：开发选择、训练预算、数据划分与两版冻结测试矩阵。

完整重放入口（必须指定新的输出目录）：

```powershell
& 'C:\isaacsim_env\Scripts\python.exe' 'isaac_windows\e1s\run_reflex_pipeline.py' --out 'isaac_windows\e1s\outputs\new_reflex_reproduction'
```

该入口顺序运行归一化对照、专家采集、第一版训练/闭环、学生状态开发集、流式训练/闭环和报告，每次 Isaac 运行均核验 manifest 后才继续。原始 NPZ、历史报告和旧源码哈希对应的快照保留。
