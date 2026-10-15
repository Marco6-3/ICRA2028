# 2026-10-14 机制猜想验证与偏心抗扰相图交付

已实际完成644条受控仿真、115920帧同步记录：纯切向力/纯力矩/组合载荷26条，无外载闭合阶跃6条，四控制器×四方向×35格及零载荷564条，+Fx特权参考36条，抓握时刻干预12条。冻结模型未重训，原实验与模型哈希不变。

## 当前可用成果

- [主要判断与机制证据](outputs/phase_report_20261014_01/FINDINGS.md)
- [全部方向相图与时序对照](outputs/phase_report_20261014_01/REPORT.md)
- [四方向 Scalar/MCF 安全差异图](outputs/phase_report_20261014_01/phase_controller_contrast.png)
- [同一补偿量的抓握时刻干预图](outputs/phase_report_20261014_01/timing_intervention.png)
- [原0.20 N +Fx信号/命令/实际力对齐](outputs/phase_timing_20261014_01/timing_020_fx.png)
- [逐格表](outputs/phase_report_20261014_01/phase_table.csv) / [逐格结果](outputs/phase_report_20261014_01/phase_results.json)
- [完整复核](outputs/phase_report_20261014_01/verification.json)

## 对机制猜想的检验结果

补偿时刻确实显著影响结果：同一最大Δg、同一0.20 N / 15 mm +Fx载荷，提前100 ms握持得到1.635 mm滑移；载荷同刻补偿2.102 mm，晚25 ms为2.959 mm，晚100 ms为11.639 mm。时钟干预是特权诊断，没有冒充触觉策略。

在当前SDF/刚体接触模型里，纯切向力和纯力矩均产生早期弱的CoP/强度变化，按基线噪声门限三帧确认均为25 ms；大幅变化门限的CoP/面积则更晚或未检出。弱响应不当作没有信息，未把这个结果推广成真实FlexiTac的物理边界。

无外载的1/3 mrad闭合阶跃使实际法向力在第一个8.333 ms采样步就接近平台，10–90%两个门限在同一帧；当前没有验证出多帧慢爬升。下一步控制研发应针对弱信号触发、补偿保持与预紧状态，PD前馈可以作为独立策略消融，不能仅凭此结果声称它已解决问题。

## 安全域

F={0.10,0.125,0.15,0.175,0.20,0.225,0.25} N，e={0,5,10,15,20} mm，四方向完整扫描。安全以独立相对位移≤2 mm、有效初态、无掉落与无环境支撑定义。

| 控制器 | 安全格 |
|---|---:|
| Nominal | 86/140 |
| High-Grip | 134/140 |
| Scalar | 116/140 |
| MCF | 119/140 |

高偏心区域e≥10 mm的网格分片面积：−Fx的MCF相对Scalar增加22.2%，+Fx减少20.0%，±Fy不变；四方向合并增加2.6%。全部未达标格均保留，没有只展示有利方向。面积为四角安全的网格代理，不给未采内点连续安全保证。

两策略共同安全的112个条件中，MCF平均法向力积分较Scalar降低5.8%；全网格较High-Grip降低32.8%。但扰动前实际握持力存在策略差异：Scalar约4.42 N、MCF约3.06 N。相同初态与名义指令不等于策略运行后仍有相同预握持力，因此相图不能独立证明传感器可观测性的因果优势。

## 复现与记录

原冻结协议为 [PHASE_PROTOCOL_INITIAL.md](PHASE_PROTOCOL_INITIAL.md)，诊断补充见 [PHASE_PROTOCOL.md](PHASE_PROTOCOL.md)，时刻干预见 [TIMING_INTERVENTION_PROTOCOL.md](TIMING_INTERVENTION_PROTOCOL.md)。原协议、矩阵与模型SHA均已核验。主相图每格为一条确定性开发试次；两个恢复重放不充当独立统计样本。

在本目录用已验证的Isaac环境运行，必须指定新输出目录：

```powershell
python run_phase_suite.py --out outputs/phase_reproduction_new
python run_timing_after_phase.py --phase outputs/phase_reproduction_new --out outputs/timing_reproduction_new
python analyze_phase.py outputs/phase_reproduction_new --out outputs/phase_report_new
python report_phase_findings.py --source outputs/phase_reproduction_new --report outputs/phase_report_new --timing outputs/timing_reproduction_new
python verify_phase.py outputs/phase_reproduction_new --timing outputs/timing_reproduction_new --report outputs/phase_report_new
```

本轮源码、记录和结果位于本地仓库工作区，尚未追加到先前GitHub发布的47d5f05归档；原发布不改写。
