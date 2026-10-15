"""Create the Chinese entry point and publication text from verified outcomes."""
import json
from pathlib import Path
here=Path(__file__).resolve().parent
e1=here.parent/'stage3s_isaac/isaac_windows/e1s'
report=e1/'outputs/alignment_report_20261014_01'
results=json.loads((report/'results.json').read_text())
verified=json.loads((report/'verification.json').read_text()); assert verified['status']=='passed'
summaries=results['summaries']
lines=['# Stage 3S：等预紧握持、成本—安全关系与弱触觉触发', '',
    '实验启动：2026-10-14；整理发布：2026-10-15。本轮将原 Scalar/MCF 不同的静息握力拉平，检验扰动后的残差反馈，并开发因果 CoP 触发补偿。原模型、原相图、负结果和第一篇论文冻结范围均保留。四指逐指 MCF 为 8D，学生整体输入为 32D。',
    '', f"完成 **{verified['trials']} 个正式物理试次、{verified['synchronized_frames']} 个同步记录步**（1163 主试次 + 27 同载荷 High-Grip 辅助参照）；另有 9 个接口试次。数据哈希、初态配对、实测预紧、静息动作封锁、因果触觉输入和绝对执行器限幅通过核验。",
    '', '## A：3.06 / 4.42 N 等实测预紧力', '',
    '| 预紧 | Fixed 安全格 | Scalar 安全格 | MCF 安全格 | Scalar / MCF 平均力积分 N·s |', '|---|---:|---:|---:|---:|']
for level in ('low','high'):
    f,s,m=[summaries[level+'_'+kind] for kind in ('fixed','scalar','mcf')]
    lines.append(f"| {f['pre_normal_n']:.3f} N | {f['safe_cells']}/140 | {s['safe_cells']}/140 | {m['safe_cells']}/140 | {s['mean_effort_n_s']:.3f} / {m['mean_effort_n_s']:.3f} |")
gaps=[summaries[level+'_mcf']['safe_cells']-summaries[level+'_scalar']['safe_cells'] for level in ('low','high')]
costs=[100*(summaries[level+'_mcf']['mean_effort_n_s']/summaries[level+'_scalar']['mean_effort_n_s']-1) for level in ('low','high')]
lines+=['',f'等预紧后，MCF 在两级分别比 Scalar 多 {gaps[0]} / {gaps[1]} 个安全格。低预紧 Scalar 未发生整体安全格崩盘。MCF 的额外安全收益伴随全网格平均力积分增加 {costs[0]:.2f}% / {costs[1]:.2f}%；这是闭环安全—代价权衡，不能把原模型的力积分优势直接套到等预紧结果。']
lines+=['', '加载前固定相同夹爪位置并更新 GRU 历史，加载时才释放 π(obs_t)−π(obs_pre)。共同绝对闭合权限为原名义指令外 [0,3] mrad。结果检验这一冻结学生部署方式，包含基线扣除后的可达输出范围与零载漂移记录。',
    '', '## B：原模型的成本优势与新静态参照', '',
    '原模型 MCF 为 119/140，Scalar 为 116/140；MCF 静息握力低 30.8%，在两者共同安全的 112 格中法向力积分低 5.76%。原四点显示 MCF 对 Scalar 的样本非支配优势；等预紧后的结果另列，避免把两批条件合并成一个机制结论。',
    '', '![原模型成本—安全关系](../stage3s_isaac/isaac_windows/e1s/outputs/alignment_report_20261014_01/pareto_original.png)',
    '', '![等预紧后的成本—安全关系](../stage3s_isaac/isaac_windows/e1s/outputs/alignment_report_20261014_01/pareto_aligned.png)',
    '', '方形为固定握持，圆形为反馈策略，黑边为当前样本的非支配点。Scalar 是反馈策略；静态参照由三个实际 Fixed 网格构成。',
    '', '## C：早期脉冲与锁存', '',
    '| 条件 | 安全格 / 24 | 平均力积分 N·s | 平均绝对饱和时长 s | 零载触发时刻 s |', '|---|---:|---:|---:|---:|']
for name,s in summaries.items():
    if name.startswith('trigger_'):
        zero=next(r for r in results['zero_load_controls'] if r['group']==name)
        alarm='无' if zero['trigger_time_s'] is None else f"{zero['trigger_time_s']:.4f}"
        lines.append(f"| {name} | {s['safe_cells']} | {s['mean_effort_n_s']:.3f} | {s['mean_saturation_seconds']:.3f} | {alarm} |")
anchor='trigger_pFx_f0.200_e0.015'
case={name:next(r for r in results['rows'] if r['group']==name and r['id']==anchor) for name in
    ('trigger_low_none','trigger_low_pulse','trigger_low_latch','trigger_max_fixed')}
base,pulse,latch,high=[case[name] for name in ('trigger_low_none','trigger_low_pulse','trigger_low_latch','trigger_max_fixed')]
reduction=100*(1-pulse['effort_normal_n_s']/high['effort_normal_n_s'])
false_alarms=[r for r in results['zero_load_controls'] if r.get('trigger') in ('pulse','latch') and r['trigger_time_s'] is not None]
pre_load={level:sum(r['normal_immediately_before_load_n'] for r in results['rows'] if r['group']==f'trigger_{level}_none' and r['trial']['direction']!='zero' and r['repeat']==0)/24 for level in ('low','high')}
timing_note=f"C 的 3.06/4.42 N 指校准静息目标。为检查未知加载起点，GRU 在 0.45 s 释放、外扰在 0.55 s 加载；加载前 75 ms 窗口的实测中位力分别约 {pre_load['low']:.3f}/{pre_load['high']:.3f} N。同级三个变体在真实触发前共享这段轨迹。A 的加载与释放则均为 0.45 s，维持其严格等预紧比较。"
zero_note='低预紧脉冲与锁存的零载对照均未触发。'
if false_alarms:
    zero_note+='高预紧存在零载误触发：'+ '、'.join(f"{r['group']} 于 {r['trigger_time_s']:.4f} s" for r in false_alarms)+'；现有门限尚不能可靠区分外部扰动与闭环自身变化。高预紧触发结果不能据此称为稳健扰动检测。'
lines+=['',f"重点条件 +Fx / 0.20 N / e15 mm：低预紧 GRU 原型 {base['max_slip_mm']:.3f} mm；早期脉冲 {pulse['max_slip_mm']:.3f} mm（加载后 {pulse['trigger_delay_ms']:.2f} ms 触发）；三帧确认锁存 {latch['max_slip_mm']:.3f} mm。脉冲相对同载荷 High-Grip 的法向力积分降低 {reduction:.2f}%，绝对最大闭合指令保持 {pulse['saturation_seconds']:.3f} s，High-Grip 为 {high['saturation_seconds']:.3f} s。",
    '', '这说明早期短脉冲在该重点条件中守住了 2 mm，并保留了回落的握力曲线。剩余失败格与锁存未达标条件完整保留，不将这个成功推广为整个矩阵全部达标。', '',zero_note,'',timing_note]
lines+=['', '独立矩阵覆盖 ±Fx、0.18–0.25 N、10/15/20 mm 偏心。触发只使用上一帧触觉与静息噪声统计；门限为 6×稳健噪声尺度（坐标尺度 1.4826 MAD，合成二维范数，另设 1e-6 读出下限），保留零载误触发检查。单帧脉冲保持 25 ms，三帧确认锁存保持 250 ms 后衰减；每试次最多触发一次。',
    '', '![困难格点的位移、握力与指令](../stage3s_isaac/isaac_windows/e1s/outputs/alignment_report_20261014_01/trigger_anchor.png)',
    '', '## 报告与数据', '',
    '- [完整结果、逐格差异与限制](../stage3s_isaac/isaac_windows/e1s/outputs/alignment_report_20261014_01/FINDINGS.md)',
    '- [全部计算结果](../stage3s_isaac/isaac_windows/e1s/outputs/alignment_report_20261014_01/results.json) / [核验记录](../stage3s_isaac/isaac_windows/e1s/outputs/alignment_report_20261014_01/verification.json)',
    '- [A/B/C 冻结协议](../stage3s_isaac/isaac_windows/e1s/ALIGNMENT_PROTOCOL.md) / [辅助静态参照协议](../stage3s_isaac/isaac_windows/e1s/TRIGGER_STATIC_REFERENCE_PROTOCOL.md)',
    '- [原相图与机制诊断](../stage3s_isaac/isaac_windows/e1s/PHASE_DELIVERY.md) / [时序干预协议](../stage3s_isaac/isaac_windows/e1s/TIMING_INTERVENTION_PROTOCOL.md)',
    '- [原始 NPZ 同仓库 Release](https://github.com/Marco6-3/ICRA2028/releases/tag/stage3s-phase-alignment-20261014) / [逐文件与 ZIP 哈希](raw_assets.json)',
    '', '本次发布一并收录前一轮 644 个物理试次、当前正式与接口试次、全四指原始观测以及所有加载校验数据。Git 保存协议、源码、模型引用、分组元数据、完整汇总表和图表；逐试次 JSON 与 NPZ 保持原字节并以 Release ZIP 提供。历史交付中“未上传”等句子是对应日期的记录，本入口指向本次发布。',
    '', '## 离线复核与仿真重放', '',
    '先克隆完整仓库，需要 Python 3.11、NumPy、PyTorch、Matplotlib；下载需要 GitHub CLI。于本目录执行：', '', '```powershell',
    'python ../stage3s_isaac/restore_data.py --download', 'python restore_data.py --download',
    'python ../stage3s_isaac/isaac_windows/e1s/analyze_alignment.py ../stage3s_isaac/isaac_windows/e1s/outputs/alignment_20261014_01 --report ../stage3s_isaac/isaac_windows/e1s/outputs/new_alignment_analysis',
    'python ../stage3s_isaac/isaac_windows/e1s/test_aligned_policy.py', '```', '',
    '下载器检查 ZIP、完整成员清单和 NPZ 哈希，不覆盖内容不同的现有文件。也可以手动下载 ZIP 后使用 `--archive-dir`。',
    '', '重跑物理实验需要原作者环境、Isaac Sim 5.1、CPU Physics 和 Windows 兼容补丁，获取方法见[原 Stage 3S 入口](../stage3s_isaac/README.md)。本轮原生运行使用 `C:/isaac_ws/FlexiTac-IsaacSim-Simulation` 与复原的夹持快照及原始轨迹输入。',
    '', '```powershell',
    'python ../stage3s_isaac/isaac_windows/e1s/run_alignment_suite.py --out ../stage3s_isaac/isaac_windows/e1s/outputs/new_alignment_reproduction', '```',
    '', '输出目录必须为新目录；使用已提交的矩阵与冻结模型，无需重新生成或覆盖冻结矩阵。辅助参照的原始命令保存在 `run_trigger_static_reference.py`；它仅在主序列结束后运行。',
    '', '这是固定夹持快照上的开发仿真：2 mm 指标是独立物体—指尖相对位移，N 来自物理接触法向投影，SDF 强度仍是读出代理。每格一次确定性运行与两个重复锚点不代表统计总体泛化；真实 FlexiTac 标定与真机闭环尚待硬件齐备。']
(here/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
finding_path=report/'FINDINGS.md'
finding=finding_path.read_text(encoding='utf-8').split('\n## 直接结论与零载检查')[0]
finding+=f"\n## 直接结论与零载检查\n\n同预紧 MCF 相对 Scalar 分别增加 {gaps[0]} / {gaps[1]} 个安全格，平均全程力积分增加 {costs[0]:.2f}% / {costs[1]:.2f}%。\n\n重点 +Fx / 0.20 N / e15 mm 的早期脉冲为 {pulse['max_slip_mm']:.3f} mm，相对同载荷 High-Grip 力积分低 {reduction:.2f}%；三帧确认锁存为 {latch['max_slip_mm']:.3f} mm。\n\n{zero_note}\n\n{timing_note}\n"
finding_path.write_text(finding,encoding='utf-8')
ws=here.parents[2]
body=['原相图中 Scalar 与 MCF 的静息握力不同。本次增加实测 3.06/4.42 N 等预紧对照、三组固定握持参照，以及因果 CoP 早期脉冲/锁存，保持原冻结模型和全部负结果。', '',
    f"同预紧 MCF / Scalar：118/116 与 130/126 个安全格（各 140 格）。重点 +Fx 0.20 N / e15 mm 的低预紧短脉冲从 {base['max_slip_mm']:.3f} 降到 {pulse['max_slip_mm']:.3f} mm，相对同载荷 High-Grip 的法向力积分低 {reduction:.2f}%。三帧确认锁存为 {latch['max_slip_mm']:.3f} mm，未达标；高预紧零载误触发单列保留。", '',
    f"已完成 {verified['trials']} 个正式物理试次、{verified['synchronized_frames']} 个同步记录步，另有 9 个接口试次。所有原始四指 NPZ、试次元数据、协议、源码和正负结果完整发布；上一轮 644 个物理试次一并归档。", '',
    '验证：数据 SHA256、成对初态和载荷、独立加载动力学、静息动作封锁、因果输入、绝对闭合限幅、原档案与模型字节；3 项控制逻辑测试通过。', '',
    '本轮为固定快照的仿真开发结果。等预紧、静态成本参照、零载误触发和失败格均单列；未据此宣布真机物理机制或普遍最优性。', '',
    '[完整报告](https://github.com/Marco6-3/ICRA2028/blob/experiment/preload-alignment-trigger-20261014/experiments/stage3s_alignment/README.md) · [原始数据](https://github.com/Marco6-3/ICRA2028/releases/tag/stage3s-phase-alignment-20261014)']
(ws/'alignment_pr_body.md').write_text('\n'.join(body)+'\n',encoding='utf-8')
(ws/'alignment_release_notes.md').write_text('\n'.join(body)+'\n\n每个 ZIP 及内部 NPZ 的 SHA256 见仓库 experiments/stage3s_alignment/raw_assets.json。\n',encoding='utf-8')
print('DELIVERY_WRITTEN',here/'README.md',ws)
