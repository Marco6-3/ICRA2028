"""Synchronize the 2026-11-04 closeout documents from completed JSON outputs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "outputs"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def main():
    off = json.loads((OUT / "v2_2_verification_results.json").read_text())
    cl = json.loads((OUT / "isaac_closedloop_v2_2_residual_rl_report.json").read_text())
    audit = json.loads((OUT / "secondary_audit_20261104.json").read_text())
    assert audit["physical_run_inputs_match_current_files"], "Isaac results must match the current source and checkpoints"
    arch = off["innovations_2_and_3_architecture_and_online_rl"]
    a, b = cl["part_a_6controller_closedloop"], cl["part_b_3stage_long_horizon_benchmark"]
    prop = "6_v2_2_online_residual_rl_proposed"
    short = dict(zip(a, ["Open-loop", "Stage 3S 1D", "V2.1 3DoF", "V2.2 No-Aux", "V2.2 WarmStart BC", "V2.2 With-Aux"]))
    baseline_keys = ["3_V22_DualHead_WarmStart_BC", "4_V22_Surrogate_Residual_RL_No_Aux", "5_V22_Surrogate_Residual_RL_With_Aux_Proposed"]
    offline_tables = []
    for split, label in [("dev_pure_c005_c008", "Dev c005–c008"), ("blind_fresh_b0_b1", "Fresh b0+b1（8 条非零试次）")]:
        rows = []
        for k, name in zip(baseline_keys, ["WarmStart BC", "No-Aux", "With-Aux"]):
            r = arch[split][k]
            rows.append([name, f'{r["clean_action_mse"]:.5f}', f'{r["occluded_action_mse"]:.5f}',
                         f'{r["wrench_r2"]:.4f}', f'{100*r["latent_probe_dir_acc"]:.2f}%',
                         f'{r["est_peak_slip_clean_mm"]:.4f}', f'{r["est_peak_slip_occluded_mm"]:.4f}'])
        offline_tables.append("### " + label + "\n\n" + table(["策略", "Clean MSE ↓", "Occ MSE ↓", "Wrench R² ↑", "Frozen probe ↑", "Clean 代理滑移 ↓", "Occ 代理滑移 ↓"], rows))
    pa = table(["控制器", "四向平均峰值滑移 mm", "最差峰值 mm", "平均峰值倾角 °", "预紧命令抖动 mm", "零扰动旋转 °"],
               [[short[k], *[f'{r[m]:.4f}' for m in ["loaded_mean_inhand_slip_mm", "loaded_worst_inhand_slip_mm", "loaded_mean_tilt_deg", "preload_mean_jitter_mm", "zero_disturbance_jitter_deg"]]] for k, r in a.items()])
    pb = table(["控制器", "Stage 2 滑移 mm", "Stage 3 滑移 mm", "绝对径向误差 mm", "绝对轴夹角 °", "真实网格端点横向误差 mm", "几何准入"],
               [[short[k], *[f'{r[m]:.4f}' for m in ["stage2_mean_transport_slip_mm", "stage3_mean_align_slip_mm", "stage3_mean_socket_radial_error_mm", "stage3_mean_socket_axis_tilt_deg", "stage3_mean_socket_tip_lateral_error_mm"]], f'{100*r["long_horizon_success_rate"]:.0f}%'] for k, r in b.items()])
    stages = off["innovation_1_10stage_operator_ablation"]
    ablation = table(["算子", "Dev", "Fresh12 起始标签", "Fresh12 即时方向", "Fresh b0+b1", "PT02 36 OOD"],
                     [[k, *[f'{100*r[m]:.2f}%' for m in ["dev_pure_c005_c008_centroid_acc", "blind_fresh12_trial_label_centroid_acc", "blind_fresh12_instantaneous_centroid_acc", "blind_fresh_b0_b1_step_ramp_centroid_acc", "pt02_36ood_centroid_acc"]]] for k, r in stages.items()])
    rev = off["innovation_1a_fresh12_reversal_breakdown"]
    reversal = table(["算子", "b2 即时方向", "反转前 t24–42", "反转后 t43–59", "起始标签保持率"],
                     [[k, *[f'{100*rev[k][m]:.2f}%' for m in ["b2_4_reversal_instantaneous_acc", "b2_4_pre_reversal_t24_42_acc", "b2_4_post_reversal_t43_59_acc", "b2_4_onset_trial_acc"]]] for k in list(rev)[-3:]])
    cats = off["innovation_1b_pt02_7category_ood_breakdown"]["9_CMQ_BPI_MCF_48D"]
    ood = table(["OOD 子集", "试次", "帧", "Centroid", "Ridge"], [[k, r["n_trials"], r["n_frames"], f'{100*r["centroid_acc"]:.2f}%', f'{100*r["ridge_acc"]:.2f}%'] for k, r in cats.items()])
    onset = table(["试次", "通过 / 7", "未达标帧及输出"],
                  [[k, f'{sum(r["passed"] for r in rows)}/7', "; ".join(f't{r["t"]}: ux={r["u_xy"][0]:.4f}, uy={r["u_xy"][1]:.4f}' for r in rows if not r["passed"]) or "无"] for k, rows in audit["onset_direction_checks"].items()])
    all_offline = all(v for split in audit["offline_metric_comparisons"].values() for row in split.values() for v in row.values())
    acc = audit["physical_acceptance"]
    decision = table(["验收项", "实际结论"], [
        ["5 项 verify_unit_audits", str(audit["unit_audits_five_passed"])],
        ["With-Aux 的离线四类指标同时优于 BC / No-Aux", str(all_offline) + "（单种子点估计，无统计显著性检验）"],
        ["起始 t30–36 四方向幅值/门控判据", f'{audit["onset_passed"]}/28，未全部通过'],
        ["所有 3DoF 的外层增益、门限与 DLS 参数一致", str(acc["matched_external_controls"])],
        ["Part A With-Aux 同时优于 V2.1、No-Aux、BC", str(all(acc["part_a_proposed_beats_baselines"].values()))],
        ["零扰动旋转 ≤ 0.16°", f'{a[prop]["zero_disturbance_jitter_deg"]:.4f}°，{acc["zero_jitter_le_016deg"]}'],
        ["Part B With-Aux 三项绝对几何误差均最优", str(all(acc["part_b_proposed_best_by_metric"].values()))],
        ["当前场景已证明插孔准入/成功", "否；所有控制器几何准入均为 0%，未验证插入深度或接触成功"],
    ])
    summary = (f'本轮收尾执行完成，但验收**未全部通过**。5 项单元审计为 True，离线指定指标比较通过；'
               f'起始帧幅值判据仅 {audit["onset_passed"]}/28。With-Aux 的 Part A 四向平均峰值滑移为 '
               f'**{a[prop]["loaded_mean_inhand_slip_mm"]:.4f} mm**，零扰动旋转为 **{a[prop]["zero_disturbance_jitter_deg"]:.4f}°**。'
               f'Part B 的绝对网格端点横向误差为 **{b[prop]["stage3_mean_socket_tip_lateral_error_mm"]:.4f} mm**，'
               '场景尚未预对准插孔，不能把初始位姿漂移当作插孔误差，也不能保留旧版“100% 插孔准入”的结论。')
    methodology = '''## 方法与本轮修订（Option A）

当前可支持的表述是：**受控接触代理目标下、带特权辅助锚定的有界残差策略精调，随后在 Isaac Lab / PhysX 中冻结评估，并报告真实物体位姿的插孔相对几何与反转方向分段结果。**

- `tac_encoder` 输入恢复为 `tac_dim`；活动能量只送门控。对每条训练轨迹的前 8 个激活帧赋予 3.0 倍权重，应用于 BC 动作/辅助损失及 Stage B 的轨迹奖励与辅助损失。训练中没有按某个方向手工翻转输出。
- `vis_avail = (norm(x_vis) > 1e-5)` 同时屏蔽 `g_vis` 和双头融合；无相机时基础双头严格退化到 `u_damp`。残差策略仍有冻结 BC 与有界残差分支，不应把完整残差输出等同于单个 `u_damp`。
- Stage B 两分支均为 65 次更新、AdamW 学习率 0.0016、weight decay 0.0001、触觉噪声 0.11、本体噪声 0.22、PG 权重 1.65、残差正则 0.01、quiet 权重 1.0、4 对对偶样本、alpha=0.35、梯度裁剪 1.5。共同的随机种子与样本顺序保持一致。开关差异是方向/4D wrench/合成伴随拷贝/视觉信任门四项辅助损失的整体，不是只隔离 wrench loss。
- Gaussian 样本先 `.detach()`，以截断前 latent 计算 log-prob，以截断后的动作计算奖励，避免把边界截断动作误当作连续 Gaussian 样本。BC 基座参数与保存的 BC 权重逐项相等且没有梯度。`alpha*delta_u` 各分量上限 0.35；最终门控为 `0.85*g_BC+0.15*g_train`，故不能把这一上限直接套到与已门控 BC 的最终输出差上。
- 奖励使用 `[Fx,Fy,Mx,My]` 载荷强度和对教师动作的误差。状态来自固定轨迹回放，没有动作驱动的环境状态转移；这是受控代理上的 score-function 策略精调，**不是已完成 Isaac Lab 在线 PPO/SAC**。`Surrogate_Slip` 是公式构造的代理指数，JSON 的 `_mm` 是兼容字段，尚无物理毫米标定，且与动作 MSE 并非独立证据。辅助标签来自仿真特权量，不宣称无需真值的自监督。
- 本体 12D 输入真实读取 `joint_pos/joint_vel/action` 历史，但关节分组投影为启发式，82% 接触状态混合与随机矩阵 `E_reaff` 注入都是受控合成设计，尚未证明真实运动学外源/自生触觉解耦。
- V2.1 和 V2.2 共用外层增益 `6.2/6.8`、门限 `0.45`、偶极下限 `0.26`、平滑 `0.40`、quiet 噪声 `0.08` 和 DLS 参数。修复了 V2.2 lag 索引的一帧差异和 V2.1 截窗后参考点漂移。各模型内部学习门控及 V2.2 的触觉锚定预处理仍不同；本项只证明外层控制参数公平。零扰动旋转改为 float64 归一化四元数距离，并通过 0.01° 非单位四元数测试；旧字段 max_ang_deg 为兼容保留，微角度应使用新增 max_normalized_rotation_deg。
- 额外试验曾加入整段无相机训练丢弃，但没有提高 20/28 的幅值通过率，且使 `-Fy,t34` 反向，故撤回。其代码、权重、日志与完整物理回放保存在 `outputs/camera_dropout_trial_20261104/`。最终训练保留原有 clean/冻结延迟视觉配对；不声称完整无相机模态训练问题已解决。

## 数据与统计口径

归一化和策略训练使用 pure_loading_01 的 c001–c004；Dev 为 c005–c008。Fresh12 为 12 条非零试次，另有 3 条 zero 对照；b0+b1 为 8 条，b2 为 4 条。`data/fresh_trials/` 包含全部 15 条试次及元数据、标定文件。pure_loading_01 和 pilot_tangent_02 仍依赖既有 E1S 数据归档；Isaac 依赖本机环境、官方资产和原始 V2.1/Stage 3S 权重。

Fresh12 已多次被评估，称历史留出集；没有进入本轮训练/归一化，但不再宣称这是一次从未查看的新盲测。PT02 共加载 44 条非零试次，其中 8 条用于本域训练、36 条用于 OOD 评估；实际为 6 类测试子集，历史脚本/字段名中的 `7category` 保留兼容，不能写成纯从 pure_loading_01 零样本迁移。

离线表为 trial macro 均值，PT02 分组为帧加权准确率；物理表是四个方向的逐试次峰值再平均。只有一个训练种子、一个固定物理场景、每方向一次回放，无置信区间或显著性检验，不将帧数当成独立重复次数。
'''
    geometry = '''## 插孔几何的定义与边界

设 `R_s,p_s` 为 `_socket_obj` 的真实位姿，`R_p,p_p` 为插头位姿。新字段使用 `p_rel=R_s.T@(p_p-p_s)`；径向误差为 `1000*norm(p_rel[:2])`；轴角为两物体局部 Z 轴的无向夹角；端点使用实际 00186 URDF 网格的 +Z 远端面中心（含 URDF 和对象缩放），在插孔坐标系计算横向距离。**不减初始误差**。每试次汇总最后 10 帧，再平均四方向。

旧版初始姿态漂移保留在 `stage3_socket_radial_drift_mm`、`stage3_socket_axis_drift_deg`、`stage3_legacy_30mm_point_lateral_drift_mm`；`stage3_final_tip_error_mm` 仍是握持滑移加倾角的旧代理量，不能引用为真实端点误差。新几何计算通过已知 3–4–5 mm 偏置检查；网格端点定义、资产哈希与准入阈值在 JSON `socket_geometry_protocol` 中记录。

当前名义动作是保持夹持姿态，阶段改变外力，未执行到插孔的搬运/接近轨迹。因而本次 Part B 是三阶段抗扰回放加绝对几何诊断，尚不是已实现的插孔任务。几何阈值 2.20 mm/1.50°/握持滑移 1.90 mm 为本协议判据，不冒充资产认证公差。控制器键保留 `6_v2_2_online_residual_rl_proposed`，不会为对应计划中的编号 3 而重命名原始结果。
'''
    evidence = "\n\n".join(["## 验收汇总\n\n" + decision, *offline_tables, "## Part A：物理闭环\n\n" + pa,
                            "## Part B：绝对插孔几何\n\n" + pb, "## 起始帧逐项判据\n\n" + f'方向符号/抑制本身通过 {audit["onset_sign_only_passed"]}/28；包含 ±0.7 幅值门槛的完整判据通过 {audit["onset_passed"]}/28。\n\n' + onset,
                            "## 十阶段算子消融\n\n" + ablation, "## b2 反转前后透明分析\n\n" + reversal,
                            "CMQ 的反转后即时方向准确率为 29.41%，低于 BPI 的 54.41%；81.65% 是起始标签准确率，不能解释为反转跟踪成功。",
                            "## PT02 的六类 OOD\n\n" + ood])
    audit_rows = [
        ["P1-1", "策略梯度断图与截断密度", "代码检查通过", "detach latent + preclip Gaussian log-prob；mu 单元梯度非零。"],
        ["P1-2", "Stage B 消融公平性", "代码检查通过", "相同训练循环、种子、更新次数、噪声、PG/正则；辅助损失束为唯一开关。"],
        ["P1-3", "冻结 BC / 有界残差 / 4D 代理", "实现通过，原在线物理 RL 主张不成立", "冻结权重一致、无梯度、残差有界；4D wrench 改变奖励。无动作驱动状态转移，采用 Option A 限定表述。"],
        ["P1-4", "闭环外层增益与门控一致", "修复并通过", "统一偶极下限及 lag；JSON fairness=True。模型内部结构不同，不能声称内部门完全相同。"],
        ["P1-5", "真实 raw tactile / 本体历史传入", "实现检查通过，物理解释受限", "data_win 六项齐全；缺 tactile 抛错；关节/速度/动作扰动敏感。启发式投影及 E_reaff 不等于物理解耦证明。"],
        ["P1-6", "真实插头—插孔几何", "公式已修复，任务验收未通过", "发现并去除初始相对位姿减法；从资产读取端点。约 100 mm 绝对错位，全部准入为 0%。"],
        ["P1-7", "因果性与离线/无视觉部署对齐", "部分通过", f'所审单条轨迹的全部前缀差异为 0；无视觉偏置为 0；起始幅值仍仅 {audit["onset_passed"]}/28。'],
        ["P2-1", "b2 分段和负结果透明", "通过", "保留即时方向、起始标签、反转前后四种口径；明确 29.41% 的反转后负结果。"],
        ["P2-2", "fresh 数据自包含与归档", "通过（保留外部依赖说明）", "15 条试次在目录内，完整 SHA256 清单覆盖代码、数据、结果、权重、图表、文档和快照。"],
    ]
    review = ("# Stage 4 V2.2 二次复核报告（2026-11-04）\n\n" + summary + "\n\n"
              + "下表按本次请求中的问题类别重建 7 项 P1 / 2 项 P2 核验映射；仓库未提供原始逐字审计清单，不冒充原审计编号。代码修复通过与科学/性能验收通过分别判断。\n\n"
              + table(["复核编号", "问题", "状态", "证据与边界"], audit_rows) + "\n\n" + evidence + "\n\n" + methodology + "\n\n" + geometry)
    checks = ["verify_three_innovations_v2_2_cmq.py", "verify_isaac_closedloop_v2_2_residual_rl.py", "socket_geometry.py",
              "outputs/v2_2_verification_results.json", "outputs/isaac_closedloop_v2_2_residual_rl_report.json",
              "outputs/checkpoints/v22_5_online_rl_with_aux_proposed.pt", "outputs/secondary_audit_20261104.json"]
    hash_table = table(["文件（相对 V2.2 目录）", "SHA256"], [[name, hashlib.sha256((ROOT / name).read_bytes()).hexdigest()] for name in checks])
    review += "\n\n## 证据入口与 SHA256\n\n" + hash_table + "\n\n完整清单见 `archive_manifest.json`，不包含清单自身以避免自引用。最终运行日志为 `outputs/offline_run_20261104.log` 和 `outputs/isaac_run_20261104.log`。Isaac 正常返回 0 并输出 30+24 条结果；日志中的 USD visual 引用警告保留，未据此声称资产渲染完美。\n"
    anchors = []
    for filename, symbols in {
        "verify_three_innovations_v2_2_cmq.py": ["extract_proprioceptive_history", "forward", "compute_episode_closedloop_surrogate", "train_warmstart_policy", "run_surrogate_residual_rl", "evaluate_fresh12_reversal_breakdown"],
        "verify_isaac_closedloop_v2_2_residual_rl.py": ["step_v21_policy", "step_v22_policy"],
        "socket_geometry.py": ["socket_alignment", "quaternion_angle_deg"],
        "audit_v2_2_closeout.py": ["load_runtime_step", "main"],
    }.items():
        lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
        for symbol in symbols:
            number = next(i for i, line in enumerate(lines, 1) if line.lstrip().startswith("def " + symbol + "("))
            anchors.append([f"{filename}:{number}", symbol])
    review += "\n\n## 代码定位\n\n" + table(["文件与行号（对应上述文件哈希）", "函数"], anchors) + "\n"
    (ROOT / "AUDIT_RECHECK_20261104.md").write_text(review, encoding="utf-8")
    commands = '''## 复现与文件入口

从仓库根目录执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python experiments/v2_2_cmq_proprio_residual_rl/verify_three_innovations_v2_2_cmq.py
& "C:/isaacsim_env/Scripts/python.exe" experiments/v2_2_cmq_proprio_residual_rl/verify_isaac_closedloop_v2_2_residual_rl.py --headless --device cpu --official-repo C:/isaac_ws/FlexiTac-IsaacSim-Simulation
python experiments/v2_2_cmq_proprio_residual_rl/audit_v2_2_closeout.py
python experiments/v2_2_cmq_proprio_residual_rl/plot_v2_2_figures.py
python experiments/v2_2_cmq_proprio_residual_rl/sync_v2_2_closeout_docs.py
python experiments/v2_2_cmq_proprio_residual_rl/refresh_archive_manifest.py
```

研究说明见 [Stage 4 记录](../../research/STAGE4_V2_2_CMQ_PROPRIO_RESIDUAL_RL_20261103.md)，逐条复核见 [二次审计报告](AUDIT_RECHECK_20261104.md)。脚本支持 `--pure-dir`、`--pt02-dir`、`--fresh-dir` 指定数据路径；本轮实际路径保存在 JSON provenance 中。

`outputs/pre_audit_v2_2_snapshot_20261103/` 保留此前原始结果；`outputs/pre_closeout_snapshot_20261104/` 保留本轮开始时的代码、结果与文档；`outputs/camera_dropout_trial_20261104/` 保留撤回的扩展试验。Stage 1、Stage 2P、Stage 3S 和 V2.1 历史文件没有改写。
'''
    readme = "# Stage 4 V2.2：受控代理残差精调与物理回放归档\n\n日期：2026-11-04（二次审计版）。\n\n" + summary + "\n\n" + evidence + "\n\n" + methodology + "\n\n" + geometry + "\n\n" + commands
    readme += "\n\n## 论文图表\n\n"
    for name, title in [("fig1_10stage_cmq_ablation", "算子消融：即时方向标签"), ("fig2_pt02_7category_ood", "36 OOD 的六类子集"), ("fig3_online_rl_anticollapse", "辅助锚定精调：未裁剪 R² 与动作/探针指标"), ("fig4_isaac_closedloop_and_longhorizon", "物理滑移、零扰动旋转与绝对插孔几何")]:
        readme += f"![{title}](outputs/figures/{name}.png)\n\n"
    readme += "关键证据哈希如下；所有文件的字节数及 SHA256 见 [归档清单](archive_manifest.json)。\n\n" + hash_table + "\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    research = "# Stage 4 V2.2 研究记录（2026-11-03 建立，2026-11-04 二次审计修订）\n\n" + summary + "\n\n" + methodology + "\n\n" + evidence + "\n\n" + geometry
    research += "\n\n## 研究判断与下一步\n\n本轮支持辅助锚定改善指定代理任务的动作误差和 wrench 解码，不能据此断言无辅助 RL 必然表征坍缩：With-Aux 的 Fresh 冻结探针仍低于 BC。Part A 的比较以本轮统一参数重跑为准，不与 V2.1 历史不同增益结果直接计算增益。插孔对齐和完整无相机起始段尚未通过；正式在线 Isaac Lab PPO/SAC、名义插孔轨迹、接触/插入深度、独立多种子与真机闭环仍未完成。这些是证据边界与后续建议，不另行修改用户的研究范围或推进授权。\n\n"
    research += "完整归档见 [V2.2 README](../experiments/v2_2_cmq_proprio_residual_rl/README.md)，逐项证据见 [二次审计](../experiments/v2_2_cmq_proprio_residual_rl/AUDIT_RECHECK_20261104.md)。此前版本保存于 V2.2 快照，不再沿用“必然坍缩”“完整证据链”“零样本真机迁移”“100% 插孔成功”等未获支持表述。\n\n## 关键证据 SHA256\n\n" + hash_table + "\n"
    (REPO / "research/STAGE4_V2_2_CMQ_PROPRIO_RESIDUAL_RL_20261103.md").write_text(research, encoding="utf-8")
    freeze = REPO / "research/ICRA2028_MINIMAL_PAPER_FREEZE.md"
    old = freeze.read_text(encoding="utf-8")
    suffix = old[old.index("> **前序状态（2026-11-03 V2.1"):]
    new = "# ICRA2028：研究方向与范围记录（2026-11-04 Stage 4 / V2.2 二次审计版）\n\n"
    new += "> **最新状态：2026-11-04 审计修订。** " + summary + "\n>\n"
    new += "> **研究范围保留，证据表述修订为 Option A：** CMQ-BPI-MCF 48D + 本体/触觉双头 + 受控代理目标上的辅助锚定有界残差精调，以及冻结策略的 PhysX 闭环和真实插孔坐标几何诊断。该精调没有动作驱动的仿真状态转移，不能等同于已完成 Isaac Lab PPO/SAC。辅助监督使用特权 4D `[Fx,Fy,Mx,My]` 标签；不预设模型优越性或真机迁移成立。\n>\n"
    new += "> CMQ 的 Fresh12 即时方向准确率为 77.02%，起始标签准确率为 81.65%；b0+b1 为 86.36%，b2 反转后仅 29.41%（BPI 54.41%）。PT02 36 OOD 为 60.97%，质心/归一化在 PT02 的 8 条训练试次拟合，不能写成跨域零样本迁移。旧 Stage 4 数值和文档已保存在快照，Stage 1/2P/3S/V2.1 历史不改写。\n>\n"
    new += "> 最新结果见 [Stage 4 研究记录](STAGE4_V2_2_CMQ_PROPRIO_RESIDUAL_RL_20261103.md)、[V2.2 归档](../experiments/v2_2_cmq_proprio_residual_rl/README.md) 和 [二次审计报告](../experiments/v2_2_cmq_proprio_residual_rl/AUDIT_RECHECK_20261104.md)。下方 V2.1 与 2026-10-10 文本为历史记录，其当时结果不作为本轮统一控制参数下的比较值。\n\n"
    freeze.write_text(new + suffix, encoding="utf-8")
    print("Synchronized README, research record, freeze entry and audit report from current JSON.")


if __name__ == "__main__":
    main()
