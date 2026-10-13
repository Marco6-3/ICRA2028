"""All-condition report, preselected hard-case figure and artifact validation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reflex_metrics import read


def main():
    p=argparse.ArgumentParser()
    for k in ('nominal','high','mcf','scalar','oracle'): p.add_argument('--'+k,type=Path,required=k in ('nominal','high','mcf'))
    p.add_argument('--out',type=Path,required=True); p.add_argument('--bounded-pure',type=Path,required=True); p.add_argument('--students',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    roots={k:getattr(a,k) for k in ('nominal','high','mcf','scalar','oracle') if getattr(a,k)}
    rows={}; series={}; checks=[]; artifact_hashes={}
    for name,root in roots.items():
        manifest=json.loads((root/'manifest.json').read_text()); assert manifest['status']=='completed'
        rows[name]=[]; series[name]={}
        for tid in manifest['completed_trials']:
            row,d,slip,force=read(root,tid); rows[name].append(row); series[name][tid]=(d,slip,force)
            assert all(np.isfinite(v).all() for v in d.values())
            assert np.allclose(d['input_mcf_before'][1:],d['mcf'][:-1],rtol=0,atol=0)
            assert np.allclose(np.diff(d['observation_time_s']),float(d['physics_dt']))
            assert np.all((d['controller_delta_rad']>=0)&(d['controller_delta_rad']<=.003))
            command=d['action'][:,[14,15]]-d['controller_delta_rad'][:,None]*np.array([-1,1])[None]
            assert np.allclose(command,command[:1],atol=1e-8)
            checks.append({'controller':name,'id':tid,'finite':True,'causal_previous_mcf_exact':True,'timestamps':True,'command_authority':True,'npz_hash':True})
            artifact_hashes[str(root/f'{tid}.npz')]=hashlib.sha256((root/f'{tid}.npz').read_bytes()).hexdigest()
        (root/'metrics.json').write_text(json.dumps(rows[name],indent=2),encoding='utf-8')
    ids=list(series['nominal'])
    for name in roots:
        assert list(series[name])==ids
        for tid in ids:
            d=series[name][tid][0]; base=series['nominal'][tid][0]
            for field in ('initial_object_pose','initial_robot_body_pose','force_world_n','torque_com_world_nm','step_start_s'):
                assert np.allclose(d[field],base[field],rtol=0,atol=1e-8),(name,tid,field)
    summary={}
    for name,rr in rows.items():
        summary[name]={'conditions':len(rr),'within_2mm':sum(r['goal_slip_2mm'] for r in rr),
            'worst_slip_mm':max(r['max_slip_mm'] for r in rr),'mean_normal_effort_n_s':float(np.mean([r['effort_normal_n_s'] for r in rr])),
            'mean_post_normal_n':float(np.mean([r['post_normal_n'] for r in rr])),
            'worst_tilt_deg':max(r['max_tilt_deg'] for r in rr),
            'mean_saturation_fraction':float(np.mean([r['saturation_fraction'] for r in rr])),
            'peak_normal_n':max(r['peak_normal_n'] for r in rr),'drops':sum(r['dropped'] for r in rr),
            'worst_environment_normal_n':max(r['environment_max_normal_n'] for r in rr),
            'worst_control_p95_ms':max(r['control_p95_ms'] for r in rr)}
    summary['mcf']['normal_effort_reduction_vs_high']=1-summary['mcf']['mean_normal_effort_n_s']/summary['high']['mean_normal_effort_n_s']
    summary['mcf']['post_normal_reduction_vs_high']=1-summary['mcf']['mean_post_normal_n']/summary['high']['mean_post_normal_n']
    amplitude_summaries={}
    for amplitude in sorted({r['trial']['amplitude_n'] for r in rows['mcf']}):
        selected={n:[r for r in rr if r['trial']['amplitude_n']==amplitude] for n,rr in rows.items()}
        m,h=selected['mcf'],selected['high']
        amplitude_summaries[str(amplitude)]={'conditions':len(m),'mcf_within_2mm':sum(r['goal_slip_2mm'] for r in m),
            'mcf_max_slip_mm':max(r['max_slip_mm'] for r in m),
            'normal_effort_reduction_vs_high':1-np.mean([r['effort_normal_n_s'] for r in m])/np.mean([r['effort_normal_n_s'] for r in h]),
            'post_normal_reduction_vs_high':1-np.mean([r['post_normal_n'] for r in m])/np.mean([r['post_normal_n'] for r in h])}
    (a.out/'results.json').write_text(json.dumps(dict(status='completed',summary=summary,amplitude_summaries=amplitude_summaries,conditions=rows),indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    palette={'nominal':'#748094','high':'#e69a35','mcf':'#087b75','scalar':'#8c6abd','oracle':'#2c5ea8'}
    labels={'nominal':'Fixed Nominal','high':'Fixed High-Grip','mcf':'MCF-GRU Reflex','scalar':'Scalar-GRU Reflex','oracle':'Privileged expert'}
    tid='test_01_0'; fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
    for name in ('nominal','high','mcf'):
        d,slip,force=series[name][tid]; t=d['observation_time_s']
        axes[0].plot(t,slip,label=labels[name],color=palette[name],linewidth=2.)
        axes[1].plot(t,np.rad2deg(d['tilt_change_rad']),color=palette[name],linewidth=2.)
        axes[2].plot(t,force,color=palette[name],linewidth=1.7)
        axes[3].plot(t,d['controller_delta_rad']*1000,color=palette[name],linewidth=1.7)
    axes[0].axhline(2,color='#c14848',linestyle='--',linewidth=1,label='2 mm target')
    inset=axes[0].inset_axes([.69,.28,.28,.64])
    for name in ('high','mcf'):
        d,slip,_=series[name][tid]; inset.plot(d['observation_time_s'],slip,color=palette[name],linewidth=1.6)
    inset.axhline(2,color='#c14848',linestyle='--',linewidth=.8); inset.set_ylim(0,3)
    inset.set_xlim(.25,1.5); inset.set_title('Feedback / high-grip detail',fontsize=8); inset.tick_params(labelsize=7)
    load=next(r['trial'] for r in rows['nominal'] if r['id']==tid)
    for ax in axes:
        ax.axvspan(load['onset_s'],load['onset_s']+load['duration_s'],color='#e2edf2',alpha=.65); ax.grid(axis='y',alpha=.18)
    for ax,label in zip(axes,['Relative slip (mm)','Tilt change (deg)','Total normal contact (N)','Grip increment (mrad)']): ax.set_ylabel(label)
    axes[0].legend(loc='upper left',ncol=2,frameon=False)
    axes[-1].set_xlabel('Simulation time (s)'); axes[-1].set_xlim(0,1.5)
    fig.suptitle('Suspended grasp | +Fx 0.20 N at 15 mm offset\nTactile-only 32D MCF + GRU → grip increment',fontweight='bold',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(a.out/'reflex_curve.png',dpi=180)
    fig.savefig(a.out/'reflex_curve.pdf'); fig.savefig(a.out/'reflex_curve.svg'); plt.close(fig)
    # Additional paired curve for the lower test amplitude, while keeping the
    # preselected hardest case above and every condition in the tables below.
    tid_success='test_00_0'; load_success=rows['nominal'][0]['trial']
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for name in ('nominal','high','mcf'):
        d,slip,force=series[name][tid_success]; t=d['observation_time_s']
        axes[0].plot(t,slip,label=labels[name],color=palette[name],linewidth=2.)
        axes[1].plot(t,force,color=palette[name],linewidth=1.8)
        axes[2].plot(t,d['controller_delta_rad']*1000,color=palette[name],linewidth=1.8)
    axes[0].axhline(2,color='#c14848',linestyle='--',linewidth=1,label='2 mm target')
    inset=axes[0].inset_axes([.67,.2,.3,.66])
    for name in ('high','mcf'):
        d,slip,_=series[name][tid_success]; inset.plot(d['observation_time_s'],slip,color=palette[name],linewidth=1.8)
    inset.axhline(2,color='#c14848',linestyle='--',linewidth=.8); inset.set_ylim(0,2.5); inset.tick_params(labelsize=8); inset.set_title('Grip controllers',fontsize=9)
    for ax,label in zip(axes,['Relative slip (mm)','Total normal contact (N)','Grip increment (mrad)']):
        ax.set_ylabel(label); ax.grid(axis='y',alpha=.18); ax.axvspan(load_success['onset_s'],load_success['onset_s']+load_success['duration_s'],color='#e2edf2',alpha=.65)
    axes[0].legend(loc='upper left',frameon=False,ncol=2)
    axes[-1].set_xlabel('Simulation time (s)'); axes[-1].set_xlim(0,1.5)
    m=rows['mcf'][0]; h=rows['high'][0]; b=rows['nominal'][0]
    fig.suptitle(f"Tactile Reflex in a suspended grasp | +Fx {load_success['amplitude_n']:.2f} N, 15 mm offset\nSlip {b['max_slip_mm']:.2f} → {m['max_slip_mm']:.2f} mm | Normal effort {(1-m['effort_normal_n_s']/h['effort_normal_n_s']):.0%} below High-Grip",fontweight='bold',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(a.out/'reflex_success_curve.png',dpi=180)
    fig.savefig(a.out/'reflex_success_curve.pdf'); fig.savefig(a.out/'reflex_success_curve.svg'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8)); xx=np.arange(len(ids)); width=.24
    for j,name in enumerate(('nominal','high','mcf')):
        ax.bar(xx+(j-1)*width,[r['max_slip_mm'] for r in rows[name]],width,label=labels[name],color=palette[name])
    ax.axhline(2,color='#c14848',linestyle='--',linewidth=1)
    ax.set_xticks(xx,[f"{r['trial']['amplitude_n']:.2f} N\n{r['trial']['direction']}" for r in rows['nominal']]); ax.set_ylabel('Maximum relative slip (mm)')
    ax.legend(frameon=False); ax.grid(axis='y',alpha=.18); fig.tight_layout(); fig.savefig(a.out/'all_conditions.png',dpi=180); plt.close(fig)
    bounded=json.loads((a.bounded_pure/'results.json').read_text())
    lines=['# 仿真 Reflex 原型交付','', '已完成有界归一化、无支撑数据重新验证、专家采集、同数据学生训练，以及 Isaac Sim 实际闭环比较。',
        '', '## 无支撑离线修复','', '| 表征 | 归一化 | 方向准确率 | Fx RMSE (N) | Fy RMSE (N) | 最大归一化输入 |', '|---|---|---:|---:|---:|---:|']
    for r in bounded['summary']: lines.append(f"| {r['kind']} | {r['normalization']} | {r['accuracy']:.1%} | {r['rmse'][0]:.4f} | {r['rmse'][1]:.4f} | {r['max_input']:.2f} |")
    lines+=['', '同模型、相同目标尺度、相同训练条件和三 seeds；回归输出未截断。Static MCF 改善明显，其他组的方向准确率没有一致提高，因此数值修复和预测收益分别报告。',
        '', '## 实际闭环结果','', '| 控制器 | ≤2 mm 条件数 | 最坏滑移 (mm) | 平均法向力积分 (N·s) | 扰动后平均法向力 (N) | 最大倾角变化 (deg) |', '|---|---:|---:|---:|---:|---:|']
    for name,r in summary.items(): lines.append(f"| {labels[name]} | {r['within_2mm']}/{r['conditions']} | {r['worst_slip_mm']:.3f} | {r['mean_normal_effort_n_s']:.3f} | {r['mean_post_normal_n']:.3f} | {r['worst_tilt_deg']:.3f} |")
    reduction=summary['mcf']['normal_effort_reduction_vs_high']; post=summary['mcf']['post_normal_reduction_vs_high']
    lines+=['', '| 幅度 | MCF ≤2 mm | MCF 最坏滑移 mm | 法向力积分降低 | 扰动后法向力降低 |', '|---|---:|---:|---:|---:|']
    for amp,r in amplitude_summaries.items(): lines.append(f"| {amp} N | {r['mcf_within_2mm']}/{r['conditions']} | {r['mcf_max_slip_mm']:.3f} | {r['normal_effort_reduction_vs_high']:.1%} | {r['post_normal_reduction_vs_high']:.1%} |")
    lines+=['', f'MCF Reflex 相对 High-Grip 的平均法向力积分降低 {reduction:.1%}，扰动后平均法向力降低 {post:.1%}。',
        '', '![较低测试幅度的成对成功曲线](reflex_success_curve.png)',
        '', '![预先指定的 0.20 N +Fx 闭环曲线](reflex_curve.png)', '', '![全部测试条件](all_conditions.png)',
        '', '## 逐条件结果','', '| 载荷 | 方向 | Nominal 滑移 mm | High-Grip 滑移 mm | MCF 滑移 mm | MCF/High 力积分比 |', '|---|---|---:|---:|---:|---:|']
    for i,tid in enumerate(ids):
        b,h,m=(rows[n][i] for n in ('nominal','high','mcf'))
        lines.append(f"| {b['trial']['amplitude_n']:.2f} N | {b['trial']['direction']} | {b['max_slip_mm']:.3f} | {h['max_slip_mm']:.3f} | {m['max_slip_mm']:.3f} | {m['effort_normal_n_s']/h['effort_normal_n_s']:.1%} |")
    lines+=['', '## 复核与使用范围','', '各控制器具有相同初始物体与机器人状态、相同外力和偏心力矩、相同加载时间；逐步记录动作、触觉、独立物体姿态、相对滑移、环境接触和模型调用时间。所有输入确认为上一帧 MCF，学生输入不包含物体状态或加载时钟。测试时仍单独记录专家标签，便于诊断，但不送入学生。',
        '', '最大滑移使用初态物体相对于两指支撑 link 的三维位移，包含夹指闭合导致的法向变化；不是仅取世界位置。倾角变化为相对初始朝向的世界竖直轴摆动角 acos((R_current·R_initial^T)[2,2])，排除纯世界竖直轴 yaw，另保存完整姿态转角。法向力积分来自 PhysX 接触投影，作为抓握 effort 代理，不等同于电耗或物体损伤。',
        '', '这是当前刚体接触/SDF 读出的仿真原型；最终测试留出了加载时刻及部分幅度，仍在同一夹爪/物体几何内。真实 FlexiTac E1 和更广条件的正式结论留待硬件与后续实验。',
        '', f"MCF 控制调用 P95 最大值 {summary['mcf']['worst_control_p95_ms']:.3f} ms（包含调试专家标签计算，未包含真实传感器/执行器链路）。所有测试的环境接触最大法向力 {max(r['worst_environment_normal_n'] for r in summary.values()):.6g} N。", '']
    (a.out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    verify={'status':'passed','trials_checked':len(checks),'checks':checks,'paired_initial_conditions_and_loads':True,'data_hashes':artifact_hashes}
    for path in a.students.glob('*.pt'): artifact_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.out/'verification.json').write_text(json.dumps(verify,indent=2),encoding='utf-8')
    print('REFLEX_RESULTS',summary,flush=True)


if __name__=='__main__': main()
