"""Mechanism probes and measured safe-cell / cell-area operating envelopes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from reflex_metrics import read
from phase_diagnostics import diagnose,first_hit

FORCES=np.array([.10,.125,.15,.175,.20,.225,.25]); OFFSETS=np.array([0,5,10,15,20.])
DIRS=['+Fx','-Fx','+Fy','-Fy']; COLORS=ListedColormap(['#d95f55','#208678'])


def area_proxy(grid,minimum_e=0):
    area=0.
    for i in range(len(FORCES)-1):
        for j in range(len(OFFSETS)-1):
            if OFFSETS[j]>=minimum_e and grid[i:i+2,j:j+2].all(): area+=(FORCES[i+1]-FORCES[i])*(OFFSETS[j+1]-OFFSETS[j])
    return float(area)


def safe(row,meta,threshold=2):
    return bool(row['max_slip_mm']<=threshold and not row['dropped'] and row['environment_max_normal_n']<=1e-4 and meta['initial_gate_passed'])


def mechanism(root,out):
    manifest=json.loads((root/'mechanism/manifest.json').read_text()); assert manifest['status']=='completed'
    rows=[]
    for tid in manifest['completed_trials']:
        r,_,_=diagnose(root/'mechanism',tid)
        parts=tid.split('_'); r['mode']=parts[1]; r['signed_axis']=int(parts[2][1:]) if parts[1]!='zero' else None
        r['sensitivity']=[{'cop_floor':cf,'delays':diagnose(root/'mechanism',tid,cf)[0]['event_delay_ms']} for cf in (.001,.004)]
        rows.append(r)
    pulse=[]
    m=json.loads((root/'pulse/manifest.json').read_text()); assert m['status']=='completed'
    fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for tid in m['completed_trials']:
        r,d,z=diagnose(root/'pulse',tid); onset=r['trial']['onset_s']; t=d['observation_time_s']
        pre=(t>=.20)&(t<onset-.025); hold=(t>=onset+.10)&(t<onset+.25)
        start=float(np.median(z['normal'][pre])); plateau=float(np.median(z['normal'][hold])); rise=plateau-start
        ten=first_hit(z['normal']-start,t,onset,.1*rise) if rise>.01 else None
        ninety=first_hit(z['normal']-start,t,onset,.9*rise) if rise>.01 else None
        r.update(normal_step_n=rise,normal_10pct_s=ten,normal_90pct_s=ninety,
            normal_10_90_ms=(ninety-ten)*1000 if ten is not None and ninety is not None else None,
            plateau_closure_mrad=float(np.median(z['joint_delta'][hold])))
        pulse.append(r)
        if tid.endswith('r0'):
            ax[0].plot(d['step_start_s'],z['delta']*1000,label=tid)
            ax[1].plot(t,z['normal']-start,label=tid)
    for a in ax: a.axvspan(.45,.70,color='#e2edf2',alpha=.6); a.grid(axis='y',alpha=.2); a.legend(frameon=False)
    ax[0].set_ylabel('Command (mrad)'); ax[1].set_ylabel('Normal force increase (N)'); ax[1].set_xlabel('Simulation time (s)'); ax[1].set_xlim(.30,.9)
    fig.suptitle('No-load grip-step response | 0 / 1 / 3 mrad'); fig.tight_layout(); fig.savefig(out/'actuator_step.png',dpi=170); plt.close(fig)
    (out/'mechanism.json').write_text(json.dumps({'mechanism':rows,'actuator':pulse},indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(11,4.7))
    for k,key in enumerate(('cop','area')):
        for j,mode in enumerate(('force','torque','combined')):
            rr=[r for r in rows if r['mode']==mode and r['trial']['repeat']==0]
            values=[r['event_delay_ms'][key] for r in rr]
            xs=np.arange(4)+(j-1)*.22
            for x,v in zip(xs,values):
                if v is not None: ax[k].bar(x,v,width=.22,label=mode if x==xs[0] else None)
                else: ax[k].text(x,0,'ND',ha='center',rotation=90,fontsize=8)
        ax[k].set_xticks(np.arange(4),DIRS); ax[k].set_ylabel('Detection confirmation delay (ms)'); ax[k].set_title(key+' | 3-frame confirmation'); ax[k].grid(axis='y',alpha=.15)
    ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(out/'mechanism_delays.png',dpi=170); plt.close(fig)
    return rows,pulse


def phase(root,out):
    names=['nominal','high','scalar','mcf']; rows=[]; grid={}; summaries={}; trace_hashes={}
    initial={}
    for name in names+['oracle_fx']:
        directory=root/name; manifest=json.loads((directory/'manifest.json').read_text()); assert manifest['status']=='completed'
        for tid in manifest['completed_trials']:
            row,d,_,_=read(directory,tid)
            meta=json.loads((directory/f'{tid}.json').read_text()); row['initial_gate_passed']=meta['initial_gate_passed']; row['group']=name
            row['safe']=safe(row,meta); rows.append(row)
            assert np.array_equal(d['input_mcf_before'][1:],d['mcf'][:-1])
            assert np.all((d['controller_delta_rad']>=0)&(d['controller_delta_rad']<=.003))
            fields={k:d[k] for k in ('initial_object_pose','initial_robot_body_pose','force_world_n','torque_com_world_nm','step_start_s')}
            if name=='nominal': initial[tid]=fields
            else:
                for key,val in fields.items(): assert np.allclose(val,initial[tid][key],rtol=0,atol=1e-8),(name,tid,key)
            trace_hashes[f'{name}/{tid}.npz']=meta['npz_sha256']
            if tid=='phase_zero': continue
            parts=tid.split('_'); di,fi,ei=[int(x[1:]) for x in parts[1:4]]
            row.update(direction=DIRS[di],force_n=float(FORCES[fi]),offset_mm=float(OFFSETS[ei]))
            if name not in grid: grid[name]={di:np.zeros((7,5),bool) for di in range(4)}
            grid[name][di][fi,ei]=row['safe']
            if name=='mcf' and di==0:
                timing=diagnose(directory,tid)[0]; row['delays_ms']=timing['event_delay_ms']
    for di,direction in enumerate(DIRS):
        figure,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        summary={}
        for name,ax in zip(names,axes.flat):
            g=grid[name][di]
            ax.imshow(g.astype(int),origin='lower',cmap=COLORS,vmin=0,vmax=1,aspect='auto',extent=(-2.5,22.5,.0875,.2625))
            selected=[r for r in rows if r.get('group')==name and r.get('direction')==direction]
            for r in selected:
                ax.text(r['offset_mm'],r['force_n'],f"{r['max_slip_mm']:.1f}",ha='center',va='center',fontsize=8,color='white')
            summary[name]={'safe_cells':int(g.sum()),'total_cells':35,'cell_area_n_mm':area_proxy(g),
                'high_e_cell_area_n_mm':area_proxy(g,10),
                'safe_cells_1mm':sum(safe(r,dict(initial_gate_passed=r['initial_gate_passed']),1.) for r in selected),
                'safe_cells_3mm':sum(safe(r,dict(initial_gate_passed=r['initial_gate_passed']),3.) for r in selected),
                'mean_effort_n_s':float(np.mean([r['effort_normal_n_s'] for r in selected])),
                'drops':sum(r['dropped'] for r in selected)}
            ax.set_title(f"{name} | safe {g.sum()}/35"); ax.set_xticks(OFFSETS); ax.set_yticks(FORCES)
        for ax in axes[-1]: ax.set_xlabel('Offset e (mm)')
        for ax in axes[:,0]: ax.set_ylabel('External force F (N)')
        figure.suptitle(f'{direction} measured operating grid | green: slip ≤2 mm / no support / no drop\nNumbers: maximum relative slip (mm); no interpolated safe boundary',fontweight='bold',fontsize=12)
        figure.tight_layout(rect=(0,0,1,.94)); token=direction.replace('+','p').replace('-','n')
        figure.savefig(out/f'phase_{token}.png',dpi=170); figure.savefig(out/f'phase_{token}.pdf'); figure.savefig(out/f'phase_{token}.svg'); plt.close(figure)
        for key in ('cell_area_n_mm','high_e_cell_area_n_mm'):
            scalar=summary['scalar'][key]; mcf=summary['mcf'][key]
            summary['comparison_'+key]={'mcf_minus_scalar':mcf-scalar,'relative_change_percent':100*(mcf/scalar-1) if scalar>0 else None}
        summaries[direction]=summary
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    for key,ax in zip(('cop','command','normal_force'),axes):
        vals=np.full((7,5),np.nan)
        for r in rows:
            if r.get('group')=='mcf' and r.get('direction')=='+Fx':
                fi=np.flatnonzero(np.isclose(FORCES,r['force_n']))[0]; ei=np.flatnonzero(OFFSETS==r['offset_mm'])[0]
                v=r['delays_ms'][key]; vals[fi,ei]=v if v is not None else np.nan
        im=ax.imshow(vals,origin='lower',extent=(-2.5,22.5,.0875,.2625),aspect='auto',cmap='viridis',vmin=0,vmax=250)
        for i in range(7):
            for j in range(5): ax.text(OFFSETS[j],FORCES[i],f'{vals[i,j]:.0f}' if np.isfinite(vals[i,j]) else 'ND',ha='center',va='center',fontsize=8,color='white')
        ax.set_title(key+' event delay (ms)'); ax.set_xticks(OFFSETS); ax.set_yticks(FORCES); ax.set_xlabel('Offset e (mm)')
    axes[0].set_ylabel('External force F (N)'); fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Delay (ms)')
    fig.suptitle('+Fx MCF trajectory: detectable feature / command / contact response'); fig.savefig(out/'phase_delays_fx.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    (out/'phase_results.json').write_text(json.dumps({'summaries':summaries,'rows':rows,'trace_sha256':trace_hashes,'safe_area_type':'All-four-corners safe-cell area proxy; not a guaranteed continuous envelope'},indent=2),encoding='utf-8')
    with (out/'phase_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['group','id','direction','force_n','offset_mm','safe','max_slip_mm','max_tilt_deg','effort_normal_n_s','peak_normal_n','saturation_fraction','dropped']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(r for r in rows if r['id']!='phase_zero')
    return summaries,rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('source',type=Path); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(exist_ok=False,parents=True)
    status=json.loads((a.source/'pipeline.json').read_text()); assert status['status']=='completed'
    mechanics,pulses=mechanism(a.source,a.out); summaries,rows=phase(a.source,a.out)
    lines=['# 机制猜想与偏心抗扰相图（2026-10-14）','', '本轮完整检验纯力/纯力矩响应时序，以及四方向(F,e)控制安全域。模型和增益在扫描前冻结。','', '## 机制分离对照','', '| 类型 | 方向 | CoP 检出 ms | 面积检出 ms | Raw 检出 ms | 0.10 mm 位移 ms |','|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'未检出' if x is None else f'{x:.1f}'
    for r in mechanics:
        if r['mode']=='zero' or r['trial']['repeat']!=0: continue
        ev=r['event_delay_ms']; lines.append(f"| {r['mode']} | {DIRS[r['signed_axis']]} | {fmt(ev['cop'])} | {fmt(ev['area'])} | {fmt(ev['raw'])} | {fmt(ev['slip'])} |")
    lines+=['', '触觉事件使用三帧确认、6×MAD及预定量纲门限；确认时间不等于最早非零响应。纯力与纯力矩都有相同初态和时序，对照才是机制证据。两个重复用于恢复一致性，不当独立统计样本。',
        '', '| 类型 | 方向 | 噪声尺度 CoP 检出 ms | 噪声尺度响应和检出 ms | 主门限首次 Raw 越界 ms |', '|---|---|---:|---:|---:|']
    for r in mechanics:
        if r['mode']=='zero' or r['trial']['repeat']!=0: continue
        lines.append(f"| {r['mode']} | {DIRS[r['signed_axis']]} | {fmt(r['noise_only_delay_ms']['cop'])} | {fmt(r['noise_only_delay_ms']['proxy'])} | {fmt(r['first_cross_delay_ms']['raw'])} |")
    lines+=['', '噪声尺度检出只使用仿真基线与1e-6数值下限；弱的亚像素CoP响应不按大幅变化门限判成不可观测。主門限、敏感性和噪声检出结果同时保存。',
        '', '![机制时序](mechanism_delays.png)', '', '## 无外载执行器响应','', '| 阶跃 | 法向力增量 N | 法向力10–90% ms | 关节闭合平台 mrad |','|---|---:|---:|---:|']
    for r in pulses:
        if r['trial']['repeat']==0: lines.append(f"| {r['id']} | {r['normal_step_n']:.3f} | {fmt(r['normal_10_90_ms'])} | {r['plateau_closure_mrad']:.4f} |")
    lines+=['', '![执行器阶跃](actuator_step.png)', '', '## 安全域与实际扩大量','', '| 方向 | Nominal安全格 | High安全格 | Scalar安全格 | MCF安全格 | 高偏心分片面积 Scalar→MCF (N·mm) | 相对变化 |','|---|---:|---:|---:|---:|---|---:|']
    for direction,s in summaries.items():
        c=s['comparison_high_e_cell_area_n_mm']; change='未定义（Scalar面积为0）' if c['relative_change_percent'] is None else f"{c['relative_change_percent']:+.1f}%"
        lines.append(f"| {direction} | {s['nominal']['safe_cells']} | {s['high']['safe_cells']} | {s['scalar']['safe_cells']} | {s['mcf']['safe_cells']} | {s['scalar']['high_e_cell_area_n_mm']:.3f}→{s['mcf']['high_e_cell_area_n_mm']:.3f} | {change} |")
    for direction in DIRS: lines+=['',f"![{direction} 安全图](phase_{direction.replace('+','p').replace('-','n')}.png)"]
    lines+=['', '![+Fx 分阶段延迟](phase_delays_fx.png)', '', '## 判断范围','', '安全为最大三维相对位移≤2 mm、有效初态、无掉落且无环境支撑。相邻网格单元四角全部安全的面积仅是网格分片代理，未采内点没有连续安全保证。高偏心区域为e≥10 mm，另外报告1/3 mm敏感性；安全格与面积变化分别计算。',
        '', '同一刺激下的轨迹时序不能单独证明感知或执行器的因果瓶颈；纯载荷分离和无外载闭合阶跃用于区分机制。当前GRU只输出Δg，没有F̂x曲线。法向触觉是SDF穿透代理，物理法向力为独立PhysX投影；未用μfN>Fx忽略重力、力矩与接触几何。',
        '', 'High-Grip和特权参考仅是已测控制权限下的参照，不能据其失败宣布理论不可控。结构可观测性和真实传感器迟滞还需真实E1标定。所有结果、未检出事件、低表现控制器和失败试次保留。','']
    (a.out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    for d,s in summaries.items(): print('PHASE_RESULT',d,{n:s[n]['safe_cells'] for n in ('nominal','high','scalar','mcf')},s['comparison_high_e_cell_area_n_mm'],flush=True)

if __name__=='__main__': main()
