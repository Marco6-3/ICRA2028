"""Verify paired interventions and report measured safety/cost, including failures."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reflex_metrics import read
from analyze_phase import safe, area_proxy, FORCES, OFFSETS, DIRS

ROOT=Path(__file__).resolve().parent
COLORS={'fixed':'#8a9199','scalar':'#d18d2e','mcf':'#157f88','nominal':'#8a9199','high':'#534687',
        'none':'#157f88','pulse':'#d18d2e','latch':'#534687'}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load_group(directory, group, verify=True):
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['status']=='completed' and not manifest['errors'] and not manifest['rejected_initializations']
    provenance=json.loads((directory/'provenance.json').read_text())
    assert json.loads((directory/'force_calibration.json').read_text())['passed']
    rows=[]; states={}; hashes={}; zero_trace=None
    for tid in manifest['completed_trials']:
        row,d,slip,normal=read(directory,tid)
        meta=json.loads((directory/f'{tid}.json').read_text())
        t=d['step_start_s']; pre=(t>=.20)&(t<.425)
        near_load=(d['observation_time_s']>=row['trial']['onset_s']-.075)&(d['observation_time_s']<row['trial']['onset_s']-1e-9)
        row.update(group=group,safe=safe(row,meta),safe_1mm=safe(row,meta,1),safe_3mm=safe(row,meta,3),
            pre_normal_n=float(np.median(normal[pre])),pre_normal_min_n=float(normal[pre].min()),
            pre_normal_max_n=float(normal[pre].max()),repeat=meta['trial']['repeat'])
        row['normal_immediately_before_load_n']=float(np.median(normal[near_load]))
        if 'absolute_extra_close_rad' in d:
            absolute=d['absolute_extra_close_rad']
            row.update(preload_target_n=meta['target_total_preload_n'],trigger=meta['trigger'],
                saturation_fraction=float(np.mean(absolute>=.00297)),
                saturation_seconds=float(np.sum(absolute>=.00297)*d['physics_dt']),
                command_initial_extra_rad=float(absolute[0]),
                command_final_extra_rad=float(absolute[-1]))
            hits=np.flatnonzero(d['trigger_fired'])
            row['trigger_time_s']=float(t[hits[0]]) if hits.size else None
            row['trigger_delay_ms']=(row['trigger_time_s']-row['trial']['onset_s'])*1000 if hits.size else None
            row['trigger_before_load']=bool(hits.size and t[hits[0]]<row['trial']['onset_s']-1e-9)
            row['raw_policy_pre_rad']=float(d['pre_policy_delta_rad'][-1])
            row['centered_policy_reachable_max_extra_rad']=float(min(.003,absolute[0]+.003-d['pre_policy_delta_rad'][-1])) if meta['controller']!='fixed' else float(absolute[0])
            row['centered_policy_reachable_min_extra_rad']=float(max(0.,absolute[0]-d['pre_policy_delta_rad'][-1])) if meta['controller']!='fixed' else float(absolute[0])
            row['zero_load_post_residual_max_abs_rad']=float(abs(d['controller_delta_rad'][t>=.45]).max())
            if verify:
                assert d['tactile'].shape==(180,4,12,32) and d['mcf'].shape==(180,4,8)
                assert all(np.isfinite(value).all() for value in d.values())
                assert np.array_equal(d['input_mcf_before'][1:],d['mcf'][:-1])
                assert np.allclose(np.diff(d['observation_time_s']),1/120)
                assert np.all(d['controller_delta_rad'][t<.45-1e-9]==0)
                assert absolute.min()>=-1e-10 and absolute.max()<=.003+1e-10
                assert abs(row['pre_normal_n']-row['preload_target_n'])<=.02,(group,tid,row['pre_normal_n'])
                assert np.ptp(d['action'][t<.45-1e-9,14:16],axis=0).max()==0
                if meta['controller']=='fixed': assert np.all(d['controller_delta_rad']==0)
        states[tid]={key:d[key].copy() for key in ('initial_object_pose','initial_robot_body_pose','initial_tactile','action','force_world_n','torque_com_world_nm','step_start_s')}
        if row['trial']['direction']=='zero':
            zero_trace={key:d[key] for key in ('step_start_s','controller_delta_rad','raw_policy_delta_rad','pre_policy_delta_rad') if key in d}
        hashes[group+'/'+tid+'.npz']=meta['npz_sha256']; rows.append(row)
    return rows,states,hashes,zero_trace

def summarize(rows):
    rr=[r for r in rows if r['trial']['direction']!='zero' and r['repeat']==0]
    return dict(safe_cells=sum(r['safe'] for r in rr),total_cells=len(rr),safe_percent=100*np.mean([r['safe'] for r in rr]),
        safe_1mm=sum(r['safe_1mm'] for r in rr),safe_3mm=sum(r['safe_3mm'] for r in rr),
        pre_normal_n=float(np.mean([r['pre_normal_n'] for r in rr])),mean_effort_n_s=float(np.mean([r['effort_normal_n_s'] for r in rr])),
        mean_peak_normal_n=float(np.mean([r['peak_normal_n'] for r in rr])),mean_post_normal_n=float(np.mean([r['post_normal_n'] for r in rr])),
        max_slip_mm=max(r['max_slip_mm'] for r in rr),mean_saturation_seconds=float(np.mean([r['saturation_fraction']*1.5 for r in rr])),
        drops=sum(r['dropped'] for r in rr),by_direction={direction:sum(r['safe'] for r in rr if r['trial']['direction']==direction) for direction in DIRS})

def nondominated(points, cost):
    return [name for name,p in points.items() if not any(q[cost]<=p[cost] and q['safe_cells']>=p['safe_cells'] and
        (q[cost]<p[cost] or q['safe_cells']>p['safe_cells']) for other,q in points.items() if other!=name)]

def plot_pareto(points, out, title, static):
    many=len(points)>4
    fig,axes=plt.subplots(1,2,figsize=(12,6 if many else 5))
    letters={name:chr(65+i) for i,name in enumerate(points)}
    handles=[]
    for ax,cost,label in zip(axes,['pre_normal_n','mean_effort_n_s'],['Pre-load measured normal force (N)','Mean full-trial normal effort (N s)']):
        frontier=nondominated(points,cost)
        for name,p in points.items():
            kind=next((k for k in COLORS if name.endswith(k)), 'fixed')
            handle=ax.scatter(p[cost],p['safe_percent'],s=85,marker='s' if name in static else 'o',color=COLORS[kind],zorder=3,
                edgecolor='black' if name in frontier else 'white',linewidth=1.2)
            offset=(5,18) if 'mcf' in name else (5,-18)
            if name.endswith('fixed'): offset=(8,-5) if many else (-85,-5)
            if name=='high': offset=(-90,-18)
            if many: offset=(8,8) if 'mcf' in name else (8,-12)
            text=letters[name] if many else f"{name}: {p['safe_cells']}/{p['total_cells']}"
            ax.annotate(text,(p[cost],p['safe_percent']),xytext=offset,textcoords='offset points',fontsize=9)
            if cost=='pre_normal_n': handles.append((handle,f"{letters[name]}: {name} ({p['safe_cells']}/{p['total_cells']})"))
        ss=sorted((points[n] for n in static),key=lambda p:p[cost])
        ax.plot([p[cost] for p in ss],[p['safe_percent'] for p in ss],'--',color='#999999',alpha=.7,label='Fixed settings: visual guide only')
        ax.set_xlabel(label); ax.set_ylabel('Measured safe cells (%)'); ax.set_ylim(40,102); ax.margins(x=.18); ax.grid(alpha=.15); ax.legend(frameon=False,fontsize=8,loc='lower right')
    fig.suptitle(title+'\nSquares: fixed grip; circles: feedback; black outline: sample non-dominated',fontsize=11)
    if many: fig.legend([h for h,_ in handles],[label for _,label in handles],loc='lower center',ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.13 if many else 0,1,.90)); fig.savefig(out.with_suffix('.png'),dpi=190); fig.savefig(out.with_suffix('.pdf')); plt.close(fig)

def phase_figures(rows, summaries, out):
    for level in ('low','high'):
        for direction in DIRS:
            fig,axes=plt.subplots(1,3,figsize=(12,4.3),sharex=True,sharey=True)
            for kind,ax in zip(('fixed','scalar','mcf'),axes):
                group=level+'_'+kind
                chosen=[r for r in rows if r['group']==group and r['trial']['direction']==direction and r['repeat']==0]
                grid=np.zeros((7,5),bool)
                for r in chosen:
                    f=r['trial']['amplitude_n']; e=r['trial']['point_m'][2]*1000
                    i=int(np.argmin(abs(FORCES-f))); j=int(np.argmin(abs(OFFSETS-e))); grid[i,j]=r['safe']
                ax.imshow(grid.astype(int),origin='lower',aspect='auto',vmin=0,vmax=1,cmap=matplotlib.colors.ListedColormap(['#d95f55','#208678']),extent=(-2.5,22.5,.0875,.2625))
                for r in chosen: ax.text(r['trial']['point_m'][2]*1000,r['trial']['amplitude_n'],f"{r['max_slip_mm']:.2f}",ha='center',va='center',fontsize=7.5,color='white')
                summaries[group].setdefault('area_by_direction',{})[direction]=dict(all_n_mm=area_proxy(grid),high_e_n_mm=area_proxy(grid,10))
                ax.set_title(f"{kind}: {grid.sum()}/35 safe"); ax.set_xticks(OFFSETS); ax.set_yticks(FORCES); ax.set_xlabel('Offset e (mm)')
            axes[0].set_ylabel('External force F (N)')
            fig.suptitle(f"{direction} | matched pre-load {summaries[level+'_fixed']['pre_normal_n']:.3f} N\nMaximum relative slip (mm); colors use unrounded <=2 mm; same actuator limit",fontsize=11)
            fig.tight_layout(rect=(0,0,1,.86)); token=direction.replace('+','p').replace('-','n')
            fig.savefig(out/f'aligned_{level}_{token}.png',dpi=180); plt.close(fig)

def plot_anchor(source,static_reference,out,summaries):
    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True)
    labels={'none':'GRU only','pulse':'Early pulse','latch':'Confirmed latch'}
    for col,level in enumerate(('low','high')):
        for variant in ('none','pulse','latch'):
            name=f'trigger_{level}_{variant}'; tid='trigger_pFx_f0.200_e0.015'
            _,d,slip,normal=read(source/name,tid); t=d['observation_time_s']
            axes[0,col].plot(t,slip,label=labels[variant],color=COLORS[variant]); axes[1,col].plot(t,normal,color=COLORS[variant])
            axes[2,col].plot(d['step_start_s'],d['absolute_extra_close_rad']*1000,color=COLORS[variant])
        _,d,slip,normal=read(static_reference,'trigger_pFx_f0.200_e0.015')
        axes[0,col].plot(d['observation_time_s'],slip,label='Fixed High-Grip',color=COLORS['fixed'],linestyle='--')
        axes[1,col].plot(d['observation_time_s'],normal,color=COLORS['fixed'],linestyle='--')
        axes[2,col].plot(d['step_start_s'],d['absolute_extra_close_rad']*1000,color=COLORS['fixed'],linestyle='--')
        for ax in axes[:,col]: ax.axvspan(.55,.80,color='#e2edf2',alpha=.6); ax.grid(alpha=.15); ax.set_xlim(.35,1.5)
        axes[0,col].axhline(2,color='#b45343',linestyle='--')
        axes[0,col].set_title(f"Calibrated preload: {summaries[f'trigger_{level}_none']['pre_normal_n']:.3f} N")
        axes[0,col].legend(frameon=False,fontsize=9)
        axes[2,col].set_xlabel('Simulation time (s)')
    for row in range(3):
        values=np.concatenate([np.asarray(line.get_ydata()) for ax in axes[row] for line in ax.lines])
        lo=float(values.min()); hi=float(values.max())
        if row==2: lo=0.
        pad=.05*(hi-lo)
        for ax in axes[row]: ax.set_ylim(lo-pad,hi+pad)
    for ax,label in zip(axes[:,0],['Relative slip (mm)','Physical normal force (N)','Absolute extra closure (mrad)']): ax.set_ylabel(label)
    fig.suptitle('+Fx 0.20 N / e15 mm | causal trigger waveform and measured response')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out/'trigger_anchor.png',dpi=180); fig.savefig(out/'trigger_anchor.pdf'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('source',type=Path); p.add_argument('--report',type=Path,required=True)
    p.add_argument('--static-reference',type=Path,default=ROOT/'outputs/trigger_static_reference_20261014_01')
    p.add_argument('--original',type=Path,default=ROOT/'outputs/phase_20261014_01'); a=p.parse_args()
    a.report.mkdir(exist_ok=True,parents=True)
    freeze=json.loads((ROOT/'alignment_matrices/freeze.json').read_text())
    for name,digest in freeze.items(): assert sha(ROOT/name)==digest,('freeze changed',name)
    pipeline=json.loads((a.source/'pipeline.json').read_text()); assert pipeline['status']=='completed'
    rows=[]; hashes={}; states={}; summaries={}; zero_traces={}
    for stage in pipeline['stages']:
        name=stage['name']; rr,ss,hh,zz=load_group(a.source/name,name)
        rows+=rr; states[name]=ss; hashes.update(hh); summaries[name]=summarize(rr); zero_traces[name]=zz
    assert len(rows)==1163
    for name,digest in json.loads((ROOT/'trigger_static_reference_freeze.json').read_text()).items(): assert sha(ROOT/name)==digest
    rr,ss,hh,zz=load_group(a.static_reference,'trigger_max_fixed')
    assert len(rr)==27
    rows+=rr; states['trigger_max_fixed']=ss; hashes.update(hh); summaries['trigger_max_fixed']=summarize(rr); zero_traces['trigger_max_fixed']=zz
    for level in ('low','high'):
        for kind in ('scalar','mcf'):
            for tid,actual in states[level+'_'+kind].items():
                fixed=states[level+'_fixed'][tid]
                for key in actual:
                    # Torques are evaluated in a fixed frame but COM paths differ;
                    # externally specified torque is independent of that path.
                    val=actual[key][:54] if key=='action' else actual[key]
                    ref=fixed[key][:54] if key=='action' else fixed[key]
                    assert np.allclose(val,ref,rtol=0,atol=1e-8),(level,kind,tid,key)
        for variant in ('pulse','latch'):
            for tid,actual in states[f'trigger_{level}_{variant}'].items():
                for key,val in actual.items():
                    ref=states[f'trigger_{level}_none'][tid][key]
                    if key=='action': val,ref=val[:54],ref[:54]
                    assert np.allclose(val,ref,rtol=0,atol=1e-8),(level,variant,tid,key)
    for tid,actual in states['trigger_max_fixed'].items():
        for key in ('force_world_n','torque_com_world_nm','step_start_s'):
            assert np.array_equal(actual[key],states['trigger_low_none'][tid][key])
    # Old reports, old model and archived original scripts must remain byte-identical.
    for name,digest in json.loads((ROOT/'reflex_artifact_manifest.json').read_text()).items(): assert sha(ROOT/name)==digest,name
    old=[]; old_summaries={}
    for name in ('nominal','high','scalar','mcf'):
        rr,_,hh,_=load_group(a.original/name,name,False); old+=rr; old_summaries[name]=summarize(rr)
    comparisons={}
    for level in ('low','high'):
        sr={r['id']:r for r in rows if r['group']==level+'_scalar' and r['repeat']==0 and r['trial']['direction']!='zero'}
        mr={r['id']:r for r in rows if r['group']==level+'_mcf' and r['repeat']==0 and r['trial']['direction']!='zero'}
        common=[tid for tid in sr if sr[tid]['safe'] and mr[tid]['safe']]
        comparisons[level]=dict(mcf_minus_scalar_safe_cells=summaries[level+'_mcf']['safe_cells']-summaries[level+'_scalar']['safe_cells'],
            common_safe_cells=len(common),scalar_common_effort_n_s=float(np.mean([sr[tid]['effort_normal_n_s'] for tid in common])),
            mcf_common_effort_n_s=float(np.mean([mr[tid]['effort_normal_n_s'] for tid in common])),
            scalar_only_safe=[tid for tid in sr if sr[tid]['safe'] and not mr[tid]['safe']],
            mcf_only_safe=[tid for tid in sr if mr[tid]['safe'] and not sr[tid]['safe']])
    anchors=[]
    for r in rows:
        if r['repeat']==1:
            match=next(x for x in rows if x['group']==r['group'] and x['id']==r['id'].replace('_repeat1',''))
            anchors.append(dict(group=r['group'],id=r['id'],slip_difference_mm=r['max_slip_mm']-match['max_slip_mm'],
                effort_difference_n_s=r['effort_normal_n_s']-match['effort_normal_n_s']))
    plot_pareto(old_summaries,a.report/'pareto_original','Original frozen policies | 140 identical sampled loads',{'nominal','high'})
    aa={k:v for k,v in summaries.items() if not k.startswith('trigger_')}
    plot_pareto(aa,a.report/'pareto_aligned','Pre-load alignment interventions | 140 identical sampled loads',{'low_fixed','high_fixed','max_fixed'})
    phase_figures(rows,summaries,a.report)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for level,ax in zip(('low','high'),axes):
        names=[f'trigger_{level}_{v}' for v in ('none','pulse','latch')]+['trigger_max_fixed']
        ax.bar(range(4),[summaries[n]['safe_cells'] for n in names],color=[COLORS[v] for v in ('none','pulse','latch','fixed')])
        for i,n in enumerate(names): ax.text(i,summaries[n]['safe_cells']+.3,f"{summaries[n]['safe_cells']}/24\n{summaries[n]['mean_effort_n_s']:.2f} N s",ha='center',fontsize=9)
        ax.set_xticks(range(4),['GRU','Pulse','Latch','High-Grip']); ax.set_ylim(0,27); ax.set_title(level+' preload | High-Grip at 5.07 N'); ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Safe cells (slip <= 2 mm)'); fig.suptitle('Independent trigger development matrix | +/-Fx, e >= 10 mm, F=0.18..0.25 N')
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(a.report/'trigger_comparison.png',dpi=180); plt.close(fig)
    plot_anchor(a.source,a.static_reference,a.report,summaries)
    zeros=[r for r in rows if r['trial']['direction']=='zero']
    results=dict(summaries=summaries,original_summaries=old_summaries,matched_comparisons=comparisons,zero_load_controls=zeros,
        repeat_anchors=anchors,rows=rows,empirical_frontiers={cost:nondominated(aa,cost) for cost in ('pre_normal_n','mean_effort_n_s')})
    (a.report/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    flat=[{key:r[key] for key in ('group','id','safe','max_slip_mm','pre_normal_n','effort_normal_n_s','peak_normal_n','saturation_fraction')} for r in rows]
    with (a.report/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    verification=dict(status='passed',trials=len(rows),primary_trials=1163,auxiliary_static_reference_trials=27,synchronized_frames=len(rows)*180,paired_initial_states=True,
        causal_tactile_inputs=True,preload_clamp=True,physical_preload_tolerance_n=.02,absolute_actuator_limits=True,
        frozen_models_and_archived_files_unchanged=True,data_hashes=hashes)
    (a.report/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    lines=['# A/B/C 实验结果','',f'完成 {len(rows)} 个新正式物理试次、{len(rows)*180} 个同步记录步。原相图与模型保持原字节。',
        '', '## A：等实测预紧力', '', '| 组别 | 静息力 N | 安全格 / 140 | +Fx | −Fx | +Fy | −Fy | 平均力积分 N·s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in aa.items(): lines.append(f"| {name} | {s['pre_normal_n']:.4f} | {s['safe_cells']} | "+' | '.join(str(s['by_direction'][d]) for d in DIRS)+f" | {s['mean_effort_n_s']:.4f} |")
    for level,c in comparisons.items():
        lines+=['',f"{level}：MCF−Scalar 安全格差 {c['mcf_minus_scalar_safe_cells']:+d}；共同安全 {c['common_safe_cells']} 格上的平均力积分：Scalar {c['scalar_common_effort_n_s']:.4f}、MCF {c['mcf_common_effort_n_s']:.4f} N·s。"]
    lines+=['','## B：成本—安全图','','原模型的 3.06 / 4.42 N 静息差异及四组全部结果保留。原图中的 Scalar 是反馈策略，不作为静态握力参照；新图静态参照来自三组完整 Fixed 网格。黑边点仅表示当前样本中的非支配点。',
        '', '![原策略成本—安全图](pareto_original.png)', '', '![校准后成本—安全图](pareto_aligned.png)', '', '## C：触发与保持',
        '', '| 组别 | 安全格 / 24 | 平均积分 N·s | 平均饱和时长 s | 最大位移 mm | 零载触发时间 s |', '|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        if name.startswith('trigger_'):
            z=next(r for r in zeros if r['group']==name)
            lines.append(f"| {name} | {s['safe_cells']} | {s['mean_effort_n_s']:.4f} | {s['mean_saturation_seconds']:.4f} | {s['max_slip_mm']:.4f} | {z['trigger_time_s']} |")
    lines+=['','![独立触发对照](trigger_comparison.png)','','![困难格点响应](trigger_anchor.png)',
        '', '## 解释边界', '', '本轮检验冻结模型在新初态与基线扣除下的部署行为，不能把部署方式的失败自动推成表征无价值。原始学生在静息输出不为零且隐藏状态继续演化；零载控制中的释放后漂移也完整记录。',
        '', '触发试验的短脉冲和锁存采用不同确认帧数；它们是两个固定工程干预，不能把差异单独归因于保持时长。加载起点改为 0.55 s，触发判断只读过去触觉，记录加载前误触发。',
        '', '固定一个夹持快照、每格单次确定性试验，重复仅为两个锚点。网格安全率与非支配关系不代表统计总体泛化或真机安全。SDF 量仍是响应代理；图中 N 来自独立物理接触法向投影。',
        '', '给定公式 π(obs)−π(obs_pre) 保留了冻结学生的 [0,3] mrad 有界输出头。虽然机械绝对限幅一致，不同基线预测仍导致不同的策略可达残差范围；每条记录同时列出 centered_policy_reachable_min/max_extra_rad，不能把部署对照单独视为表征机制的唯一因果证明。',
        '', '所有失败格、重复差异、初态校准、触发时刻、绝对饱和、峰值握力和尾部握力均见 results.json / metrics.csv。未经校准的 N 峰值仅作接触代价，不能直接称为物体损伤。']
    (a.report/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ALIGNMENT_VERIFIED',len(rows),len(rows)*180,'frames')
    print(json.dumps({k:{x:v[x] for x in ('safe_cells','pre_normal_n','mean_effort_n_s')} for k,v in summaries.items()},indent=2))

if __name__=='__main__': main()
