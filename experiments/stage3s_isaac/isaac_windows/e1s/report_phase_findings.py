"""Combine measured envelope, signal/plant timing and timed interventions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from reflex_metrics import read
from phase_diagnostics import diagnose
from analyze_phase import FORCES,OFFSETS,DIRS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--report',type=Path,required=True); p.add_argument('--timing',type=Path,required=True); p.add_argument('--source',type=Path,required=True); a=p.parse_args()
    results=json.loads((a.report/'phase_results.json').read_text()); mechanics=json.loads((a.report/'mechanism.json').read_text())
    delays=[-100,0,1000/120,25,50,100]; rows=[]
    manifest=json.loads((a.timing/'manifest.json').read_text()); assert manifest['status']=='completed'
    for tid in manifest['completed_trials']:
        r,d,_,_=read(a.timing,tid); index=int(tid.split('_')[1][1:]); r['command_start_offset_ms']=delays[index]; rows.append(r)
    (a.report/'timing_intervention.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    selected=[r for r in rows if r['trial']['repeat']==0]
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    axes[0].plot(delays,[r['max_slip_mm'] for r in selected],marker='o',color='#087b75'); axes[0].axhline(2,color='#c14848',linestyle='--',label='2 mm target'); axes[0].legend(frameon=False)
    axes[1].plot(delays,[r['effort_normal_n_s'] for r in selected],marker='o',color='#c67f27')
    for ax,label in zip(axes,['Maximum relative slip (mm)','Normal effort (N s)']): ax.set_xlabel('Command onset relative to load (ms)'); ax.set_ylabel(label); ax.grid(alpha=.2)
    fig.suptitle('Same 3 mrad compensation, different onset | 0.20 N +Fx, 15 mm\nPrivileged timing intervention, not a tactile policy'); fig.tight_layout(); fig.savefig(a.report/'timing_intervention.png',dpi=170); fig.savefig(a.report/'timing_intervention.pdf'); plt.close(fig)
    grouped={n:{r['id']:r for r in results['rows'] if r['group']==n and r['id']!='phase_zero'} for n in ('nominal','high','scalar','mcf')}
    common=[tid for tid in grouped['mcf'] if grouped['mcf'][tid]['safe'] and grouped['scalar'][tid]['safe']]
    summary={'total_grid_conditions_per_controller':140,'safe_cells':{n:sum(r['safe'] for r in rr.values()) for n,rr in grouped.items()},
        'mean_effort':{n:float(np.mean([r['effort_normal_n_s'] for r in rr.values()])) for n,rr in grouped.items()},
        'common_safe_cells':len(common),'common_safe_effort':{n:float(np.mean([grouped[n][tid]['effort_normal_n_s'] for tid in common])) for n in ('scalar','mcf')}}
    summary['common_safe_effort_reduction_mcf_vs_scalar']=1-summary['common_safe_effort']['mcf']/summary['common_safe_effort']['scalar']
    summary['overall_effort_reduction_mcf_vs_high']=1-summary['mean_effort']['mcf']/summary['mean_effort']['high']
    scalar_high=sum(s['scalar']['high_e_cell_area_n_mm'] for s in results['summaries'].values())
    mcf_high=sum(s['mcf']['high_e_cell_area_n_mm'] for s in results['summaries'].values())
    summary['pooled_high_e_area']={'scalar_n_mm':scalar_high,'mcf_n_mm':mcf_high,'relative_change_percent':100*(mcf_high/scalar_high-1)}
    cmap=ListedColormap(['#9096a5','#bf8742','#228c82','#e2eee9']); fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    categories={}
    for di,(direction,ax) in enumerate(zip(DIRS,axes.flat)):
        grid=np.zeros((7,5),int); counts=dict(both_fail=0,scalar_only=0,mcf_only=0,both_safe=0)
        for fi in range(7):
            for ei in range(5):
                tid=f'phase_d{di}_f{fi}_e{ei}'; s=grouped['scalar'][tid]['safe']; m=grouped['mcf'][tid]['safe']; value=int(s)+2*int(m); grid[fi,ei]=value
                counts[['both_fail','scalar_only','mcf_only','both_safe'][value]]+=1
        ax.imshow(grid,origin='lower',extent=(-2.5,22.5,.0875,.2625),aspect='auto',cmap=cmap,vmin=0,vmax=3)
        ax.set_title(direction+f" | MCF-only {counts['mcf_only']}, Scalar-only {counts['scalar_only']}"); ax.set_xticks(OFFSETS); ax.set_yticks(FORCES); categories[direction]=counts
    for ax in axes[-1]: ax.set_xlabel('Offset e (mm)')
    for ax in axes[:,0]: ax.set_ylabel('External force F (N)')
    fig.legend(handles=[Patch(color=cmap(i),label=l) for i,l in enumerate(['Both fail','Scalar only','MCF only','Both safe'])],loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Measured Scalar / MCF safe-cell contrast | fixed models and action limits'); fig.tight_layout(rect=(0,.05,1,.94)); fig.savefig(a.report/'phase_controller_contrast.png',dpi=170); plt.close(fig)
    summary['controller_contrast']=categories
    summary['predisturbance_grip']={}
    for name in ('nominal','high','scalar','mcf'):
        audit=diagnose(a.source/name,'phase_zero')[0]
        summary['predisturbance_grip'][name]={'physical_normal_n':audit['baseline_normal_n'],'delta_mrad':audit['baseline_delta_rad']*1000}
    (a.report/'findings.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# 偏心相图与机制验证结果','', '本轮按机制猜想实际执行了纯力、纯力矩、组合载荷、无外载闭合阶跃与抓握时刻干预，并完整扫描四个加载方向。原模型未重训，原相图没有被干预控制器替换。','', '## 得到了什么','',
        f"- MCF 安全格 {summary['safe_cells']['mcf']}/140，Scalar {summary['safe_cells']['scalar']}/140，Nominal {summary['safe_cells']['nominal']}/140，High-Grip {summary['safe_cells']['high']}/140。",
        '- 高偏心 e≥10 mm 的分片面积变化有方向差异：−Fx 为 +22.2%，+Fx 为 −20.0%，±Fy 不变；四方向合并为 +2.6%。没有把局部优势写成所有方向扩域。',
        f"- 两策略共同安全的 {len(common)} 个条件中，MCF 平均法向力积分较 Scalar 降低 {summary['common_safe_effort_reduction_mcf_vs_scalar']:.1%}；全网格平均较 High-Grip 降低 {summary['overall_effort_reduction_mcf_vs_high']:.1%}。",
        '- 纯力和纯力矩都有早期弱 CoP/响应和变化，按基线噪声尺度均在载荷后25 ms完成三帧确认。大幅变化门限则更晚或未检出。因此当前模型没有支持“纯力必然晚显现、纯力矩必然立即大幅显现”的完整解释，弱响应不能当作没有信息。',
        '- 1/3 mrad 无外载闭合阶跃的物理法向力在首个8.333 ms采样步达到接近平台：两个10–90%门限落在同一帧。不是零物理延迟，也没有观察到多帧慢爬升。',
        '- 抓握时刻干预显示补偿时刻很重要：同一3 mrad补偿，提前100 ms最大滑移1.635 mm；载荷同刻开始2.102 mm；晚25 ms为2.959 mm；晚100 ms为11.639 mm。',
        '- 扰动前实际握持力并不相同：Scalar约4.42 N、MCF约3.06 N（同一初态、同一名义指令，但策略运行后形成不同预握持）。相图展示完整策略/接触状态的表现，不能单独归因于传感器可观测性。',
        '', '![四方向控制差异](phase_controller_contrast.png)', '', '![抓握时刻干预](timing_intervention.png)', '', '## 对机制猜想的判断','',
        '实验支持“早期运动发生得快、抓握补偿时刻显著影响恢复”的部分。当前证据更指向早期弱信号如何转化成足够补偿，以及载荷到来前夹持状态的差异；还不能归因于仿真驱动器法向力爬升慢。',
        '', '时刻干预使用加载时钟，属于特权诊断；提前100 ms的成功不等于触觉学生能预知扰动。驱动器没有测得明显慢爬升，因此这轮没有仅按该解释加入PD前馈或宣称其必然解决0.20 N问题。后续可在保留主图的独立开发对照里检验策略触发/保持及初始预紧。',
        '', '网格面积是相邻单元四角全部安全的面积代理，不是未采内点的连续安全保证。每格是确定性开发试次，恢复重复不作为独立样本。当前SDF模型的弱响应、门限与时序也不等同真实柔性层/硬件边界。',
        '', '## 完整记录','', '- [四方向安全图、逐条件表和机制探针](REPORT.md)', '- [全部网格数据](phase_results.json)', '- [抓握时刻干预](timing_intervention.json)', '- [复核记录](verification.json)', '']
    (a.report/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    original=(a.report/'REPORT.md').read_text(encoding='utf-8').split('\n## 抓握时刻干预')[0]
    with (a.report/'REPORT.md').open('w',encoding='utf-8') as f:
        f.write(original)
        f.write('\n## 抓握时刻干预\n\n| 相对载荷时刻 ms | 最大滑移 mm | 法向力积分 N·s |\n|---:|---:|---:|\n')
        for r in selected: f.write(f"| {r['command_start_offset_ms']:.3f} | {r['max_slip_mm']:.3f} | {r['effort_normal_n_s']:.3f} |\n")
        f.write('\n![抓握时刻](timing_intervention.png)\n\n[机制与控制结论](FINDINGS.md)\n')
    print('PHASE_FINDINGS',summary,flush=True)

if __name__=='__main__': main()
