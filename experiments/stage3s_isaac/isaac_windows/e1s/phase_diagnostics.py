"""Load / observable-feature / actual motion / command / plant event alignment."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reflex_metrics import read


def first_hit(values,t,onset,threshold,persistence=1):
    hits=(np.asarray(values)>=threshold)&(t>=onset)
    run=0
    for i,v in enumerate(hits):
        run=run+1 if v else 0
        if run>=persistence: return float(t[i])
    return None


def noise_threshold(values,base,floor):
    x=np.asarray(values)[base]; med=np.median(x)
    return float(max(floor,med+6*1.4826*np.median(np.abs(x-med))))


def diagnose(root,tid,cop_floor=.002,slip_floor_mm=.1):
    row,d,slip,normal=read(root,tid); t=d['observation_time_s']; onset=row['trial']['onset_s']
    base=(t>=.20)&(t<onset-.025); assert base.sum()>=10
    s=d['state4'][:,2:4]; baseline=np.median(s[base],axis=0)
    cop=np.linalg.norm(s[:,:,1:3]-baseline[None,:,1:3],axis=2).max(axis=1)
    area=np.abs(s[:,:,3]-baseline[None,:,3]).max(axis=1)
    proxy=np.abs(s[:,:,0]-baseline[None,:,0]).max(axis=1)
    rawdiff=np.linalg.norm((d['tactile'][:,2:4]-np.median(d['tactile'][base,2:4],axis=0)).reshape(len(t),-1),axis=1)
    delta=d.get('controller_delta_rad',np.zeros(len(t))); db=float(np.median(delta[base]))
    joint=(d['joint_pos'][:,15]-d['joint_pos'][:,14])/2; jb=float(np.median(joint[base])); nb=float(np.median(normal[base]))
    thresholds={'cop':noise_threshold(cop,base,cop_floor),'area':noise_threshold(area,base,1/384),
        'proxy':noise_threshold(proxy,base,.05),'raw':noise_threshold(rawdiff,base,.002)}
    events={k:first_hit(v,t,onset,thresholds[k],3) for k,v in [('cop',cop),('area',area),('proxy',proxy),('raw',rawdiff)]}
    first_cross={k:first_hit(v,t,onset,thresholds[k],1) for k,v in [('cop',cop),('area',area),('proxy',proxy),('raw',rawdiff)]}
    noise_thresholds={k:noise_threshold(v,base,1e-6) for k,v in [('cop',cop),('area',area),('proxy',proxy),('raw',rawdiff)]}
    noise_events={k:first_hit(v,t,onset,noise_thresholds[k],3) for k,v in [('cop',cop),('area',area),('proxy',proxy),('raw',rawdiff)]}
    events['slip']=first_hit(slip-np.median(slip[base]),t,onset,slip_floor_mm)
    events['slip_2mm']=first_hit(slip,t,onset,2.)
    events['command']=first_hit(delta-db,d['step_start_s'],onset,.0001)
    events['actual_closure']=first_hit(joint-jb,t,onset,.00005)
    events['normal_force']=first_hit(normal-nb,t,onset,.1)
    tactile=[events[k] for k in ('cop','area','proxy','raw') if events[k] is not None]
    events['any_tactile']=min(tactile) if tactile else None
    onset_actual=float(d['step_start_s'][np.flatnonzero((np.linalg.norm(d['force_world_n'],axis=1)>0)|(np.linalg.norm(d['torque_com_world_nm'],axis=1)>0))[0]]) if np.any(d['force_world_n']) or np.any(d['torque_com_world_nm']) else onset
    out=dict(row,events_s=events,event_delay_ms={k:(v-onset_actual)*1000 if v is not None else None for k,v in events.items()},thresholds=thresholds,
        first_cross_delay_ms={k:(v-onset_actual)*1000 if v is not None else None for k,v in first_cross.items()},
        noise_only_delay_ms={k:(v-onset_actual)*1000 if v is not None else None for k,v in noise_events.items()},noise_only_thresholds=noise_thresholds,
        controller_output='delta_g_only',baseline_normal_n=nb,baseline_delta_rad=db,
        cop_floor=cop_floor,slip_floor_mm=slip_floor_mm)
    return out,d,dict(cop=cop,area=area,proxy=proxy,raw=rawdiff,slip=slip,normal=normal,delta=delta,joint_delta=(joint-jb)*1000)


def plot_case(roots,tid,out):
    colors={'nominal':'#748094','high':'#e69a35','mcf':'#087b75','scalar':'#8c6abd','oracle':'#2c5ea8'}
    fig,ax=plt.subplots(5,1,figsize=(10,11),sharex=True)
    for name,root in roots.items():
        row,d,z=diagnose(root,tid); t=d['observation_time_s']; color=colors[name]
        if name=='mcf':
            ax[0].plot(d['step_start_s'],np.linalg.norm(d['force_world_n'],axis=1),color='#313b50',label='External force (N)')
            ax[0].plot(d['step_start_s'],np.linalg.norm(d['torque_com_world_nm'],axis=1)*1000,color='#be5d3c',label='COM moment (N mm)')
            ax[1].plot(t,z['cop'],color=color,label='CoP displacement'); ax[1].plot(t,z['area'],color='#bf6c94',label='Area displacement')
        ax[2].plot(t,z['slip'],color=color,label=name)
        ax[3].plot(d['step_start_s'],z['delta']*1000,color=color,label=name+' command')
        if name=='mcf': ax[3].plot(t,z['joint_delta'],color=color,linestyle='--',label='MCF realized closure')
        ax[4].plot(t,z['normal'],color=color,label=name)
    onset=row['trial']['onset_s']; ending=onset+row['trial']['duration_s']
    for a,label in zip(ax,['Known excitation','Normal spatial features','Relative slip (mm)','Increment (mrad)','Normal force (N)']):
        a.set_ylabel(label); a.axvspan(onset,ending,color='#e2edf2',alpha=.6); a.grid(axis='y',alpha=.18)
    ax[2].axhline(2,color='#c14848',linestyle='--',linewidth=1)
    for a in ax: a.legend(frameon=False,loc='upper left',ncol=2,fontsize=8)
    ax[-1].set_xlabel('Simulation time (s)'); ax[-1].set_xlim(.25,1.2)
    fig.suptitle('Load → normal tactile observation → grip command → realized contact\nNo force-estimation head in this controller',fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out/'timing_020_fx.png',dpi=170); fig.savefig(out/'timing_020_fx.pdf'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(exist_ok=False,parents=True)
    root=Path(__file__).resolve().parent/'outputs'
    roots={n:root/f'reflex_stream_{n}_01' for n in ('nominal','high','mcf','scalar','oracle')}
    rows=[]; sensitivities=[]
    for name,directory in roots.items():
        for tid in ('test_00_0','test_01_0'):
            row,_,_=diagnose(directory,tid); rows.append(row)
            for cf in (.001,.002,.004):
                for sf in (.05,.1,.2):
                    rr,_,_=diagnose(directory,tid,cf,sf); sensitivities.append(dict(controller=name,id=tid,cop_floor=cf,slip_floor_mm=sf,delays=rr['event_delay_ms']))
    (a.out/'timing_results.json').write_text(json.dumps(dict(rows=rows,sensitivity=sensitivities),indent=2),encoding='utf-8')
    plot_case(roots,'test_01_0',a.out)
    for r in rows: print('TIMING',r['controller'],r['id'],r['event_delay_ms'],flush=True)

if __name__=='__main__': main()
