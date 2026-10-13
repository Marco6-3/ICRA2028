"""Predeclared causal probes on deduplicated and condition-held-out pilot traces."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

LABELS=['+Fx','-Fx','+Fy','-Fy']
METHODS=['state_now','mcf_now','raw_normalized_now']+[f'{mode}_w{w}' for mode in ('mean','delta_sum','mean_delta') for w in (4,8,16)]


def causal_features(data,method):
    s=np.asarray(data['state4'],dtype=float).reshape(len(data['state4']),-1)
    if method=='state_now':
        return s.copy()
    if method=='mcf_now':
        return np.asarray(data['mcf'],dtype=float).reshape(len(s),-1).copy()
    if method=='raw_normalized_now':
        raw=np.asarray(data['tactile'],dtype=float)
        return (raw/np.maximum(raw.sum(axis=(2,3),keepdims=True),1e-12)).reshape(len(s),-1)
    mode,ww=method.rsplit('_w',1); w=int(ww)
    output=[]
    for i in range(len(s)):
        mean=s[max(0,i-w+1):i+1].mean(axis=0)
        # W differences ending at current frame; at stream start, only the
        # available history is used. Evaluation begins after a full history.
        old=s[max(0,i-w)]
        delta=s[i]-old
        output.append(mean if mode=='mean' else delta if mode=='delta_sum' else np.r_[mean,delta])
    return np.stack(output)


def condition_group(tr):
    if tr['direction']=='zero': return 'zero'
    if tr['waveform']!='step': return tr['waveform']
    if tr['preload_delta_rad']!=0: return 'preload'
    if tr['friction_scale']!=1: return 'friction'
    if tr['sdf_gain_scale']!=1: return 'sdf_gain'
    return 'step_unseen_offset' if tr['point_m'][2]>0 else 'train'


def target_and_mask(data,reference):
    force=data['force_world_n']@reference
    torque=data['torque_com_world_nm']@reference
    axes=np.argmax(abs(force[:,:2]),axis=1)
    labels=axes*2+(force[np.arange(len(force)),axes]<0).astype(int)
    mask=np.linalg.norm(force,axis=1)>1e-8
    # Common complete history for ALL methods; no different test frames.
    mask &= np.arange(len(force))>=16
    return labels,mask,np.c_[force[:,:2],torque[:,:2]]


def fit_probe(x,y,target):
    mu=x.mean(axis=0); sd=x.std(axis=0); sd=np.where(sd>1e-8,sd,1.)
    xx=(x-mu)/sd
    centers=np.stack([xx[y==i].mean(axis=0) for i in range(4)])
    mean_target=target.mean(axis=0)
    # Dual ridge avoids a 1536-square solve for Raw; alpha predeclared.
    weights=xx.T@np.linalg.solve(xx@xx.T+np.eye(len(xx)),target-mean_target)
    return mu,sd,centers,mean_target,weights


def predict(model,x):
    mu,sd,centers,mean_target,weights=model
    xx=(x-mu)/sd
    pred=((xx[:,None]-centers[None])**2).sum(axis=2).argmin(axis=1)
    return pred,xx@weights+mean_target


def aggregate(items):
    acc=np.asarray([r['accuracy'] for r in items])
    rng=np.random.default_rng(20261013)
    samples=rng.integers(0,len(acc),(2000,len(acc)))
    boot=acc[samples].mean(axis=1)
    confusion=np.sum([r['confusion'] for r in items],axis=0)
    return {'condition_count':len(items),'macro_condition_accuracy':float(acc.mean()),
            'condition_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'frame_confusion_true_rows_pred_cols':confusion.tolist(),
            'rmse_FxFyMxMy':np.sqrt(np.mean([r['mse'] for r in items],axis=0)).tolist()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Preserve previous diagnostics; use a new output directory')
    args.out.mkdir(parents=True)
    provenance=json.loads((args.source/'provenance.json').read_text(encoding='utf-8'))
    manifest=json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed': raise RuntimeError('Source simulation incomplete')
    reference=np.asarray(provenance['reference_rotation_world_from_load'])
    records=[]; hashes={}
    for trial_id in manifest['completed_trials']:
        meta=json.loads((args.source/f'{trial_id}.json').read_text(encoding='utf-8'))
        if meta['trial']['repeat']!=0: continue
        sha=hashlib.sha256((args.source/f'{trial_id}.npz').read_bytes()).hexdigest()
        if sha!=meta['npz_sha256']: raise RuntimeError('Source hash mismatch: '+trial_id)
        hashes[trial_id]=sha
        with np.load(args.source/f'{trial_id}.npz',allow_pickle=False) as f:
            data={k:f[k] for k in ('state4','mcf','tactile','force_world_n','torque_com_world_nm','observation_time_s','loading_gain')}
        labels,mask,target=target_and_mask(data,reference)
        records.append({'id':trial_id,'trial':meta['trial'],'group':condition_group(meta['trial']),
                        'data':data,'labels':labels,'mask':mask,'target':target})
    train=[r for r in records if r['group']=='train']; tests=[r for r in records if r['group'] not in ('train','zero')]
    protocol={'methods':METHODS,'primary_method':'mean_delta_w8','train_ids':[r['id'] for r in train],
              'test_ids':[r['id'] for r in tests],'deduplication':'repeat 0 only','source_npz_sha256':hashes,
              'source_protocol_sha256':hashlib.sha256(Path(__file__).with_name('TEMPORAL_PROTOCOL.md').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'parameters':'fixed nearest centroid; ridge alpha=1; W=4/8/16; seed=20261013; no tuning'}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results={}; predictions={}
    for method in METHODS:
        encoding={r['id']:causal_features(r['data'],method) for r in records}
        x=np.concatenate([encoding[r['id']][r['mask']] for r in train])
        y=np.concatenate([r['labels'][r['mask']] for r in train])
        target=np.concatenate([r['target'][r['mask']] for r in train])
        model=fit_probe(x,y,target)
        trial_results=[]; prediction_arrays={}
        for r in tests:
            features=encoding[r['id']][r['mask']]
            truth=r['labels'][r['mask']]
            pred,reconstruction=predict(model,features)
            confusion=np.zeros((4,4),int)
            for a,b in zip(truth,pred): confusion[a,b]+=1
            phase=[]
            for sign in (1,-1):
                m=r['data']['loading_gain'][r['mask']]*sign>0
                if m.any(): phase.append({'loading_gain_sign':sign,'frames':int(m.sum()),'accuracy':float((truth[m]==pred[m]).mean())})
            trial_results.append({'id':r['id'],'group':r['group'],'accuracy':float((truth==pred).mean()),
                'confusion':confusion.tolist(),'mse':np.mean((reconstruction-r['target'][r['mask']])**2,axis=0).tolist(),
                'phases':phase,'wrong_frame_indices':np.flatnonzero(r['mask'])[truth!=pred].tolist()})
            prediction_arrays[r['id']]=np.c_[np.flatnonzero(r['mask']),truth,pred,reconstruction,r['target'][r['mask']]]
        latency=[]
        benchmark=x[:1]
        for i in range(220):
            begin=time.perf_counter_ns(); predict(model,benchmark); elapsed=time.perf_counter_ns()-begin
            if i>=20: latency.append(elapsed/1e6)
        results[method]={'dimension':int(x.shape[1]),'all':aggregate(trial_results),
            'groups':{g:aggregate([r for r in trial_results if r['group']==g]) for g in sorted({r['group'] for r in trial_results})},
            'trial_results':trial_results,'batch1_cpu_probe_p50_ms':float(np.quantile(latency,.5)),
            'batch1_cpu_probe_p95_ms':float(np.quantile(latency,.95))}
        np.savez_compressed(args.out/f'predictions_{method}.npz',**prediction_arrays)
        predictions[method]=trial_results
        print('TEMPORAL',method,'all',results[method]['all']['macro_condition_accuracy'],
              'ramp',results[method]['groups']['ramp']['macro_condition_accuracy'],
              'reverse',results[method]['groups']['reverse']['macro_condition_accuracy'],flush=True)
    # Paired difference, with the same held-out CONDITIONS as sampling units.
    primary=np.array([r['accuracy'] for r in predictions['mean_delta_w8']])
    old=np.array([r['accuracy'] for r in predictions['mcf_now']])
    difference=primary-old; rng=np.random.default_rng(20261013)
    boot=difference[rng.integers(0,len(difference),(2000,len(difference)))].mean(axis=1)
    paired={'comparison':'mean_delta_w8 minus mcf_now','mean_accuracy_difference':float(difference.mean()),
            'condition_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
    output={'status':'completed','train_conditions':len(train),'test_conditions':len(tests),
            'labels':LABELS,'results':results,'paired_primary':paired,
            'limits':'Development diagnostics within one simulator session; no deformable elastomer, real calibration, minimal sufficiency or GRU necessity established.'}
    (args.out/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    lines=['# 因果时序探针：去重与条件留出结果','',f'训练 {len(train)} 条件，测试 {len(tests)} 个未见条件；仅使用重复 0。未重跑仿真，原静态报告保留。',
           '', '原 0%/50% 是整段均值/分段均值探针结果，本表是当前方向的逐帧因果诊断；评价单位不同，不能直接算原值到新值的增益。历史与单帧比较在本表使用完全相同帧和划分。',
           '', '| 探针 | 维数 | 全条件 | 未见偏心 | 渐变 | 反向 | 摩擦 | 预载 | 读出增益 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,r in results.items():
        vals=[r['all']['macro_condition_accuracy']]+[r['groups'][g]['macro_condition_accuracy'] for g in ('step_unseen_offset','ramp','reverse','friction','preload','sdf_gain')]
        lines.append(f'| {method} | {r["dimension"]} | '+' | '.join(f'{v:.1%}' for v in vals)+' |')
    lines+=['',f"预指定 W=8 均值+差分累积相对同划分单帧 MCF 的条件平均准确率变化：{paired['mean_accuracy_difference']:+.1%}；成对条件 bootstrap 95% 区间：{paired['condition_bootstrap_ci95']}。",
        '', '全部窗口如实展示，不以测试最优窗口替代预指定窗口。差分累计等于状态端点之差，不增加信息论上的独立观测。probe 延迟只含标准化/分类/线性重构 CPU 计算，不含传感器、执行器或神经网络；不能视作 120 Hz 闭环时延或能耗。',
        '', '单帧和历史在同样条件划分下的结果可以作为后续 representation × memory 检验的动机；它们既不证明 GRU 必须，也不判定 MCF 无价值。真实柔性接触 H-Obs、正式 E2 和 E3 仍待相应实验。',
        '', '当前外部力/COM 力矩真值、试次标签与加载时序只用于监督和评价，没有进入探针输入。每方法的逐帧预测、真值、误判索引和反向两段结果已保存。', '']
    (args.out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5)); names=list(results)
    for g in ('ramp','reverse','friction'):
        ax.plot(range(len(names)),[results[m]['groups'][g]['macro_condition_accuracy'] for m in names],marker='o',label=g)
    ax.set_xticks(range(len(names)),names,rotation=35,ha='right'); ax.set_ylim(0,1)
    ax.set_ylabel('Condition macro accuracy'); ax.legend(); ax.set_title('Causal probes: deduplicated conditions, no tuning / GRU claim')
    fig.tight_layout(); fig.savefig(args.out/'comparison.png',dpi=150); plt.close(fig)
    print('PRIMARY_PAIRED',paired)


if __name__=='__main__': main()
