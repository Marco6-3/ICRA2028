"""Fixed-budget development E2 probes, not real calibration / control evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from temporal_probe import condition_group,target_and_mask

CONFIGS=[('scalar_t1','scalar',1,1.),('scalar_t16','scalar',16,1.),
         ('static_s4','static',1,1.),('mcf_t1','mcf',1,1.),('mcf_t16','mcf',16,1.),
         ('dense_t16','raw',16,1.),('s4_t16','s4',16,1.),
         ('mcf_no_cop_t16','no_cop',16,1.),('mcf_no_area_t16','no_area',16,1.),
         ('mcf_half','mcf',16,.5),('dense_half','raw',16,.5)]
CONFIGS.append(('dense_t17','raw',17,1.))
CONFIGS.append(('static_mcf','static_mcf',1,1.))
DIMS={'scalar':8,'static':16,'mcf':32,'s4':16,'no_cop':16,'no_area':24,'raw':32}
DIMS['static_mcf']=32
SEEDS=[7,11,19]


class Probe(nn.Module):
    def __init__(self,kind,hidden):
        super().__init__(); self.kind=kind; self.hidden=hidden
        if kind=='raw':
            self.encoder=nn.Sequential(nn.Conv2d(4,8,3,padding=1),nn.ReLU(),
                nn.Conv2d(8,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,8)),
                nn.Flatten(),nn.Linear(192,32),nn.ReLU())
        if kind.startswith('static'):
            self.spatial=nn.Sequential(nn.Linear(DIMS[kind],hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        else:
            self.gru=nn.GRU(DIMS[kind],hidden,num_layers=2,batch_first=True)
        self.head=nn.Linear(hidden,8)

    def forward(self,x):
        if self.kind=='raw':
            b,t=x.shape[:2]; x=self.encoder(x.reshape(b*t,4,12,32)).reshape(b,t,32)
        if self.kind.startswith('static'):
            z=self.spatial(x[:,-1])
        else:
            z,_=self.gru(x); z=z[:,-1]
        return self.head(z)


def matched_hidden(kind,budget=50000):
    choices=range(120,260) if kind.startswith('static') else range(32,97)
    counts={h:sum(p.numel() for p in Probe(kind,h).parameters()) for h in choices}
    h=min(choices,key=lambda h:abs(counts[h]-budget))
    return h,counts[h]


def representation(d,kind):
    if kind=='raw': return d['tactile'].astype(np.float32)
    if kind=='scalar': return d['scalar'].reshape(len(d['scalar']),8).astype(np.float32)
    if kind in ('static','s4'): return d['state4'].reshape(len(d['state4']),16).astype(np.float32)
    m=d['mcf']
    if kind=='no_cop': m=m[:,:,[0,3,4,7]]
    if kind=='no_area': m=m[:,:,[0,1,2,4,5,6]]
    return m.reshape(len(m),-1).astype(np.float32)


def evaluate(model,x,labels,targets,meta,scale,mean):
    outputs=[]
    with torch.inference_mode():
        for begin in range(0,len(x),64): outputs.append(model(x[begin:begin+64]).numpy())
    output=np.concatenate(outputs); pred=output[:,4:].argmax(axis=1)
    reconstruction=output[:,:4]*scale+mean
    trials=[]
    for tid,group in sorted(set(meta)):
        mask=np.array([item[0]==tid for item in meta])
        confusion=np.zeros((4,4),int)
        for a,b in zip(labels[mask],pred[mask]): confusion[a,b]+=1
        trials.append({'id':tid,'group':group,'accuracy':float((labels[mask]==pred[mask]).mean()),
                      'mse':np.mean((reconstruction[mask]-targets[mask])**2,axis=0).tolist(),
                      'confusion':confusion.tolist()})
    groups={}
    for group in sorted({r['group'] for r in trials}):
        selected=[r for r in trials if r['group']==group]
        groups[group]={'accuracy':float(np.mean([r['accuracy'] for r in selected])),
            'rmse_FxFyMxMy':np.sqrt(np.mean([r['mse'] for r in selected],axis=0)).tolist()}
    overall={'condition_macro_accuracy':float(np.mean([r['accuracy'] for r in trials])),
             'rmse_FxFyMxMy':np.sqrt(np.mean([r['mse'] for r in trials],axis=0)).tolist()}
    return overall,groups,trials,pred,reconstruction


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--profiles',nargs='+')
    parser.add_argument('--protocol',type=Path,default=Path(__file__).with_name('E2_DEV_PROTOCOL.md'))
    parser.add_argument('--normalization',choices=['legacy_zscore','robust_floor_clip8'],default='legacy_zscore')
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new output directory')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    provenance=json.loads((args.source/'provenance.json').read_text(encoding='utf-8'))
    manifest=json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed': raise RuntimeError('Source incomplete')
    reference=np.asarray(provenance['reference_rotation_world_from_load'])
    records=[]; data_hashes={}
    for tid in manifest['completed_trials']:
        meta=json.loads((args.source/f'{tid}.json').read_text(encoding='utf-8'))
        if meta['trial']['repeat']!=0 or meta['trial']['direction']=='zero': continue
        sha=hashlib.sha256((args.source/f'{tid}.npz').read_bytes()).hexdigest()
        if sha!=meta['npz_sha256']: raise RuntimeError('Source hash failed')
        data_hashes[tid]=sha
        with np.load(args.source/f'{tid}.npz',allow_pickle=False) as f:
            data={k:f[k] for k in ('tactile','scalar','state4','mcf','force_world_n','torque_com_world_nm')}
        labels,mask,target=target_and_mask(data,reference)
        records.append({'id':tid,'group':condition_group(meta['trial']),'data':data,
                        'labels':labels,'mask':mask,'target':target.astype(np.float32)})
    train=[r for r in records if r['group']=='train']; tests=[r for r in records if r['group']!='train']
    configs=[c for c in CONFIGS if (args.profiles is None and c[0] not in ('dense_t17','static_mcf')) or
             (args.profiles is not None and c[0] in args.profiles)]
    if not configs: raise ValueError('No known profiles')
    sizes={kind:matched_hidden(kind) for _,kind,_,_ in configs}
    protocol={'train_conditions':[r['id'] for r in train],'test_conditions':[r['id'] for r in tests],
              'source_hashes':data_hashes,'configs':configs,'seeds':SEEDS,'matched_sizes':sizes,
              'epochs':80,'batch_size':64,'adam_lr':.001,'weight_decay':.0001,'grad_clip':1.,
              'torch':torch.__version__,'threads':2,'device':'cpu','dtype':'float32',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'protocol_path':str(args.protocol)}
    protocol['normalization']=args.normalization
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    target_train=np.concatenate([r['target'][r['mask']] for r in train])
    target_mean=target_train.mean(axis=0); target_scale=np.maximum(target_train.std(axis=0),1e-8)
    all_runs=[]; latency={}; prediction_bank={}
    for name,kind,history,fraction in configs:
        frames={r['id']:representation(r['data'],kind) for r in records}
        fit_frames=np.concatenate([frames[r['id']] for r in train])
        if args.normalization=='robust_floor_clip8':
            import bounded_scaler
            mean,std=bounded_scaler.fit(fit_frames,kind)
            normalized={tid:bounded_scaler.transform(x,mean,std) for tid,x in frames.items()}
        else:
            mean=fit_frames.mean(axis=0); std=np.maximum(fit_frames.std(axis=0),1e-6)
            normalized={tid:(x-mean)/std for tid,x in frames.items()}
        def windows(rr):
            xx=[]; yy=[]; targets=[]; meta=[]
            for r in rr:
                indices=np.flatnonzero(r['mask'])
                if rr is train and fraction<1:
                    indices=indices[::2] # all conditions and loading phases covered
                for i in indices:
                    xx.append(normalized[r['id']][i-history+1:i+1]); yy.append(r['labels'][i])
                    targets.append(r['target'][i]); meta.append((r['id'],r['group']))
            return torch.tensor(np.stack(xx),dtype=torch.float32),np.asarray(yy),np.stack(targets),meta
        x,y,t,meta=windows(train); tx,ty,tt,tmeta=windows(tests)
        normalized_target=torch.tensor((t-target_mean)/target_scale,dtype=torch.float32)
        y_tensor=torch.tensor(y,dtype=torch.long)
        for seed in SEEDS:
            begin=time.perf_counter(); torch.manual_seed(seed); np.random.seed(seed)
            hidden,count=sizes[kind]; model=Probe(kind,hidden)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
            generator=torch.Generator().manual_seed(seed)
            for epoch in range(80):
                model.train(); order=torch.randperm(len(x),generator=generator)
                for start in range(0,len(x),64):
                    index=order[start:start+64]; optimizer.zero_grad(set_to_none=True)
                    out=model(x[index]); loss=nn.functional.mse_loss(out[:,:4],normalized_target[index])+nn.functional.cross_entropy(out[:,4:],y_tensor[index])
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                if epoch in (0,39,79): print('E2_TRAIN',name,seed,epoch+1,'loss',float(loss.detach()),flush=True)
            model.eval()
            overall,groups,trials,pred,reconstruction=evaluate(model,tx,ty,tt,tmeta,target_scale,target_mean)
            # Training fit is recorded too: poor test scores are not silently
            # reinterpreted as a physical impossibility if optimization failed.
            train_scores=evaluate(model,x,y,t,meta,target_scale,target_mean)[0]
            run={'name':name,'kind':kind,'history':history,'fraction':fraction,'seed':seed,
                'hidden':hidden,'parameters':count,'train_frames':len(x),'test_frames':len(tx),
                'train_scores':train_scores,'overall':overall,'groups':groups,'trials':trials,
                'training_wall_s':time.perf_counter()-begin,'final_train_batch_loss':float(loss.detach())}
            all_runs.append(run)
            torch.save({'state_dict':model.state_dict(),'kind':kind,'hidden':hidden,'history':history,
                        'mean':torch.tensor(mean),'std':torch.tensor(std),'normalization':args.normalization,
                        'input_clip':8. if args.normalization=='robust_floor_clip8' else None,
                        'target_mean':torch.tensor(target_mean),'target_scale':torch.tensor(target_scale)},
                       args.out/f'{name}_seed{seed}.pt')
            np.savez_compressed(args.out/f'{name}_seed{seed}_predictions.npz',truth=ty,prediction=pred,
                truth_FxFyMxMy=tt,prediction_FxFyMxMy=reconstruction,
                trial_ids=np.asarray([m[0] for m in tmeta]))
            prediction_bank[(name,seed)]={r['id']:r['accuracy'] for r in trials}
            (args.out/'runs.json').write_text(json.dumps(all_runs,indent=2),encoding='utf-8')
            print('E2_DONE',name,seed,'test',overall['condition_macro_accuracy'],'train',train_scores['condition_macro_accuracy'],
                  'wall_s',run['training_wall_s'],flush=True)
        # Benchmark after all seeds of this configuration. Full window latency
        # is separate from streaming one-frame GRU latency (same trained model).
        bench=tx[:1].clone(); cold=torch.zeros(2,1,hidden)
        whole=[]; stream=[]; encoder=[]
        with torch.inference_mode():
            for i in range(220):
                begin=time.perf_counter_ns(); model(bench); elapsed=(time.perf_counter_ns()-begin)/1e6
                if i>=20: whole.append(elapsed)
                if not kind.startswith('static'):
                    last=bench[:,-1:]; begin=time.perf_counter_ns()
                    if kind=='raw': z=model.encoder(last[:,0]).unsqueeze(1)
                    else: z=last
                    middle=time.perf_counter_ns(); v,_=model.gru(z,cold); model.head(v[:,-1]); end=time.perf_counter_ns()
                    if i>=20:
                        encoder.append((middle-begin)/1e6); stream.append((end-begin)/1e6)
        latency[name]={'full_window_p50_ms':float(np.quantile(whole,.5)),
            'full_window_p95_ms':float(np.quantile(whole,.95)),
            'streaming_one_frame_p95_ms':float(np.quantile(stream,.95)) if stream else None,
            'encoder_only_p95_ms':float(np.quantile(encoder,.95)) if encoder else None,
            'boundary':'CPU normalized-input software latency; excludes raw descriptor extraction, sensor, actuator and energy.'}
    summaries={}
    for name,kind,history,fraction in configs:
        selected=[r for r in all_runs if r['name']==name]
        summaries[name]={'parameters':selected[0]['parameters'],'hidden':selected[0]['hidden'],
            'history':history,'fraction':fraction,'seed_accuracy':[r['overall']['condition_macro_accuracy'] for r in selected],
            'mean_accuracy':float(np.mean([r['overall']['condition_macro_accuracy'] for r in selected])),
            'mean_rmse_FxFyMxMy':np.sqrt(np.mean([[v*v for v in r['overall']['rmse_FxFyMxMy']] for r in selected],axis=0)).tolist(),
            'groups':{g:float(np.mean([r['groups'][g]['accuracy'] for r in selected])) for g in selected[0]['groups']}}
    interaction=None
    if all(k in summaries for k in ('scalar_t1','scalar_t16','mcf_t1','mcf_t16')):
        ids=[r['id'] for r in tests]
        values=np.array([np.mean([(prediction_bank[('mcf_t16',seed)][tid]-prediction_bank[('scalar_t16',seed)][tid])-
                         (prediction_bank[('mcf_t1',seed)][tid]-prediction_bank[('scalar_t1',seed)][tid]) for seed in SEEDS]) for tid in ids])
        rng=np.random.default_rng(20261013); boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
        interaction={'mean_accuracy_interaction':float(values.mean()),'condition_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
                     'condition_ids':ids,'condition_values':values.tolist()}
    results={'status':'completed','runs':len(all_runs),'summaries':summaries,'latency':latency,'interaction':interaction,
             'limits':'Same simulated object/session; tests hold out conditions; no real E1, GRU optimality, minimal sufficiency or closed-loop evidence.'}
    (args.out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    lines=['# 仿真离线 E2 开发结果','',f'完成 {len(all_runs)} 次固定预算训练，seeds=7/11/19；8 训练条件、36 个未见测试条件，无精确重复泄漏。',
        '', '目标为外部已施加激励的当前 Fx/Fy/Mx/My，不是完整接触 wrench。真机 E1 尚未完成，因此本表不视为正式 E2 科学通过。',
        '', '| 配置 | 参数 | 历史 | 训练帧比例 | 三 seed 平均方向准确率 | 单帧流式 CPU P95 (ms) |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in summaries.items():
        l=latency[name]['streaming_one_frame_p95_ms']; l=latency[name]['full_window_p95_ms'] if l is None else l
        lines.append(f"| {name} | {r['parameters']} | {r['history']} | {r['fraction']:.0%} | {r['mean_accuracy']:.1%} | {l:.4f} |")
    if interaction:
        lines+=['',f"预指定 representation × memory 交互：{interaction['mean_accuracy_interaction']:+.1%}；条件 bootstrap 95% 区间 {interaction['condition_bootstrap_ci95']}。三个 seed 先在同条件内平均，不当作新增独立试次。"]
    lines+=['', 'Raw CNN 编码计入总参数和流式网络入口。延迟来自归一化输入的 CPU 软件调用；Raw→Scalar/MCF 特征提取另测，不能直接称真实闭环时延或能耗。不同 GRU 隐藏宽度是参数预算匹配的结果，未根据测试分数选择。',
        '', '50% 对照是相同训练条件下的帧预算，不是独立 trial 样本复杂度。未用已留出摩擦条件训练来伪称泛化，没有用最佳 seed 代表平均表现。',
        '', '各条件、各 seed、四维误差、训练拟合、混淆矩阵、全部预测与模型文件均已保存。即使某配置表现较好，也不能推出 32D 最小充分、GRU 必需或闭环抗倾斜有效。', '']
    (args.out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('E2_COMPLETE',results['runs'],'interaction',interaction,flush=True)


if __name__=='__main__': main()
