"""LeFlexiTac exploratory offline experiment. No token, Mamba, or policy rollout.
Run: python outputs/run_offline_validation.py --prepare-only
     python outputs/run_offline_validation.py
Dependencies: numpy, pyarrow, torch, scikit-learn, matplotlib.
Data cached under work/leflexitac; reproducible results under outputs/offline_validation.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import urllib.request
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT/'work'/'leflexitac'
OUT = ROOT/'outputs'/'offline_validation'
BASE = 'https://huggingface.co/datasets/Tna001/tactile_test_tube_pyflexitac'
REV = 'ebd3c711d678aa18bfc7f6a0a1e894104c8c1ea4'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def save(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def download(entry):
    path = CACHE/Path(entry['path']).name
    expected = entry.get('lfs',{}).get('oid')
    if not path.exists():
        url = BASE+'/resolve/'+REV+'/'+entry['path']
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=25) as response:
                    content = response.read()
                path.write_bytes(content)
                break
            except OSError:
                if attempt == 2:
                    raise
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected:
        assert digest == expected, f'SHA256 mismatch: {path}'
    return path, dict(path=entry['path'],bytes=path.stat().st_size,sha256=digest,
                      published_lfs_sha256_match=digest == expected if expected else None)


def prepare():
    listing_path = CACHE/'pinned_listing.json'
    if not listing_path.exists():
        url = 'https://huggingface.co/api/datasets/Tna001/tactile_test_tube_pyflexitac/tree/'+REV+'/data/chunk-000'
        with urllib.request.urlopen(url,timeout=25) as response:
            listing_path.write_bytes(response.read())
    entries = [e for e in json.loads(listing_path.read_text(encoding='utf-8'))
               if e['type']=='file' and e['path'].endswith('.parquet')]
    entries.sort(key=lambda e:e['path'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(download, entries))
    columns = ['index','episode_index','frame_index','timestamp','action',
               'observation.state','observation.tactile.primary']
    rows = {c:[] for c in columns}
    for path,_ in files:
        table = pq.read_table(path,columns=columns)
        for c in columns:
            rows[c].extend(table[c].to_pylist())
    arrays = {c:np.asarray(v,dtype=np.float32 if c in ['action','observation.state',
              'observation.tactile.primary'] else None) for c,v in rows.items()}
    order = np.argsort(arrays['index'])
    arrays = {k:v[order] for k,v in arrays.items()}
    assert np.array_equal(arrays['index'],np.arange(75041))
    assert len(np.unique(arrays['episode_index']))==64
    x = arrays['observation.tactile.primary']
    assert x.shape==(75041,12,32) and np.isfinite(x).all() and (x>=0).all()
    for key in ['action','observation.state','timestamp']:
        assert np.isfinite(arrays[key]).all()
    same = np.diff(arrays['episode_index'])==0
    dt = np.diff(arrays['timestamp'])[same]
    assert (dt>0).all()
    p = x.sum((1,2))
    audit = dict(dataset=BASE,revision=REV,rows=len(x),episodes=64,shape=list(x.shape),
        min=float(x.min()),max=float(x.max()),zero_mass_fraction=float(np.mean(p==0)),
        interval_quantiles_s=np.quantile(dt,[0,.5,.95,1]).tolist(),
        adjacent_identical_fraction=float(np.mean(np.all(x[1:]==x[:-1],axis=(1,2))[same])),
        files=[m for _,m in files],
        limits='Stored 0..1 responses; exact acquisition preprocessing and sensor timestamps unverified.')
    np.savez_compressed(CACHE/'all_data.npz', **arrays)
    save('data_audit.json',audit)
    print('DATA_AUDIT',json.dumps({k:v for k,v in audit.items() if k!='files'}),flush=True)
    return arrays


def physical_batch(w, pref, threshold=.01):
    """8 static channels, no additional baseline subtraction of stored responses.
    Covariance uses exact positive mass. Isotropic orientation has explicit guard.
    Zero or weak mass uses filled zeros and separate validity, not measured CoP.
    """
    w = np.asarray(w,dtype=np.float64).reshape(-1,384)
    yy,xx = np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,32),indexing='ij')
    x,y = xx.ravel(),yy.ravel()
    p = w.sum(1)
    valid = p>1e-6
    denom = np.where(valid,p,1)
    cx,cy = w@x/denom,w@y/denom
    a = np.maximum(w@(x*x)/denom-cx*cx,0)
    d = np.maximum(w@(y*y)/denom-cy*cy,0)
    b = w@(x*y)/denom-cx*cy
    gap = np.hypot(a-d,2*b)
    trace = a+d
    l1,l2 = (trace+gap)/2,np.maximum((trace-gap)/2,0)
    eps = 1e-12
    q = gap/(trace+eps)
    o1,o2 = q*(a-d)/(gap+eps),q*2*b/(gap+eps)
    s = np.c_[np.log1p(p/pref),cx,cy,np.mean(w>threshold,axis=1),
              np.sqrt(l1),np.sqrt(l2),o1,o2]
    s[~valid]=0
    assert np.isfinite(s).all()
    return s.astype(np.float32),valid


def causal_ema(x,episode,alpha=.5):
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = x[i] if i==0 or episode[i]!=episode[i-1] else alpha*x[i]+(1-alpha)*result[i-1]
    return result


def construct(data,train_ids,horizon=5,stride=3,threshold=.01):
    ep = data['episode_index']
    state = data['observation.state']
    filtered = causal_ema(data['observation.tactile.primary'].reshape(-1,384),ep)
    # Common preprocessing for all tactile representations, fixed a priori.
    # Cut off small values without inventing a new baseline for normalized data.
    filtered = np.where(filtered>threshold,filtered,0).astype(np.float32)
    mass = filtered.sum(1)
    train_contact = np.isin(ep,train_ids)&(mass>1e-6)
    pref = float(np.median(mass[train_contact]))
    static,valid = physical_batch(filtered,pref,0)
    # Compute full-rate causal derivatives first; selected readout rows retain
    # current and previous 30 Hz input, even though readout training is strided.
    idx = np.arange(1,len(ep)-horizon)
    idx = idx[(ep[idx]==ep[idx-1])&(ep[idx]==ep[idx+horizon])]
    dt = data['timestamp'][idx]-data['timestamp'][idx-1]
    assert (dt>0).all()
    both = valid[idx]&valid[idx-1]
    change = static[idx][:,[0,1,2]]-static[idx-1][:,[0,1,2]]
    change[:,1:]*=both[:,None]
    phys = np.c_[static[idx],change].astype(np.float32)
    state_input = np.c_[state[idx],(state[idx]-state[idx-1])/dt[:,None]].astype(np.float32)
    common_mask = np.c_[valid[idx],valid[idx-1]].astype(np.float32)
    scalar = np.c_[static[idx,0],change[:,0]].astype(np.float32)
    raw = np.c_[filtered[idx],filtered[idx-1]].astype(np.float32)
    target = data['action'][idx+horizon]-state[idx]
    selected = data['frame_index'][idx]%stride==0
    arrays = dict(state=state_input,scalar=scalar,physics=phys,raw=raw,
                  mask=common_mask,y=target,episode=ep[idx],frame=data['frame_index'][idx],
                  contact=valid[idx],original_index=idx)
    arrays = {k:v[selected] for k,v in arrays.items()}
    assert arrays['physics'].shape[1]==11
    # Causality regression: future input mutation cannot change an earlier EMA.
    prefix = min(500,len(filtered)-1)
    altered = data['observation.tactile.primary'].reshape(-1,384)[:prefix+20].copy()
    original = altered.copy()
    altered[prefix:]=99
    assert np.array_equal(causal_ema(original,ep[:len(original)])[:prefix],
                          causal_ema(altered,ep[:len(altered)])[:prefix])
    save('descriptor_protocol.json',dict(P_ref=pref,baseline='No second subtraction',
        threshold=threshold,threshold_rule='w_i if w_i>threshold, else zero; shared by all tactile inputs',
        contact_mass_min=1e-6,ema_alpha=.5,derivatives='adjacent stored-frame differences, not rates',
        centroid_difference='masked unless both frames have positive contact mass',
        static_order=['log_load','cx','cy','area','sigma1','sigma2','ox','oy'],
        geometry='Each array axis normalized to [-1,1]; not physical metric geometry',
        horizon_frames=horizon,horizon_nominal_ms=horizon*1000/30,readout_stride=stride,
        state_input_dim=12,scalar_dim=2,physics_dim=11,raw_current_previous_dim=768,
        validity_mask_dim=2,causality_prefix_check=True,
        contact_labels='Input-derived positive-mass mask; no independent event ground truth'))
    return arrays


def scaler_fit(x):
    mean,std=x.mean(0),x.std(0)
    return mean,np.where(std>1e-5,std,1)


def standardized(x,scaler):
    return (x-scaler[0])/scaler[1]


def metrics(pred,y,episode,contact,eval_scale):
    per=[]
    for e in np.unique(episode):
        mask=episode==e
        err=pred[mask]-y[mask]
        active=contact[mask]
        per.append(dict(episode=int(e),frames=int(mask.sum()),
            nmse=float(np.mean((err/eval_scale)**2)),
            contact_nmse=float(np.mean((err[active]/eval_scale)**2)) if active.any() else None,
            mae_per_joint=np.mean(abs(err),axis=0).tolist()))
    return dict(macro_nmse=float(np.mean([r['nmse'] for r in per])),
        macro_contact_nmse=float(np.mean([r['contact_nmse'] for r in per if r['contact_nmse'] is not None])),
        macro_mae_per_joint=np.mean([r['mae_per_joint'] for r in per],axis=0).tolist(),trials=per)


class Head(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,6))
    def forward(self,x):
        return self.net(x)


class Learned(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(768,64),nn.ReLU(),nn.Linear(64,11))
        self.head=Head(25) # 12 state + 2 mask + 11 learned coordinates
    def forward(self,x):
        return self.head(torch.cat([x[:,:14],self.encoder(x[:,14:])],1))


def inputs(arrays,name):
    if name=='State':
        return arrays['state']
    feature={'Scalar':'scalar','Physics11D':'physics','Raw':'raw','Learned11D':'raw'}[name]
    return np.c_[arrays['state'],arrays['mask'],arrays[feature]]


def fit_mlp(x,y,val_x,val_y,val_episode,name,seed,epochs,device,lr=.001,weight_decay=1e-4,patience=8):
    torch.manual_seed(seed)
    xs,ys=scaler_fit(x),scaler_fit(y)
    train_x=torch.as_tensor(standardized(x,xs),dtype=torch.float32,device=device)
    train_y=torch.as_tensor(standardized(y,ys),dtype=torch.float32,device=device)
    vx=torch.as_tensor(standardized(val_x,xs),dtype=torch.float32,device=device)
    vy=torch.as_tensor(standardized(val_y,ys),dtype=torch.float32,device=device)
    _,inverse,counts=np.unique(val_episode,return_inverse=True,return_counts=True)
    weights=torch.as_tensor(1/counts[inverse]/len(counts),dtype=torch.float32,device=device)
    model=(Learned() if name=='Learned11D' else Head(x.shape[1])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    best,best_epoch,stale,best_state=float('inf'),0,0,None
    history=[]
    begin=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(x),device=device)
        for indices in order.split(1024):
            pred=model(train_x[indices]);loss=torch.mean((pred-train_y[indices])**2)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            vl=float(torch.sum(torch.mean((model(vx)-vy)**2,dim=1)*weights).item())
        history.append(vl)
        if vl<best-1e-6:
            best,best_epoch,stale=vl,epoch+1,0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if stale>=patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model,xs,ys,dict(best_epoch=best_epoch,epochs_run=len(history),
        validation_standardized_mse=best,validation_curve=history,
        parameter_count=sum(p.numel() for p in model.parameters()),
        training_wall_s=time.perf_counter()-begin)


def paired_summary(results):
    full=[r for r in results if r.get('fraction')==1 and r['family']=='MLP']
    ref=[r for r in full if r['representation']=='State']
    ref_trial={e:float(np.mean([next(t['nmse'] for t in r['test']['trials'] if t['episode']==e)
                              for r in ref])) for e in [t['episode'] for t in ref[0]['test']['trials']]}
    rng=np.random.default_rng(909)
    summaries=[]
    for name in ['State','Scalar','Physics11D','Raw','Learned11D']:
        rows=[r for r in full if r['representation']==name]
        trial=np.array([np.mean([next(t['nmse'] for t in r['test']['trials'] if t['episode']==e)
                                for r in rows]) for e in ref_trial])
        baseline=np.array(list(ref_trial.values()))
        delta=trial-baseline
        boot=np.mean(delta[rng.integers(0,len(delta),(5000,len(delta)))],axis=1)
        summaries.append(dict(representation=name,mean_macro_nmse=float(trial.mean()),
            sd_across_training_seeds=float(np.std([r['test']['macro_nmse'] for r in rows])),
            relative_change_vs_state_percent=float((trial.mean()/baseline.mean()-1)*100),
            paired_delta_nmse_CI95=np.quantile(boot,[.025,.975]).tolist(),
            contact_nmse=float(np.mean([r['test']['macro_contact_nmse'] for r in rows]))))
    return summaries


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name in ['State','Scalar','Physics11D','Raw','Learned11D']:
        points=[]
        for fraction in [.25,.5,1]:
            rows=[r for r in results if r['family']=='MLP' and r['representation']==name and r['fraction']==fraction]
            if rows:
                points.append((rows[0]['train_episode_count'],np.mean([r['test']['macro_nmse'] for r in rows]),
                               np.std([r['test']['macro_nmse'] for r in rows])))
        ax.errorbar([p[0] for p in points],[p[1] for p in points],yerr=[p[2] for p in points],
                    marker='o',capsize=3,label=name)
    ax.set(xlabel='Training episodes',ylabel='Test macro normalized MSE (lower is better)',
           title='LeFlexiTac: action command 5 frames ahead, conditioned on proprioception')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(OUT/'data_efficiency.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--epochs',type=int,default=80)
    args=parser.parse_args()
    data=prepare()
    ids=np.random.default_rng(20261006).permutation(np.arange(64))
    train_ids,val_ids,test_ids=ids[:44],ids[44:54],ids[54:]
    save('split.json',dict(seed=20261006,train=train_ids.tolist(),validation=val_ids.tolist(),test=test_ids.tolist(),
         note='Fixed whole-episode split; not object/material OOD. Nested training subsets use train order.'))
    arrays=construct(data,train_ids)
    save('protocol.json',dict(date='2026-10-06',dataset=BASE,revision=REV,epochs=args.epochs,
        fractions=[.25,.5,1],training_seeds=[7,19,31],lr=.001,weight_decay=1e-4,
        batch_size=1024,early_stopping_patience=8,hidden_layers=[64,64],
        target='action[t+5] - observation.state[t], six position commands; no torque or force label',
        common_input='state[t] and backward state slope; tactile models also get current/previous contact-valid masks',
        raw_input='current and previous causal-EMA tactile fields',
        learned='Supervised 768->64->11 encoder, same 64->64 readout; not PCA and not a fixed pretrained encoder',
        normalization='Input and target scaling fit only chosen training episodes; full-train target std used only for common evaluation units',
        metrics='Whole-episode macro normalized MSE; per-joint MAE; contact-conditioned secondary score',
        limitations=['No images, token baseline, Mamba, independent event ground truth, object OOD or real hardware latency.',
                     'Fixed exploratory hyperparameters; matched readout widths, unequal total parameter counts.',
                     'Frames are correlated; bootstrap clusters over ten test episodes, averaged across three seeds.',
                     'Future action targets are labels only; all model inputs are causal.',
                     'Readout rows every third frame; derivatives and future targets computed on stored 30 Hz timeline.']))
    if args.prepare_only:
        print('PREPARATION_COMPLETE',flush=True);return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    save('environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        numpy=np.__version__,torch=torch.__version__,device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    ep=arrays['episode'];y=arrays['y'];eval_scale=scaler_fit(y[np.isin(ep,train_ids)])[1]
    vm,tm=np.isin(ep,val_ids),np.isin(ep,test_ids)
    results=[]
    results.append(dict(family='Persistence',representation='CurrentPosition',fraction=1,
        test=metrics(np.zeros_like(y[tm]),y[tm],ep[tm],arrays['contact'][tm],eval_scale)))
    # Cheap linear diagnostic, hyperparameter selected only on validation trials.
    for name in ['State','Scalar','Physics11D','Raw']:
        x=inputs(arrays,name);tr=np.isin(ep,train_ids);xs=scaler_fit(x[tr]);ys=scaler_fit(y[tr])
        best=None
        for alpha in [.1,1,10,100]:
            model=Ridge(alpha=alpha).fit(standardized(x[tr],xs),standardized(y[tr],ys))
            pred=model.predict(standardized(x[vm],xs))*ys[1]+ys[0]
            score=metrics(pred,y[vm],ep[vm],arrays['contact'][vm],eval_scale)['macro_nmse']
            if best is None or score<best[0]:best=(score,alpha,model)
        pred=best[2].predict(standardized(x[tm],xs))*ys[1]+ys[0]
        results.append(dict(family='Ridge',representation=name,fraction=1,alpha=best[1],
            validation_nmse=best[0],test=metrics(pred,y[tm],ep[tm],arrays['contact'][tm],eval_scale)))
    for fraction in [.25,.5,1]:
        subset=train_ids[:round(44*fraction)];tr=np.isin(ep,subset)
        for seed in [7,19,31]:
            for name in ['State','Scalar','Physics11D','Raw','Learned11D']:
                x=inputs(arrays,name)
                model,xs,ys,fit=fit_mlp(x[tr],y[tr],x[vm],y[vm],ep[vm],name,seed,args.epochs,device)
                with torch.no_grad():
                    pred=model(torch.as_tensor(standardized(x[tm],xs),dtype=torch.float32,device=device)).cpu().numpy()*ys[1]+ys[0]
                score=metrics(pred,y[tm],ep[tm],arrays['contact'][tm],eval_scale)
                result=dict(family='MLP',representation=name,fraction=fraction,seed=seed,
                    train_episode_count=len(subset),train_rows=int(tr.sum()),fit=fit,test=score)
                results.append(result);save('results.json',results)
                if fraction==1:
                    torch.save(dict(state_dict=model.cpu().state_dict(),x_mean=xs[0],x_std=xs[1],
                        y_mean=ys[0],y_std=ys[1]),OUT/f'{name}_seed{seed}.pt')
                print(f'FIT fraction={fraction} seed={seed} model={name} test_nmse={score["macro_nmse"]:.6f} epoch={fit["best_epoch"]}',flush=True)
    summary=paired_summary(results);save('summary.json',summary);plot(results)
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
