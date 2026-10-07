"""Matched-parameter temporal probes on frozen TaF source splits.

Continuous causal histories, independently measured current ATI Fx,Fy,Mx,My.
No test-guided tuning. Scalar and MCF share the entire network definition.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from threadpoolctl import threadpool_limits

SPLIT={'train':list(range(7,23)),'validation':list(range(23,27)),'test':list(range(27,55))}
HISTORIES=[1,20,50];SEEDS=[7,19,31]
EPS=np.array([66.,0.024913746863603592,0.04166668653488159])

def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def stat(w):
    x,y=np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,12));f=w.sum((1,2));den=np.maximum(f,1e-8)
    return np.c_[f,(w*x).sum((1,2))/den,(w*y).sum((1,2))/den,(w>0).mean((1,2))].astype('float32')

def prepare(cache,out):
    records={};examples=[];audit=[]
    for g in range(7,55):
        path=cache/'taf'/f'obj{g}.parquet';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        listing=json.loads((cache/'taf'/f'obj{g}_listing.json').read_text());assert digest in [e.get('lfs',{}).get('oid') for e in listing]
        t=pq.read_table(path);w=np.array(t['observation.pressure_matrix'].to_pylist(),dtype='float32');ft=np.array(t['observation.force_torque'].to_pylist(),dtype='float64')
        ep=np.asarray(t['episode_index']).ravel();ts=np.asarray(t['timestamp']).ravel()
        ok=bool(np.max(abs(ft[:30,2]))<.5);audit.append({'group':g,'sha256':digest,'rows':len(w),'calibration_ok':ok})
        if not ok:continue
        ft=ft-np.median(ft[:30],axis=0)
        baseline=np.median(w[:30],axis=0);mad=np.median(abs(w[:30]-baseline),axis=0);delta=w-baseline;w=np.where(delta>3*1.4826*mad,delta,0)
        s=stat(w);ds=np.zeros_like(s);ds[1:]=s[1:]-s[:-1];same=np.r_[False,ep[1:]==ep[:-1]];ds[~same]=0
        changes=np.c_[abs(ds[:,0]),np.linalg.norm(ds[:,1:3],axis=1)*((s[:,0]>0)&np.r_[False,s[:-1,0]>0]),abs(ds[:,3])]
        active=(changes>EPS).any(1)&((s[:,0]>0)|np.r_[False,s[:-1,0]>0])
        # Keep the exact prior anchor grid for a target-only controlled rerun.
        # The end-of-episode 30-frame reserve is inherited, not future input.
        for e in np.unique(ep):
            rows=np.flatnonzero(ep==e);lo=max(30,int(rows[0]));hi=int(rows[-1]);anchors=np.arange(lo+50,hi-30+1)
            anchors=anchors[(anchors%20==0)&active[anchors]]
            assert np.all(ep[anchors-50]==ep[anchors]) and np.all(ep[anchors+30]==ep[anchors])
            assert np.all(np.diff(ts[rows])>0)
            for i in anchors:
                examples.append((g,int(i),int(e),ft[i,[0,1,3,4]],float(ts[i]-ts[i-1])))
        scalar=np.zeros((len(s),8),dtype='float32');scalar[:,:2]=np.c_[s[:,0],ds[:,0]]
        records[g]={'MCF':np.c_[s,ds],'Scalar':scalar,'raw':w.reshape(len(w),144),'ep':ep}
        print('PREPARED',g,flush=True)
    groups=np.array([e[0] for e in examples]);anchors=np.array([e[1] for e in examples]);episodes=np.array([e[2] for e in examples]);y=np.array([e[3] for e in examples],dtype='float32');dt=np.array([e[4] for e in examples])
    ids={k:np.flatnonzero(np.isin(groups,v)) for k,v in SPLIT.items()}
    assert all(len(v)>0 for v in ids.values())
    save(out/'data_audit.json',audit)
    np.savez_compressed(out/'examples.npz',group=groups,anchor=anchors,episode=episodes,y=y,dt=dt,**ids)
    norms={}
    for name in ['Scalar','MCF']:
        values=np.concatenate([r[name][31:] for g,r in records.items() if g in SPLIT['train']]);mean=values.mean(0);std=values.std(0);std=np.where(std>1e-6,std,1)
        norms[name]=(mean.astype('float32'),std.astype('float32'))
    values=np.concatenate([r['raw'][31:].ravel() for g,r in records.items() if g in SPLIT['train']]);mu=float(values.mean());sd=max(float(values.std()),1e-6)
    norms['Dense']=(np.array([mu,mu,0,0,0,0,0,0],dtype='float32'),np.array([sd,sd,1,1,1,1,1,1],dtype='float32'))
    return records,groups,anchors,y,ids,norms

class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.token=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.pool=nn.Linear(64,32)
        self.gru=nn.GRU(32,64,num_layers=2,batch_first=True)
        self.head=nn.Linear(64,4)
    def forward(self,x,h=None):
        z=self.token(x);z=torch.cat([z.mean(2),z.max(2).values],dim=-1);z=self.pool(z)
        z,h=self.gru(z,h);return self.head(z[:,-1]),h

def batch(records,groups,anchors,ix,T,name,norm):
    if name=='Dense':
        x=np.zeros((len(ix),T,144,8),dtype='float32');xx,yy=np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,12));x[:,:,:,2]=xx.ravel();x[:,:,:,3]=yy.ravel()
        for j,k in enumerate(ix):
            r=records[int(groups[k])];t=anchors[k];x[j,:,:,0]=r['raw'][t-T+1:t+1];x[j,:,:,1]=r['raw'][t-T:t]
    else:
        x=np.stack([records[int(groups[k])][name][anchors[k]-T+1:anchors[k]+1] for k in ix])[:,:,None,:]
    return torch.from_numpy((x-norm[0])/norm[1]).cuda()

def source_metrics(y,p,g):
    per=[]
    for source in np.unique(g):
        error=(p[g==source]-y[g==source]).reshape(-1,2,2)
        per.append({'group':int(source),'windows':int((g==source).sum()),'mse':np.mean(error**2,axis=(0,2)).tolist(),'rmse':np.sqrt(np.mean(error**2,axis=(0,2))).tolist(),'mae':np.mean(abs(error),axis=(0,2)).tolist()})
    return {'macro':{key:np.mean([r[key] for r in per],axis=0).tolist() for key in ['mse','rmse','mae']},'per_source':per}

def weights(groups):
    u,c=np.unique(groups,return_counts=True);d=dict(zip(u,c));w=np.array([1/d[g] for g in groups],dtype='float32');return w/w.mean()

def fit(records,g,t,y,ids,norm,name,T,seed,out):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=Probe().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    tr,va,te=[ids[k] for k in ['train','validation','test']]
    ym=y[tr].mean(0);ys=np.maximum(y[tr].std(0),1e-8);target=torch.as_tensor((y-ym)/ys,device='cuda')
    wt=weights(g[tr]);wt_lookup=np.zeros(len(y),dtype='float32');wt_lookup[tr]=wt;wt_tensor=torch.as_tensor(wt_lookup,device='cuda')
    wv=weights(g[va]);yv=(y[va]-ym)/ys
    best=float('inf');bad=0;history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(40):
        net.train();order=rng.permutation(tr)
        for ix in np.array_split(order,max(1,int(np.ceil(len(order)/64)))):
            x=batch(records,g,t,ix,T,name,norm);pred,_=net(x);loss=(((pred-target[ix])**2).mean(1)*wt_tensor[ix]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        net.eval();parts=[]
        with torch.no_grad():
            for ix in np.array_split(va,max(1,int(np.ceil(len(va)/128)))):parts.append(net(batch(records,g,t,ix,T,name,norm))[0].cpu().numpy())
        vp=np.concatenate(parts);score=float(np.mean(np.mean((vp-yv)**2,axis=1)*wv));assert np.isfinite(score);history.append(score)
        if score<best-1e-7:best=score;best_epoch=epoch+1;best_state=copy.deepcopy(net.state_dict());bad=0
        else:bad+=1
        if (epoch+1)%10==0:print('TRAIN',name,T,seed,epoch+1,score,flush=True)
        if bad>=8:break
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start;train_peak=torch.cuda.max_memory_allocated()
    net.load_state_dict(best_state);net.eval();predictions=[]
    with torch.no_grad():
        for ix in np.array_split(te,max(1,int(np.ceil(len(te)/128)))):predictions.append(net(batch(records,g,t,ix,T,name,norm))[0].cpu().numpy()*ys+ym)
    pred=np.concatenate(predictions);assert np.isfinite(pred).all()
    checkpoint={'state_dict':{k:v.cpu() for k,v in best_state.items()},'normalization':norm,'target_mean':ym,'target_std':ys,'name':name,'T':T,'seed':seed,'best_epoch':best_epoch,'validation_loss':best,'history':history,'parameters':sum(p.numel() for p in net.parameters()),'train_seconds':train_seconds,'peak_train_allocated_bytes':train_peak}
    key=f'{name}_T{T}_seed{seed}';torch.save(checkpoint,out/(key+'.pt'));np.save(out/(key+'_pred.npy'),pred)
    result={k:v for k,v in checkpoint.items() if k not in ['state_dict','normalization','target_mean','target_std','history']};result.update(source_metrics(y[te],pred,g[te]));result['key']=key
    if seed==7:
        result['benchmark']=benchmark(net,batch(records,g,t,te[:64],T,name,norm))
        x=batch(records,g,t,te[:8],T,name,norm).cpu();cpu=Probe();cpu.load_state_dict(checkpoint['state_dict']);cpu.eval()
        with torch.no_grad():p=cpu(x)[0].numpy()*ys+ym
        err=float(np.max(abs(p-pred[:8])));scaled_err=float(np.max(abs((p-pred[:8])/ys)));assert scaled_err<1e-4;result['cpu_cold_load_max_abs_error']=err;result['cpu_cold_load_max_normalized_error']=scaled_err
    print('RESULT',key,result['macro'],flush=True)
    return result

def benchmark(net,x):
    # Same batch size and history length; excludes input extraction and disk IO.
    with torch.no_grad():
        for _ in range(10):net(x)
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            start=time.perf_counter();net(x);torch.cuda.synchronize();times.append(time.perf_counter()-start)
        peak=torch.cuda.max_memory_allocated()-base
        frame=x[:1,-1:];h=None
        for _ in range(30):_,h=net(frame,h)
        torch.cuda.synchronize();st=time.perf_counter()
        for _ in range(200):_,h=net(frame,h)
        torch.cuda.synchronize();stream=(time.perf_counter()-st)/200
    return {'batch':len(x),'history_frames':x.shape[1],'tokens_per_frame':x.shape[2],'window_batch_median_ms':float(np.median(times)*1000),'window_frames_per_second':float(len(x)*x.shape[1]/np.median(times)),'incremental_peak_inference_bytes':int(peak),'stream_batch1_ms_per_frame':stream*1000,'stream_batch1_frames_per_second':1/stream,'scope':'model only; GPU synchronized wall time, 10 warmups/30 batch measurements; 30 warmups/200 state-carrying streaming frames; features already resident on GPU; not sensor-to-action latency'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--forecast-examples',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'protocol.json').exists(),'Use a fresh directory; preserve all runs'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    protocol={
        'stage1_status':'Conditional Pass; frozen unchanged',
        'task':'Now-cast contemporaneous ATI [Fx,Fy,Mx,My] at anchor t from tactile history ending at t',
        'targets':['Fx','Fy','Mx','My'],
        'metric_groups':['tangential_force_xy','lateral_moment_xy'],
        'label_calibration':'Subtract each source first30 ATI median; inherited unloaded-prefix gate; stored force/moment units, not new physical calibration',
        'histories':HISTORIES,'split':SPLIT,'seeds':SEEDS,
        'filter_epsilon_from_frozen_stage1':EPS.tolist(),
        'filtering':'Exact same anchors as prior forecast probe for target-only comparison. Whole continuous history; only t filtered. Inherited 30-frame end reserve retained; no future signal enters input or target.',
        'backbone':'identical token MLP8-32-32, mean+max pool, Linear64-32, two-layer GRU hidden64, head4',
        'representations':['Scalar','MCF'],
        'input':'Scalar=[f,df] zero padded to8; MCF8; one token per frame. No ATI, actions, proprioception or future signals in inputs.',
        'optimization':{'lr':.001,'weight_decay':.0001,'batch':64,'max_epochs':40,'patience':8,'selection':'equal-source normalized validation MSE, fixed hyperparameters, no test tuning'},
        'normalization':'Training source input statistics, training target mean/std; force and moment scored separately in original saved units',
        'hypothesis':'History20/50 may improve current shear-force/lateral-moment reconstruction and amplify MCF relative to Scalar; to be tested',
        'history_test':'Compare each representation T20/T50 against its own T1, then compare MCF versus Scalar at each T; paired source bootstrap',
        'limits':'Current ATI reconstruction is measured, not direct identification of shear strain, friction coefficient, slip or viscoelastic mechanism. Existing exploratory source split. No full dense-token attention compute claim in this run.',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'precision':'FP32; cudnn and matmul TF32 disabled after failed initial CPU/GPU parity check; all models rerun, no tolerance relaxed',
        'parameters':sum(p.numel() for p in Probe().parameters()),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    save(a.out/'protocol.json',protocol);start=time.time()
    with threadpool_limits(limits=4):
        records,g,t,y,ids,norms=prepare(a.cache_root,a.out)
        save(a.out/'sample_counts.json',{s:{'windows':len(ix),'sources':len(np.unique(g[ix]))} for s,ix in ids.items()})
        print('COUNTS',{s:len(ix) for s,ix in ids.items()},flush=True)
        save(a.out/'zero_reference.json',source_metrics(y[ids['test']],np.zeros_like(y[ids['test']]),g[ids['test']]))
        source_mean=np.mean([y[ids['train']][g[ids['train']]==source].mean(0) for source in np.unique(g[ids['train']])],axis=0)
        save(a.out/'train_mean_reference.json',{'training_source_balanced_mean':source_mean.tolist(),**source_metrics(y[ids['test']],np.broadcast_to(source_mean,y[ids['test']].shape),g[ids['test']])})
        prior=np.load(a.forecast_examples)
        assert np.array_equal(prior['group'],g) and np.array_equal(prior['anchor'],t)
        for key in ids:assert np.array_equal(prior[key],ids[key])
        save(a.out/'anchor_verification.json',{'identical_to_forecast_anchors':True,'forecast_examples_sha256':hashlib.sha256(a.forecast_examples.read_bytes()).hexdigest(),'windows':len(y)})
        results=[]
        for T in HISTORIES:
            for name in ['Scalar','MCF']:
                for seed in SEEDS:
                    results.append(fit(records,g,t,y,ids,norms[name],name,T,seed,a.out));save(a.out/'results.json',results)
    save(a.out/'execution.json',{'completed':True,'seconds':time.time()-start,'models':len(results),'script_sha256':protocol['script_sha256']})

if __name__=='__main__':main()
