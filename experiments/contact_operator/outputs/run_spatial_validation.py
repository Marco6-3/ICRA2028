"""Reproducible source-held-out spatial diagnosis on pinned TaF records.
Run python outputs/run_spatial_validation.py --prepare-only; then without flag.
No robot policies; ATI is a label, never an input except explicit oracle.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, time
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from screen_spatial_contact_data import CACHE, BASE, REV, fetch

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'/'spatial_validation_expanded'
OUT.mkdir(exist_ok=True)
SPLITS={'train':list(range(7,23)), 'validation':list(range(23,27)), 'test':list(range(27,55))}
CONFIGS=[{'lr':lr,'wd':wd} for lr in [.001,.0003] for wd in [.0001,.01]]
NAMES=['Scalar','ScalarCentroid','Physics11D','Raw','Learned11D']
SEEDS=[7,19,31]

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def protocol():
    p=dict(revision=REV,split=SPLITS,
      amendment='Original obj27..30 all failed unchanged prefix calibration; no models trained. Extend fixed test candidate range once to27..54 before downloading31..54. Original audit/protocol retained under spatial_validation.',
      readout='Same time t and t-1, no future input; target ATI [-Ty/Fz,Tx/Fz]',
      calibration='First 30 rows must have all stored |Fz|<0.5; frozen per-taxel median, cutoff 3*1.4826*MAD; ATI subtract prefix median',
      eligibility='Rows >=31, same converted episode as previous row, pressure mass>0, |Fz|>=2, finite labels; no shear restriction for primary',
      conditional='15-row nonoverlapping windows, P and |Fz| CV<=.05, independent label span>=.002, all shear ratio<=.2',
      splits_are='Complete source-sequence splits, physical object identity not verified; exclusions do not move IDs between splits',
      training='One row per 3 source rows for optimization; validation tuning also stride3; final test all eligible rows',
      linear='Ridge alpha [.01,.1,1,10,100,1000] selected on equal-source validation MSE; fractions .25,.5,1; seeds7,19,31',
      neural=dict(configs=CONFIGS,seeds=SEEDS,max_epochs=200,patience=20,batch_size=4096,head=[64,64,2],
        learned_encoder=[288,64,11],selection='Equal-source validation MSE, seed7 tuning; final3 seeds, no test-based selection'),
      scalar='log1p(P/P_ref), backward difference; centroid variant also cx,cy and differences',
      physics='8 static moments + 3 backward differences, anisotropy weighted double-angle orientation',
      raw='Current and previous 144 positive responses, no EMA for any representation',
      normalization='Training-subset-only input/output mean,std and P_ref; each group weighted equally in loss',
      sensitivity='Linear frozen-primary alpha, recalculated features for cutoff0 and5*MAD; no retuning on test',
      measurements='Macro-source MAE and RMSE in stored moment/force ratio units; per-source paired errors; bootstrap source clusters',
      inference_limits='Prediction of external ratio, not calibrated CoP, slip, tipping, recovery or causal necessity; conditional window eligibility uses ATI and is retrospective diagnosis')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==p,'Protocol changed'
    else:save('protocol.json',p)

def load_record(g):
    directory=f'taf_dataset/gs_mini/gs_mini_obj{g}/data/chunk-000'
    listing=CACHE/f'obj{g}_listing.json'
    if not listing.exists():listing.write_bytes(fetch('https://huggingface.co/api/datasets/jiamig/taf-dataset/tree/'+REV+'/'+directory))
    entries=[e for e in json.loads(listing.read_text()) if e['type']=='file' and e['path'].endswith('.parquet')]
    assert len(entries)==1,'Multiple shards need explicit chronological concatenation'
    e=entries[0];path=CACHE/f'obj{g}.parquet'
    if not path.exists():path.write_bytes(fetch(BASE+'/resolve/'+REV+'/'+e['path']))
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==e['lfs']['oid']
    t=pq.read_table(path)
    raw=np.asarray(t['observation.pressure_matrix'].to_pylist(),dtype=np.float32)
    ft=np.asarray(t['observation.force_torque'].to_pylist(),dtype=np.float64)
    ep=np.asarray(t['episode_index']);ts=np.asarray(t['timestamp'])
    assert raw.shape[1:]==(12,12) and ft.shape==(len(raw),6)
    assert np.isfinite(raw).all() and np.isfinite(ft).all() and (raw>=0).all()
    good=bool(np.max(abs(ft[:30,2]))<.5)
    meta=dict(group=g,rows=len(raw),sha256=digest,calibration_ok=good,prefix_max_abs_Fz=float(np.max(abs(ft[:30,2]))))
    if not good:return None,meta
    baseline=np.median(raw[:30],axis=0);mad=np.median(abs(raw[:30]-baseline),axis=0)
    ft-=np.median(ft[:30],axis=0)
    delta=raw-baseline
    current=np.where(delta>3*1.4826*mad,delta,0)
    p=current.sum((1,2));fz=ft[:,2];normal=abs(fz)
    with np.errstate(divide='ignore',invalid='ignore'):
        y=np.c_[-ft[:,4]/fz,ft[:,3]/fz];shear=np.linalg.norm(ft[:,:2],axis=1)/normal
    conditional=np.zeros(len(raw),bool)
    window_count=0
    for e in np.unique(ep):
        ix=np.flatnonzero(ep==e);ix=ix[ix>=30]
        for start in range(0,len(ix)-14,15):
            q=ix[start:start+15]
            if normal[q].min()<2 or p[q].min()<=0:continue
            if p[q].std()/p[q].mean()>.05 or normal[q].std()/normal[q].mean()>.05:continue
            if np.linalg.norm(np.ptp(y[q],axis=0))<.002 or shear[q].max()>.2:continue
            conditional[q]=True;window_count+=1
    ix=np.arange(31,len(raw));ix=ix[(ep[ix]==ep[ix-1])&(normal[ix]>=2)&(p[ix]>0)&np.isfinite(y[ix]).all(1)]
    assert ((ts[ix]-ts[ix-1])>0).all()
    meta.update(eligible_rows=len(ix),conditional_windows=window_count,conditional_rows=int(conditional[ix].sum()),
       dt_quantiles=np.quantile(ts[ix]-ts[ix-1],[0,.5,1]).tolist() if len(ix) else [])
    # Keep baseline-subtracted arrays before cutoff for prespecified sensitivity.
    a=dict(delta=delta[ix].reshape(-1,144),previous_delta=delta[ix-1].reshape(-1,144),
      noise=np.broadcast_to((1.4826*mad).reshape(1,144),(len(ix),144)).copy(),
      y=y[ix].astype(np.float32),fz=normal[ix].astype(np.float32),group=np.full(len(ix),g),row=ix,
      conditional=conditional[ix],dt=(ts[ix]-ts[ix-1]).astype(np.float32))
    print('SOURCE',g,meta,flush=True)
    return a,meta

def prepare():
    protocol()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(load_record,range(7,55)))
    save('data_audit.json',[m for _,m in records])
    good=[a for a,_ in records if a is not None and len(a['y'])]
    for name,ids in SPLITS.items():assert any(g in ids for a in good for g in np.unique(a['group'])),f'Empty {name}'
    data={k:np.concatenate([a[k] for a in good]) for k in good[0]}
    np.savez_compressed(CACHE/'stage1_data_expanded.npz',**data)
    save('eligible_split.json',{s:[int(g) for g in np.unique(data['group']) if g in ids] for s,ids in SPLITS.items()})
    print('PREPARED',len(data['y']),flush=True)
    return data

def stat(w,pref):
    w=w.astype(np.float64);p=w.sum(1);valid=p>1e-8;den=np.where(valid,p,1)
    xx,yy=np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,12));x=xx.ravel();y=yy.ravel()
    cx=w@x/den;cy=w@y/den
    a=np.maximum(w@(x*x)/den-cx*cx,0);d=np.maximum(w@(y*y)/den-cy*cy,0);b=w@(x*y)/den-cx*cy
    gap=np.hypot(a-d,2*b);trace=a+d
    # Weighted double angle stays continuous as covariance becomes isotropic.
    s=np.c_[np.log1p(p/pref),cx,cy,(w>0).mean(1),np.sqrt((trace+gap)/2),
       np.sqrt(np.maximum((trace-gap)/2,0)),(a-d)/(trace+1e-12),2*b/(trace+1e-12)]
    s[~valid]=0
    assert np.isfinite(s).all()
    return s.astype(np.float32),valid

def features(data,train,cutoff=3):
    w=np.where(data['delta']>cutoff*data['noise'],data['delta'],0).astype(np.float32)
    prev=np.where(data['previous_delta']>cutoff*data['noise'],data['previous_delta'],0).astype(np.float32)
    pref=float(np.median(w[train].sum(1)))
    assert pref>0
    s,v=stat(w,pref);old,pv=stat(prev,pref);diff=s[:,:3]-old[:,:3];diff[:,1:]*=(v&pv)[:,None]
    # Primary rows have contact; common previous-contact flag appended to all.
    common=np.c_[v,pv].astype(np.float32)
    out={'Scalar':np.c_[s[:,0],diff[:,0]],'ScalarCentroid':np.c_[s[:,:3],diff],
       'Physics11D':np.c_[s,diff],'Raw':np.c_[w,prev],
       'FzOracle':np.c_[np.log1p(data['fz']),np.zeros(len(w))]}
    # Oracle current Fz only; no fabricated previous ATI observation.
    out['FzOracle']=out['FzOracle'][:,:1]
    for k in list(out):out[k]=np.c_[out[k],common].astype(np.float32)
    return out,pref

def macro(y,pred,g,conditional=None):
    if conditional is not None:y,pred,g=y[conditional],pred[conditional],g[conditional]
    per=[]
    for group in np.unique(g):
        err=pred[g==group]-y[g==group]
        per.append(dict(group=int(group),frames=len(err),mse=float(np.mean(err**2)),
            mae=float(np.mean(abs(err))),rmse=float(np.sqrt(np.mean(err**2)))))
    return dict(mse=float(np.mean([a['mse'] for a in per])) if per else None,
       mae=float(np.mean([a['mae'] for a in per])) if per else None,
       rmse=float(np.mean([a['rmse'] for a in per])) if per else None,per_source=per)

def scales(x,y,ix):
    xm=x[ix].mean(0);xs=x[ix].std(0);xs=np.where(xs>1e-6,xs,1)
    ym=y[ix].mean(0);ys=y[ix].std(0);ys=np.where(ys>1e-8,ys,1)
    return xm,xs,ym,ys

def weights(g):
    ids,cnt=np.unique(g,return_counts=True);lookup=dict(zip(ids,cnt))
    w=np.array([1/lookup[a] for a in g],dtype=np.float32);return w/w.mean()

class Net(nn.Module):
    def __init__(self,d,learned=False):
        super().__init__()
        self.learned=learned
        self.encoder=nn.Sequential(nn.Linear(d-2,64),nn.ReLU(),nn.Linear(64,11)) if learned else nn.Identity()
        self.head=nn.Sequential(nn.Linear(13 if learned else d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x):
        # Common validity flags bypass the learned 11D bottleneck, just as for physics.
        z=torch.cat([self.encoder(x[:,:-2]),x[:,-2:]],dim=1) if self.learned else x
        return self.head(z)

def train_nn(x,y,g,tr,va,cfg,seed,learned=False):
    torch.manual_seed(seed);np.random.seed(seed)
    xm,xs,ym,ys=scales(x,y,tr)
    model=Net(x.shape[1],learned).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    tx=torch.tensor((x[tr]-xm)/xs,device='cuda');ty=torch.tensor((y[tr]-ym)/ys,device='cuda')
    tw=torch.tensor(weights(g[tr]),device='cuda');vx=torch.tensor((x[va]-xm)/xs,device='cuda')
    vy=torch.tensor((y[va]-ym)/ys,device='cuda');vw=torch.tensor(weights(g[va]),device='cuda')
    # Optimize and select equal-source standardized MSE; same two output scales for every model.
    best=float('inf');best_state=None;epoch_best=0
    for epoch in range(200):
        model.train();order=torch.randperm(len(tx),device='cuda')
        for ix in order.split(4096):
            opt.zero_grad(set_to_none=True);loss=(((model(tx[ix])-ty[ix])**2).mean(1)*tw[ix]).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():vl=float((((model(vx)-vy)**2).mean(1)*vw).mean())
        if vl<best-1e-7:best=vl;epoch_best=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch+1-epoch_best>=20:break
    model.load_state_dict(best_state);model.eval()
    def predict(ix):
        pieces=[]
        with torch.no_grad():
            for batch in np.array_split(ix,max(1,int(np.ceil(len(ix)/8192)))):
                pieces.append(model(torch.tensor((x[batch]-xm)/xs,device='cuda')).cpu().numpy()*ys+ym)
        return np.concatenate(pieces)
    return model,predict,dict(validation_standardized_mse=best,best_epoch=epoch_best,last_epoch=epoch+1,
       parameters=sum(p.numel() for p in model.parameters())),dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)

def run(data):
    threadpool_limits(limits=4);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    g=data['group'];y=data['y']
    tr=np.flatnonzero(np.isin(g,SPLITS['train'])&(data['row']%3==0))
    va=np.flatnonzero(np.isin(g,SPLITS['validation'])&(data['row']%3==0))
    te=np.flatnonzero(np.isin(g,SPLITS['test']))
    assert not set(g[tr])&set(g[te]) and not set(g[va])&set(g[te])
    feat,pref=features(data,tr)
    save('feature_checks.json',dict(p_ref=pref,dimensions={k:v.shape[1] for k,v in feat.items()},
      train_frames=len(tr),validation_tuning_frames=len(va),test_frames=len(te),all_finite=all(np.isfinite(x).all() for x in feat.values()),
      torch=torch.__version__,gpu=torch.cuda.get_device_name(0),test_conditional_frames=int(data['conditional'][te].sum())))
    results=[];linear_choices={}
    # Linear hyperparameters selected before any test score is saved.
    for name,x in feat.items():
        xm,xs,ym,ys=scales(x,y,tr);z=((x-xm)/xs).astype(np.float64);target=((y-ym)/ys).astype(np.float64)
        choices=[]
        for alpha in [.01,.1,1,10,100,1000]:
            m=Ridge(alpha=alpha).fit(z[tr],target[tr],sample_weight=weights(g[tr]))
            pred=m.predict(z[va])*ys+ym;choices.append((macro(y[va],pred,g[va])['mse'],alpha))
        linear_choices[name]=min(choices)[1]
    save('linear_selected_alpha.json',linear_choices)
    # Linear efficiency curves: refit reference, input and label scaling per selected source subset.
    for fraction in [.25,.5,1.]:
      for seed in SEEDS:
        ids=np.random.default_rng(seed).permutation(np.unique(g[tr]));ids=ids[:max(1,int(np.ceil(len(ids)*fraction)))]
        ix=tr[np.isin(g[tr],ids)];subset_features,_=features(data,ix)
        for name,x in subset_features.items():
            xm,xs,ym,ys=scales(x,y,ix);z=((x-xm)/xs).astype(np.float64)
            model=Ridge(alpha=linear_choices[name]).fit(z[ix],((y[ix]-ym)/ys).astype(np.float64),sample_weight=weights(g[ix]))
            pred=model.predict(z[te])*ys+ym
            entry=dict(family='linear',name=name,fraction=fraction,seed=seed,train_groups=ids.tolist(),
              alpha=linear_choices[name],test=macro(y[te],pred,g[te]),conditional=macro(y[te],pred,g[te],data['conditional'][te]))
            results.append(entry)
            if fraction==1 and seed==7:
                np.savez_compressed(OUT/f'linear_{name}_prediction.npz',prediction=pred,target=y[te],group=g[te],row=data['row'][te],conditional=data['conditional'][te])
                np.savez_compressed(OUT/f'linear_{name}_weights.npz',coef=model.coef_,intercept=model.intercept_,x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)
        save('linear_results.json',results)
        print('LINEAR',fraction,seed,flush=True)
    # Sensitivity at fixed membership, train groups, and primary chosen regularization.
    sensitivity=[]
    for cutoff in [0,5]:
      f,_=features(data,tr,cutoff)
      for name,x in f.items():
        xm,xs,ym,ys=scales(x,y,tr);z=((x-xm)/xs).astype(np.float64)
        m=Ridge(alpha=linear_choices[name]).fit(z[tr],((y[tr]-ym)/ys).astype(np.float64),sample_weight=weights(g[tr]))
        p=m.predict(z[te])*ys+ym
        sensitivity.append(dict(cutoff=cutoff,name=name,test=macro(y[te],p,g[te]),conditional=macro(y[te],p,g[te],data['conditional'][te])))
    save('sensitivity.json',sensitivity)
    # Neural tuning uses validation only. Test prediction is deferred until all choices locked.
    selected={};tuning=json.loads((OUT/'neural_tuning.json').read_text(encoding='utf-8')) if (OUT/'neural_tuning.json').exists() else []
    for name in NAMES:
      x=feat['Raw'] if name=='Learned11D' else feat[name]
      for cfg in CONFIGS:
        if any(r['name']==name and r['config']==cfg for r in tuning):continue
        model,predict,meta,scale=train_nn(x,y,g,tr,va,cfg,7,name=='Learned11D')
        entry=dict(name=name,config=cfg,**meta);tuning.append(entry)
        print('TUNE',entry,flush=True);save('neural_tuning.json',tuning)
      selected[name]=min([r for r in tuning if r['name']==name],key=lambda a:a['validation_standardized_mse'])['config']
    save('neural_selected_config.json',selected)
    for name in NAMES:
      x=feat['Raw'] if name=='Learned11D' else feat[name]
      for seed in SEEDS:
        model,predict,meta,scale=train_nn(x,y,g,tr,va,selected[name],seed,name=='Learned11D')
        pred=predict(te)
        entry=dict(family='neural',name=name,seed=seed,fraction=1.,config=selected[name],**meta,
          test=macro(y[te],pred,g[te]),conditional=macro(y[te],pred,g[te],data['conditional'][te]))
        results.append(entry);save('all_results.json',results)
        torch.save(dict(state=model.cpu().state_dict(),scale=scale,meta=meta,name=name,input_dim=x.shape[1]),OUT/f'neural_{name}_seed{seed}.pt')
        np.savez_compressed(OUT/f'neural_{name}_seed{seed}_prediction.npz',prediction=pred,target=y[te],group=g[te],row=data['row'][te],conditional=data['conditional'][te])
        print('RESULT',name,seed,entry['test']['rmse'],entry['conditional']['rmse'],flush=True)
    save('all_results.json',results)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    protocol();path=CACHE/'stage1_data_expanded.npz'
    data=dict(np.load(path)) if path.exists() else prepare()
    if not args.prepare_only:run(data)
