"""MCF contact-active experiments. Run with --cache-root PATH --out PATH.

Task B is a threshold-proxy diagnostic unless independent event labels exist.
No experimental success is assumed. Existing source/episode splits are reused.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, copy
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 19, 31]
CONFIGS = [{'lr': 0.001, 'wd': 0.0001}, {'lr': 0.0003, 'wd': 0.01}]

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def state(w):
    n,h,k = w.shape
    x,y = np.meshgrid(np.linspace(-1,1,k), np.linspace(-1,1,h))
    f = w.sum((1,2)); den = np.where(f>0,f,1)
    return np.column_stack([f, (w*x).sum((1,2))/den, (w*y).sum((1,2))/den, (w>0).mean((1,2))]).astype('float32')

def record(w, ep, ts, group, start=1, target=None):
    s=state(w); ix=np.arange(max(1,start),len(w))
    ix=ix[(ep[ix]==ep[ix-1]) & (ts[ix]>ts[ix-1])]
    d=s[ix]-s[ix-1]
    # Undefined zero-contact CoP is filled with zero in inputs; filtering only
    # interprets CoP displacement when both frames have contact.
    both=(s[ix,0]>0)&(s[ix-1,0]>0)
    changes=np.c_[abs(d[:,0]), np.linalg.norm(d[:,1:3],axis=1)*both, abs(d[:,3])]
    a={'Scalar':np.c_[s[ix,0],d[:,0]], 'MCF':np.c_[s[ix],d],
       'Raw':np.c_[w[ix].reshape(len(ix),-1),w[ix-1].reshape(len(ix),-1)],
       'changes':changes, 'contact':(s[ix,0]>0)|(s[ix-1,0]>0),
       'group':np.full(len(ix),group), 'row':ix, 'episode':ep[ix],
       'dt':ts[ix]-ts[ix-1]}
    if target is not None: a['y']=(target[ix]-target[ix-1]).astype('float32')
    return a

def concat(records):
    return {k:np.concatenate([r[k] for r in records]) for k in records[0]}

def prepare_a(cache,out):
    split={'train':list(range(7,23)), 'validation':list(range(23,27)), 'test':list(range(27,55))}
    records=[]; audit=[]
    for g in range(7,55):
        path=cache/'taf'/f'obj{g}.parquet'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        entries=json.loads((cache/'taf'/f'obj{g}_listing.json').read_text())
        assert digest in [e.get('lfs',{}).get('oid') for e in entries]
        t=pq.read_table(path)
        w=np.asarray(t['observation.pressure_matrix'].to_pylist(),dtype='float32')
        ft=np.asarray(t['observation.force_torque'].to_pylist(),dtype='float64')
        ep=np.asarray(t['episode_index']).ravel(); ts=np.asarray(t['timestamp']).ravel()
        assert np.isfinite(w).all() and np.isfinite(ft).all()
        ok=bool(np.max(abs(ft[:30,2]))<.5)
        audit.append({'source':g,'rows':len(w),'sha256':digest,'prefix_calibration_ok':ok})
        if not ok: continue
        base=np.median(w[:30],axis=0); mad=np.median(abs(w[:30]-base),axis=0)
        delta=w-base; w=np.where(delta>3*1.4826*mad,delta,0)
        records.append(record(w,ep,ts,g,31,ft[:,3:6]))
        print('prepared TaF',g,flush=True)
    save(out/'data_audit.json',audit)
    return concat(records),split

def prepare_b(cache,out,split_json=None):
    p=cache/'leflexitac'
    listing=json.loads((p/'pinned_listing.json').read_text(encoding='utf-8'))
    audits=[]; arrays=[]
    for entry in sorted(listing,key=lambda e:e['path']):
        if not entry['path'].endswith('.parquet'):continue
        path=p/Path(entry['path']).name
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==entry['lfs']['oid']
        table=pq.read_table(path); arrays.append(table)
        audits.append({'file':path.name,'sha256':digest,'columns':table.column_names})
    t=__import__('pyarrow').concat_tables(arrays)
    order=np.argsort(np.asarray(t['index']).ravel())
    w=np.asarray(t['observation.tactile.primary'].to_pylist(),dtype='float32')[order]
    ep=np.asarray(t['episode_index']).ravel()[order];ts=np.asarray(t['timestamp']).ravel()[order]
    records=[]
    for g in np.unique(ep):
        ix=np.flatnonzero(ep==g);records.append(record(w[ix],ep[ix],ts[ix],int(g)))
    split=json.loads((split_json or ROOT/'outputs/offline_validation_tuned/split.json').read_text())
    split={k:split[k] for k in ['train','validation','test']}
    save(out/'data_audit.json',audits)
    return concat(records),split

def thresholds(data,train,q):
    return np.array([np.quantile(v[v>0],q) if (v>0).any() else 0 for v in data['changes'][train].T])

def metrics(y,p,g,binary):
    per=[]
    for group in np.unique(g):
        m=g==group;a=y[m];b=p[m]
        r={'group':int(group),'frames':int(m.sum())}
        if binary:
            label=b[:,0]>=.5; truth=a[:,0]
            r.update(f1=float(f1_score(truth,label,zero_division=0)),precision=float(precision_score(truth,label,zero_division=0)),recall=float(recall_score(truth,label,zero_division=0)),accuracy=float(np.mean(truth==label)),positive_fraction=float(truth.mean()),brier=float(np.mean((a-b)**2)),auroc=float(roc_auc_score(truth,b[:,0])) if len(np.unique(truth))==2 else None,auprc=float(average_precision_score(truth,b[:,0])) if truth.sum() else None)
        else:
            err=a-b;r.update(mse=float(np.mean(err**2)),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),xy_rmse=float(np.sqrt(np.mean(err[:,:2]**2))),axis_rmse=np.sqrt(np.mean(err**2,axis=0)).tolist())
        per.append(r)
    keys=['f1','precision','recall','accuracy','brier','auroc','auprc'] if binary else ['mse','rmse','mae','xy_rmse']
    return {'macro':{k:float(np.mean([r[k] for r in per if r[k] is not None])) if any(r[k] is not None for r in per) else None for k in keys},'per_source':per}

class DenseTokenNet(nn.Module):
    def __init__(self,d,o):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(2,8),nn.GELU())
        self.head=nn.Sequential(nn.Linear(d//2*8,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,o))
    def forward(self,x):
        # One 8D token per taxel, current/previous values; fixed spatial order.
        tokens=self.encoder(x.reshape(len(x),2,-1).transpose(1,2))
        return self.head(tokens.flatten(1))

def fit(x,y,g,tr,va,cfg,seed,binary,tokens=False):
    torch.manual_seed(seed);np.random.seed(seed)
    xm=x[tr].mean(0);xs=x[tr].std(0);xs=np.where(xs>1e-7,xs,1)
    ym=np.zeros(y.shape[1],dtype='float32') if binary else y[tr].mean(0)
    ys=np.ones(y.shape[1],dtype='float32') if binary else np.maximum(y[tr].std(0),1e-8)
    xt=torch.as_tensor((x[tr]-xm)/xs,device='cuda');xv=torch.as_tensor((x[va]-xm)/xs,device='cuda')
    yt=torch.as_tensor((y[tr]-ym)/ys,device='cuda');yv=torch.as_tensor((y[va]-ym)/ys,device='cuda')
    def weights(groups):
        ids,cnt=np.unique(groups,return_counts=True);lookup=dict(zip(ids,cnt))
        w=np.array([1/lookup[a] for a in groups],dtype='float32');return torch.as_tensor(w/w.mean(),device='cuda')
    wt=weights(g[tr]);wv=weights(g[va])
    net=(DenseTokenNet(x.shape[1],y.shape[1]) if tokens else nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,y.shape[1]))).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    best=float('inf');bad=0;best_epoch=0;history=[]
    for epoch in range(100):
        net.train();perm=torch.randperm(len(tr),device='cuda')
        for ix in perm.split(4096):
            pred=net(xt[ix]);loss=nn.functional.binary_cross_entropy_with_logits(pred,yt[ix],reduction='none').mean(1) if binary else ((pred-yt[ix])**2).mean(1)
            loss=(loss*wt[ix]).mean();opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            pred=net(xv);loss=nn.functional.binary_cross_entropy_with_logits(pred,yv,reduction='none').mean(1) if binary else ((pred-yv)**2).mean(1)
            score=float((loss*wv).mean())
        assert np.isfinite(score)
        history.append(score)
        if score<best-1e-7:best=score;best_epoch=epoch+1;best_state=copy.deepcopy(net.state_dict());bad=0
        else:bad+=1
        if bad>=15:break
    net.load_state_dict(best_state)
    checkpoint={'state_dict':{k:v.cpu() for k,v in best_state.items()},'xm':xm,'xs':xs,'ym':ym,'ys':ys,'seed':seed,'config':cfg,'epochs':epoch+1,'best_epoch':best_epoch,'validation_loss':best,'history':history,'input_dim':x.shape[1],'output_dim':y.shape[1],'binary':binary}
    checkpoint['architecture']='dense_taxel_tokens' if tokens else 'mlp'
    return net,checkpoint

def predict(net,c,x,ix):
    pred=[]
    with torch.no_grad():
        for chunk in np.array_split(ix,max(1,int(np.ceil(len(ix)/8192)))):
            p=net(torch.as_tensor((x[chunk]-c['xm'])/c['xs'],device='cuda'))
            if c['binary']:p=torch.sigmoid(p)
            pred.append(p.cpu().numpy()*c['ys']+c['ym'])
    p=np.concatenate(pred);assert np.isfinite(p).all();return p

def run(data,split,out,binary,annotations=None,tokens_only=False):
    train=np.isin(data['group'],split['train'])
    eps=thresholds(data,train & data['contact'],.75)
    active=data['contact'] & (data['changes']>eps).any(1)
    if binary and annotations is not None:
        reviewed=json.loads(annotations.read_text(encoding='utf-8'))
        eligible=np.zeros(len(active),bool); labels=np.zeros(len(active),dtype='float32')
        for a in reviewed['annotations']:
            if a['status']!='visible_release_transition':continue
            lo,hi=a['event_interval_frames'];rl,rh=[round(t*30) for t in a['reviewed_interval_seconds']]
            group=data['group']==a['episode'];row=data['row']
            positive=group&(row>=lo)&(row<=hi)
            # Discard 0.2s before and 0.5s after coarse event intervals.
            negative=group&(row>=rl)&(row<=rh)&((row<lo-6)|(row>hi+15))
            eligible|=positive|negative;labels[positive]=1
        active &= eligible
        data['y']=labels[:,None];jump=None
        save(out/'annotations.json',reviewed)
    elif binary:
        jump=thresholds(data,train & data['contact'],.95)
        data['y']=(data['changes']>jump).any(1).astype('float32')[:,None]
    else:jump=None
    indices={s:np.flatnonzero(np.isin(data['group'],ids)&active) for s,ids in split.items()}
    # Same optimization rows for all representations. Evaluate every active test frame.
    stride=1 if binary else 3
    tr=indices['train'][::stride];va=indices['validation'][::stride];te=indices['test']
    for s,ix in indices.items():assert len(ix)>0,s
    protocol={'task':'B threshold-proxy diagnostic' if binary else 'A torque delta', 'split':split,'active_epsilon_f_p_A':eps.tolist(),'threshold_selection':'75th percentile of positive changes on contact-containing training pairs only; OR; no test-based tuning','proxy_jump_threshold':jump.tolist() if jump is not None else None,'proxy_label_warning':'Label is derived from the MCF components; detects threshold-defined jumps, does not independently validate slip or instability' if binary else None,'target':'same-frame threshold event' if binary else 'ATI [Tx,Ty,Tz][t] - [Tx,Ty,Tz][t-1], stored units; contemporaneous delta estimation, not future forecast','input':'Scalar=[f,df] 2D; MCF=[f,cx,cy,A,df,dcx,dcy,dA] 8D; Dense Raw baseline=all taxels at t,t-1 flattened into MLP (not official VLA token model)','force':'sum of preprocessed normal taxel responses, a force proxy','area':'fraction of positive taxels','cop':'normalized [-1,1] coordinates; zero-filled if no contact, CoP filter requires both frames contact','calibration':'TaF existing first30 |Fz|<0.5 gate, per-taxel median and 3*MAD cutoff; LeFlexiTac uses saved responses as provided','seeds':SEEDS,'configs':CONFIGS,'head':[64,64],'max_epochs':100,'patience':15,'batch':4096,'optimization_stride':3,'normalization':'train-only input and target means/std','validation':'equal-source standardized MSE or BCE, config selected using seed7','test':'all active test rows; equal-source metrics, paired source bootstrap','test_reuse':'existing exploratory held-out splits, not a new confirmatory dataset','counts':{s:{'all_pairs':int(np.isin(data['group'],ids).sum()),'active':len(indices[s]),'groups':len(np.unique(data['group'][indices[s]]))} for s,ids in split.items()},'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    if binary and annotations is not None:
        protocol.update(task='B visually annotated release-transition diagnostic',proxy_label_warning=None,target='AI video-reviewed release-transition interval versus neighboring non-transition frames; not micro-slip or precursor ground truth',annotation_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),annotation_scope=reviewed['scope'],annotation_method=reviewed['method'],optimization_stride=stride,eligibility='User contact-active filter AND visually reviewed labeled intervals; uncertain episodes and time gaps omitted; filter coverage reported separately')
        coverage=[]
        for a in reviewed['annotations']:
            if a['status']!='visible_release_transition':continue
            lo,hi=a['event_interval_frames'];mask=(g0:=data['group'])==a['episode'];pos=mask&(data['row']>=lo)&(data['row']<=hi)
            coverage.append({'episode':a['episode'],'split':next(s for s,ids in split.items() if a['episode'] in ids),'annotated_positive_frames':int(pos.sum()),'active_positive_frames':int((pos&active).sum())})
        save(out/'annotation_filter_coverage.json',coverage)
    if tokens_only:
        protocol['input']='All current and previous raw taxels -> shared Linear(2,8)+GELU per taxel -> tokens flattened in fixed spatial order -> [64,64] readout; 144 tokens for TaF, 384 for LeFlexiTac; trained from scratch, not official pretrained VLA'
    save(out/'protocol.json',protocol)
    np.savez_compressed(out/'selected_rows.npz',group=data['group'][te],episode=data['episode'][te],row=data['row'][te],y=data['y'][te],dt=data['dt'][te],changes=data['changes'][te])
    save(out/'filter_audit.json',[{'group':int(g),'pairs':int((data['group']==g).sum()),'contact_pairs':int((data['contact']&(data['group']==g)).sum()),'active':int((active&(data['group']==g)).sum()),'force_events':int(((data['changes'][:,0]>eps[0])&(data['group']==g)).sum()),'cop_events':int(((data['changes'][:,1]>eps[1])&(data['group']==g)).sum()),'area_events':int(((data['changes'][:,2]>eps[2])&(data['group']==g)).sum())} for g in np.unique(data['group'])])
    print('PROTOCOL',protocol['counts'],'eps',eps,flush=True)
    results=[];predictions={}
    y=data['y'];g=data['group']
    ref=np.full_like(y[te],float(y[tr].mean())) if binary else np.zeros_like(y[te])
    save(out/'reference.json',metrics(y[te],ref,g[te],binary))
    for name in (['DenseTokens'] if tokens_only else ['Scalar','MCF','Raw']):
        x=data['Raw' if tokens_only else name].astype('float32');candidates=[]
        for cfg in CONFIGS:
            net,c=fit(x,y,g,tr,va,cfg,7,binary,tokens_only);candidates.append((net,c));print('tuned',name,cfg,c['validation_loss'],flush=True)
        best=min(candidates,key=lambda nc:nc[1]['validation_loss']);selected=best[1]['config']
        save(out/f'{name}_tuning.json',[{k:v for k,v in c.items() if k in ['config','validation_loss','epochs','best_epoch','history']} for _,c in candidates])
        for seed in SEEDS:
            net,c=best if seed==7 else fit(x,y,g,tr,va,selected,seed,binary,tokens_only)
            p=predict(net,c,x,te);torch.save(c,out/f'{name}_{seed}.pt');predictions[f'{name}_{seed}']=p
            score=metrics(y[te],p,g[te],binary);results.append({'name':name,'seed':seed,'config':selected,'best_epoch':c['best_epoch'],**score})
            save(out/'results.json',results);print('RESULT',name,seed,score['macro'],flush=True)
        del candidates,best,net
    np.savez_compressed(out/'predictions.npz',**predictions)
    comparisons=[];rng=np.random.default_rng(20261007)
    key='brier' if binary else 'mse'
    for other in ([] if tokens_only else ['Scalar','Raw']):
        pairs=[]
        for source in np.unique(g[te]):
            def value(name):
                return np.mean([next(r for r in a['per_source'] if r['group']==source)[key] for a in results if a['name']==name])
            pairs.append(value('MCF')-value(other))
        pairs=np.array(pairs);boot=pairs[rng.integers(0,len(pairs),(10000,len(pairs)))].mean(1)
        comparisons.append({'difference':f'MCF minus {other} {key}','mean':float(pairs.mean()),'source_bootstrap_95_CI':np.quantile(boot,[.025,.975]).tolist(),'groups':len(pairs)})
    save(out/'paired_comparisons.json',comparisons)
    print('COMPARISONS',comparisons,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--task',choices=['A','B'],required=True);p.add_argument('--annotations',type=Path);p.add_argument('--tokens-only',action='store_true');p.add_argument('--split-json',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'results.json').exists(),'Use a new output directory to preserve prior runs'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    start=time.time()
    with threadpool_limits(limits=4):
        data,split=prepare_a(a.cache_root,a.out) if a.task=='A' else prepare_b(a.cache_root,a.out,a.split_json)
        run(data,split,a.out,a.task=='B',a.annotations,a.tokens_only)
    save(a.out/'execution.json',{'elapsed_seconds':time.time()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'completed':True})

if __name__=='__main__':main()
