"""Same privileged labels, tactile-only MCF and Scalar residual students."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from reflex_policy import ReflexGRU,MAX_DELTA
import bounded_scaler as scaling


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(exist_ok=False,parents=True); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest=json.loads((a.source/'manifest.json').read_text()); assert manifest['status']=='completed'
    rr=[]
    for tid in manifest['completed_trials']:
        meta=json.loads((a.source/f'{tid}.json').read_text()); path=a.source/f'{tid}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['npz_sha256']
        with np.load(path) as z: rr.append(dict(id=tid,x=z['input_mcf_before'],y=z['expert_delta_rad']/MAX_DELTA))
    # IDs explicitly declare validation conditions; no frame random split.
    train=[r for r in rr if not r['id'].startswith('val_')]; val=[r for r in rr if r['id'].startswith('val_')]
    assert train and val
    protocol=dict(train=[r['id'] for r in train],validation=[r['id'] for r in val],history=16,seeds=[7,11,19],epochs=160,
        hidden=64,layers=2,max_delta_rad=MAX_DELTA,model_selection='Minimum condition-macro validation MSE among three fixed seeds',
        input='32D MCF or 8D Scalar only; no time, external excitation or object state',source=str(a.source),
        source_hashes={r['id']:hashlib.sha256((a.source/(r['id']+'.npz')).read_bytes()).hexdigest() for r in rr})
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    runs=[]
    for kind in ('mcf','scalar'):
        feat={r['id']:(r['x'].reshape(-1,32) if kind=='mcf' else r['x'][:,:,[0,4]].reshape(-1,8)).astype(np.float32) for r in rr}
        center,scale=scaling.fit(np.concatenate([feat[r['id']] for r in train]),kind)
        norm={tid:scaling.transform(x,center,scale) for tid,x in feat.items()}
        def windows(records):
            xx=[]; yy=[]; ids=[]
            for r in records:
                x=norm[r['id']]; padded=np.concatenate([np.repeat(x[:1],15,axis=0),x])
                for i in range(len(x)): xx.append(padded[i:i+16]); yy.append(r['y'][i]); ids.append(r['id'])
            return torch.tensor(np.stack(xx)),torch.tensor(np.array(yy),dtype=torch.float32),np.array(ids)
        x,y,ids=windows(train); vx,vy,vids=windows(val)
        for seed in (7,11,19):
            begin=time.perf_counter(); torch.manual_seed(seed); model=ReflexGRU(kind); model.head.bias.data.fill_(-3.)
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001); gen=torch.Generator().manual_seed(seed)
            best=float('inf'); best_state=None; best_epoch=0
            for epoch in range(160):
                model.train()
                for batch in torch.randperm(len(x),generator=gen).split(128):
                    opt.zero_grad(); pred=model(x[batch]); loss=((pred-y[batch])**2*(1+3*y[batch])).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                if (epoch+1)%10==0:
                    model.eval()
                    with torch.inference_mode(): vp=torch.cat([model(b) for b in vx.split(256)])
                    vmse=float(np.mean([float(((vp[vids==tid]-vy[vids==tid])**2).mean()) for tid in np.unique(vids)]))
                    if vmse<best: best=vmse; best_epoch=epoch+1; best_state={k:v.clone() for k,v in model.state_dict().items()}
                    if (epoch+1)%40==0: print('REFLEX_TRAIN',kind,seed,epoch+1,'val_mse',vmse,flush=True)
            ck=dict(state_dict=best_state,kind=kind,history=16,center=center,scale=scale,max_delta_rad=MAX_DELTA,seed=seed,best_epoch=best_epoch,validation_mse=best)
            torch.save(ck,a.out/f'{kind}_seed{seed}.pt')
            row=dict(kind=kind,seed=seed,best_epoch=best_epoch,validation_rmse_rad=float(np.sqrt(best)*MAX_DELTA),parameters=sum(p.numel() for p in model.parameters()),train_frames=len(x),validation_frames=len(vx),wall_s=time.perf_counter()-begin)
            runs.append(row); print('REFLEX_FIT',row,flush=True)
        selected=min([r for r in runs if r['kind']==kind],key=lambda r:r['validation_rmse_rad'])
        (a.out/f'{kind}_selected.pt').write_bytes((a.out/f"{kind}_seed{selected['seed']}.pt").read_bytes())
    (a.out/'results.json').write_text(json.dumps(dict(status='completed',runs=runs),indent=2),encoding='utf-8')


if __name__=='__main__': main()
