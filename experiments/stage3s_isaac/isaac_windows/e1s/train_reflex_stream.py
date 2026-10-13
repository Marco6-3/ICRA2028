"""Persistent-state GRU training with shared teacher / student-state labels."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from reflex_policy import ReflexGRU,MAX_DELTA
import bounded_scaler as scaling


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--dagger',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(exist_ok=False,parents=True); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    rr=[]; hashes={}
    for root in (a.source,a.dagger):
        manifest=json.loads((root/'manifest.json').read_text()); assert manifest['status']=='completed'
        for tid in manifest['completed_trials']:
            meta=json.loads((root/f'{tid}.json').read_text()); path=root/f'{tid}.npz'
            sha=hashlib.sha256(path.read_bytes()).hexdigest(); assert sha==meta['npz_sha256']; hashes[str(path)]=sha
            with np.load(path) as z: rr.append(dict(id=tid,x=z['input_mcf_before'],y=z['expert_delta_rad']/MAX_DELTA))
    train=[r for r in rr if not r['id'].startswith('val_')]; val=[r for r in rr if r['id'].startswith('val_')]
    protocol=dict(train=[r['id'] for r in train],validation=[r['id'] for r in val],epochs=240,seeds=[7,11,19],
        streaming=True,full_trial_backprop=True,hidden=64,layers=2,max_delta_rad=MAX_DELTA,source_hashes=hashes,
        model_selection='Minimum validation condition-macro MSE; fixed seeds; no final test selection',
        v1_final_data_used_for_training=False,input='Tactile only; persistent hidden state reset at each trial')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8'); runs=[]
    for kind in ('mcf','scalar'):
        feat={r['id']:(r['x'].reshape(-1,32) if kind=='mcf' else r['x'][:,:,[0,4]].reshape(-1,8)).astype(np.float32) for r in rr}
        center,scale=scaling.fit(np.concatenate([feat[r['id']] for r in train]),kind)
        def tensors(records):
            return torch.tensor(np.stack([scaling.transform(feat[r['id']],center,scale) for r in records])),torch.tensor(np.stack([r['y'] for r in records]),dtype=torch.float32)
        x,y=tensors(train); vx,vy=tensors(val)
        for seed in (7,11,19):
            begin=time.perf_counter(); torch.manual_seed(seed); model=ReflexGRU(kind); model.head.bias.data.fill_(-3.)
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001); gen=torch.Generator().manual_seed(seed)
            best=float('inf'); state=None; best_epoch=0
            for epoch in range(240):
                model.train()
                for batch in torch.randperm(len(x),generator=gen).split(8):
                    opt.zero_grad(); z,_=model.gru(x[batch]); pred=torch.sigmoid(model.head(z))[...,0]
                    loss=((pred-y[batch])**2*(1+3*y[batch])).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                if (epoch+1)%10==0:
                    model.eval()
                    with torch.inference_mode(): z,_=model.gru(vx); vp=torch.sigmoid(model.head(z))[...,0]; vmse=float(((vp-vy)**2).mean(dim=1).mean())
                    if vmse<best: best=vmse; best_epoch=epoch+1; state={k:v.clone() for k,v in model.state_dict().items()}
                    if (epoch+1)%60==0: print('STREAM_TRAIN',kind,seed,epoch+1,vmse,flush=True)
            ck=dict(state_dict=state,kind=kind,history=1,streaming=True,center=center,scale=scale,max_delta_rad=MAX_DELTA,seed=seed,best_epoch=best_epoch,validation_mse=best)
            torch.save(ck,a.out/f'{kind}_seed{seed}.pt')
            row=dict(kind=kind,seed=seed,best_epoch=best_epoch,validation_rmse_rad=float(np.sqrt(best)*MAX_DELTA),parameters=sum(p.numel() for p in model.parameters()),train_trials=len(x),train_frames=int(y.numel()),validation_trials=len(vx),wall_s=time.perf_counter()-begin)
            runs.append(row); print('STREAM_FIT',row,flush=True)
        selected=min([r for r in runs if r['kind']==kind],key=lambda r:r['validation_rmse_rad'])
        (a.out/f'{kind}_selected.pt').write_bytes((a.out/f"{kind}_seed{selected['seed']}.pt").read_bytes())
    (a.out/'results.json').write_text(json.dumps(dict(status='completed',runs=runs),indent=2),encoding='utf-8')


if __name__=='__main__': main()
