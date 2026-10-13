"""Paired normalization development check on unique suspended conditions."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from e2_dev import Probe,matched_hidden,representation,evaluate
from temporal_probe import target_and_mask
import bounded_scaler as scaling


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(exist_ok=False,parents=True); torch.set_num_threads(2)
    manifest=json.loads((a.source/'manifest.json').read_text()); assert manifest['status']=='completed'
    reference=np.array(json.loads((a.source/'provenance.json').read_text())['reference_rotation_world_from_load'])
    records=[]
    for tid in manifest['completed_trials']:
        meta=json.loads((a.source/f'{tid}.json').read_text())
        if meta['trial']['repeat']!=0 or meta['trial']['direction']=='zero': continue
        path=a.source/f'{tid}.npz'; assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['npz_sha256']
        with np.load(path) as z: d={k:z[k] for k in z.files}
        labels,mask,target=target_and_mask(d,reference)
        records.append(dict(id=tid,data=d,labels=labels,mask=mask,target=target.astype(np.float32),group='test' if meta['trial']['preload_delta_rad'] else 'train'))
    train=[r for r in records if r['group']=='train']; test=[r for r in records if r['group']=='test']
    protocol={'train':[r['id'] for r in train],'test':[r['id'] for r in test],'seeds':[7,11,19],
        'epochs':80,'split':'Nominal preload training; unseen +0.001 rad looser preload test; identical repeats excluded',
        'methods':['legacy_zscore','robust_floor_clip8'],'profiles':['static_mcf','mcf','scalar','raw'],
        'scale_floors_mcf_per_finger':scaling.floors('mcf')[:8].tolist(),'input_clip':[-8,8],
        'target_scale':[.2,.2,.003,.003],'output_clipping':False}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[]
    for kind in ('static_mcf','mcf','scalar','raw'):
        h,_=matched_hidden(kind); history=1 if kind.startswith('static') else 16
        raw={r['id']:representation(r['data'],kind) for r in records}
        fit=np.concatenate([raw[r['id']] for r in train])
        for method in ('legacy_zscore','robust_floor_clip8'):
            if method=='legacy_zscore': center=fit.mean(axis=0); scale=np.maximum(fit.std(axis=0),1e-6)
            else: center,scale=scaling.fit(fit,kind)
            norm={tid:((x-center)/scale).astype(np.float32) if method=='legacy_zscore' else scaling.transform(x,center,scale) for tid,x in raw.items()}
            def windows(rr):
                xx=[]; yy=[]; tt=[]; mm=[]
                for r in rr:
                    for i in np.flatnonzero(r['mask']):
                        xx.append(norm[r['id']][i-history+1:i+1]); yy.append(r['labels'][i]); tt.append(r['target'][i]); mm.append((r['id'],r['group']))
                return torch.tensor(np.stack(xx)),np.array(yy),np.stack(tt),mm
            x,y,t,meta=windows(train); tx,ty,tt,tmeta=windows(test); target_scale=np.array([.2,.2,.003,.003],np.float32)
            for seed in (7,11,19):
                torch.manual_seed(seed); model=Probe(kind,h); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
                gen=torch.Generator().manual_seed(seed)
                for epoch in range(80):
                    for ids in torch.randperm(len(x),generator=gen).split(64):
                        opt.zero_grad(); out=model(x[ids]); loss=torch.nn.functional.mse_loss(out[:,:4],torch.tensor(t[ids]/target_scale))+torch.nn.functional.cross_entropy(out[:,4:],torch.tensor(y[ids])); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                model.eval(); scores,groups,trials,pred,recon=evaluate(model,tx,ty,tt,tmeta,target_scale,np.zeros(4))
                row=dict(kind=kind,normalization=method,seed=seed,history=history,parameters=sum(p.numel() for p in model.parameters()),scores=scores,trials=trials,max_abs_test_input=float(np.max(np.abs(tx.numpy()))),finite_predictions=bool(np.isfinite(recon).all()))
                results.append(row)
                torch.save(dict(state_dict=model.state_dict(),kind=kind,hidden=h,history=history,center=center,scale=scale,normalization=method,target_scale=target_scale),a.out/f'{kind}_{method}_{seed}.pt')
                np.savez_compressed(a.out/f'{kind}_{method}_{seed}.npz',truth=tt,prediction=recon,direction_truth=ty,direction_prediction=pred)
                print('BOUNDED',kind,method,seed,scores,flush=True)
                (a.out/'runs.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    summary=[]
    for kind in ('static_mcf','mcf','scalar','raw'):
        for method in ('legacy_zscore','robust_floor_clip8'):
            rr=[r for r in results if r['kind']==kind and r['normalization']==method]
            summary.append(dict(kind=kind,normalization=method,accuracy=float(np.mean([r['scores']['condition_macro_accuracy'] for r in rr])),rmse=np.sqrt(np.mean([np.array(r['scores']['rmse_FxFyMxMy'])**2 for r in rr],axis=0)).tolist(),max_input=max(r['max_abs_test_input'] for r in rr)))
    (a.out/'results.json').write_text(json.dumps(dict(status='completed',runs=len(results),summary=summary),indent=2),encoding='utf-8')
    print('BOUNDED_COMPLETE',summary,flush=True)


if __name__=='__main__': main()
