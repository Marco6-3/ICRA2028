"""Validation-only tuning after an exploratory pilot hit its epoch cap.
Keeps original pilot files. Same test split remains exploratory, not pristine.
Run: python outputs/tune_offline_validation.py
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import run_offline_validation as exp
from validate_contact_operator import descriptor

exp.OUT=exp.ROOT/'outputs'/'offline_validation_tuned'
exp.OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
device='cuda' if torch.cuda.is_available() else 'cpu'
with np.load(exp.CACHE/'all_data.npz',allow_pickle=False) as z:
    data={k:z[k] for k in z.files}
split=json.loads((exp.ROOT/'outputs'/'offline_validation'/'split.json').read_text(encoding='utf-8'))
train_ids=np.array(split['train']);val_ids=np.array(split['validation']);test_ids=np.array(split['test'])
exp.save('split.json',split)
arrays=exp.construct(data,train_ids)
ep=arrays['episode'];y=arrays['y'];vm=np.isin(ep,val_ids);tm=np.isin(ep,test_ids);tr=np.isin(ep,train_ids)
eval_scale=exp.scaler_fit(y[tr])[1]
names=['State','Scalar','Physics11D','Raw','Learned11D']
grid=[dict(lr=lr,weight_decay=wd) for lr in [.001,.0003] for wd in [.0001,.01]]
exp.save('protocol.json',dict(initial_pilot='outputs/offline_validation',tuning_reason='25/45 pilot fits had best epoch >=75 of 80',
    hyperparameter_selection='Lowest whole-validation-episode macro standardized MSE at full training budget, seed7 only; no candidate test scores',
    grid=grid,max_epochs=240,early_stopping_patience=16,
    final_seeds=[7,19,31],fractions=[.25,.5,1],test_split_status='Reused exploratory pilot test; not independent confirmatory evidence',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    primary='action[t+5]-state[t]; no images, independent contact labels, tokens or temporal memory model',
    parameter_budget='Matched head widths, unequal parameter counts; see each fit',
    limitations='One task, one split; no OOD. Threshold .01 and EMA .5 fixed without test selection.'))

# Validate vectorized moments against the scalar mathematical implementation.
rng=np.random.default_rng(301)
w=rng.uniform(0,1,(32,384));xx,yy=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,12))
xy=np.c_[xx.ravel(),yy.ravel()]
batch,_=exp.physical_batch(w,10,0)
reference=np.array([descriptor(row,xy,pref=10) for row in w])
error=float(np.max(abs(batch-reference)))
assert error<1e-6
zero,_=exp.physical_batch(np.zeros((1,384)),10,0)
assert np.array_equal(zero,np.zeros((1,8)))
exp.save('implementation_checks.json',dict(vectorized_vs_scalar_max_error=error,
    zero_contact_finite=True,causal_prefix_check=True,
    full_physics_shape=list(arrays['physics'].shape),all_physics_finite=bool(np.isfinite(arrays['physics']).all())))

candidates=[];chosen={}
for name in names:
    x=exp.inputs(arrays,name)
    for config in grid:
        model,xs,ys,fit=exp.fit_mlp(x[tr],y[tr],x[vm],y[vm],ep[vm],name,7,240,device,patience=16,**config)
        item=dict(representation=name,config=config,fit=fit)
        candidates.append(item)
        if name not in chosen or fit['validation_standardized_mse']<chosen[name]['score']:
            chosen[name]=dict(config=config,score=fit['validation_standardized_mse'])
        exp.save('validation_tuning.json',candidates)
        print('TUNE',name,config,'val',round(fit['validation_standardized_mse'],5),'epoch',fit['best_epoch'],flush=True)
exp.save('selected_hyperparameters.json',chosen)

results=[]
results.append(dict(family='Persistence',representation='CurrentPosition',fraction=1,
    test=exp.metrics(np.zeros_like(y[tm]),y[tm],ep[tm],arrays['contact'][tm],eval_scale)))
pilot=json.loads((exp.ROOT/'outputs'/'offline_validation'/'results.json').read_text(encoding='utf-8'))
results.extend([r for r in pilot if r['family']=='Ridge'])
for fraction in [.25,.5,1]:
    subset=train_ids[:round(44*fraction)];train_mask=np.isin(ep,subset)
    for seed in [7,19,31]:
        for name in names:
            x=exp.inputs(arrays,name)
            config=chosen[name]['config']
            model,xs,ys,fit=exp.fit_mlp(x[train_mask],y[train_mask],x[vm],y[vm],ep[vm],name,seed,240,device,
                                      patience=16,**config)
            with torch.no_grad():
                pred=model(torch.as_tensor(exp.standardized(x[tm],xs),dtype=torch.float32,device=device)).cpu().numpy()*ys[1]+ys[0]
            score=exp.metrics(pred,y[tm],ep[tm],arrays['contact'][tm],eval_scale)
            result=dict(family='MLP',representation=name,fraction=fraction,seed=seed,config=config,
                train_episode_count=len(subset),train_rows=int(train_mask.sum()),fit=fit,test=score)
            results.append(result);exp.save('results.json',results)
            if fraction==1:
                model.cpu()
                torch.save(dict(state_dict=model.state_dict(),x_mean=xs[0],x_std=xs[1],y_mean=ys[0],y_std=ys[1]),
                           exp.OUT/f'{name}_seed{seed}.pt')
                np.savez_compressed(exp.OUT/f'predictions_{name}_seed{seed}.npz',
                    prediction=pred,target=y[tm],episode=ep[tm],frame=arrays['frame'][tm],
                    contact=arrays['contact'][tm],eval_scale=eval_scale)
            print('FINAL',fraction,seed,name,round(score['macro_nmse'],6),'epoch',fit['best_epoch'],flush=True)

summary=exp.paired_summary(results);exp.save('summary.json',summary);exp.plot(results)

# Descriptive CPU preprocessing timing, not physical event-to-action latency.
pref=json.loads((exp.OUT/'descriptor_protocol.json').read_text())['P_ref']
previous_w=np.zeros(384,dtype=np.float32);previous_s=np.zeros(8,dtype=np.float32);previous_valid=False
samples=[]
for i in range(1200):
    frame=data['observation.tactile.primary'][i].ravel()
    start=time.perf_counter_ns()
    smooth=.5*frame+.5*previous_w
    weights=np.where(smooth>.01,smooth,0)
    static,valid=exp.physical_batch(weights,pref,0)
    change=static[0,[0,1,2]]-previous_s[[0,1,2]]
    change[1:]*=bool(valid[0]) and previous_valid
    state=np.r_[static[0],change]
    previous_w,previous_s,previous_valid=smooth,static[0],bool(valid[0])
    elapsed=(time.perf_counter_ns()-start)/1e6
    if i>=200:samples.append(elapsed)
exp.save('cpu_descriptor_timing.json',dict(warmup_frames=200,measured_frames=1000,
    p50_ms=float(np.quantile(samples,.5)),p95_ms=float(np.quantile(samples,.95)),
    scope='Python CPU EMA, threshold, 8 moments, 3 backward differences and state update; unverified acquisition input',
    excludes='sensor sampling, transport, policy model, actuator and physical event labeling'))
print('SUMMARY',json.dumps(summary),flush=True)
