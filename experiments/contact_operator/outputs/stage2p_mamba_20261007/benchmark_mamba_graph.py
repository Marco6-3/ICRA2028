"""Secondary deployment benchmark. No retraining or accuracy selection.

CUDA Graph replays identical eager FP32 operations at batch1 with static caches.
Runs after all training to avoid GPU workload interference.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
repo=Path(r'C:\Users\mingzhe Liu\Documents\Codex\2026-10-06\https-github-com-marco6-3-icra2028\work\ICRA2028')
sys.path.insert(0,str(repo/'experiments/contact_operator'))
import run_stage2p_mamba as exp
D=Path('outputs/stage2p_mamba');rs=json.loads((D/'results.json').read_text())
assert len(rs)==24 and json.loads((D/'execution.json').read_text())['completed']
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')

def clear(h):
    if isinstance(h,torch.Tensor):h.zero_()
    else:
        for pair in h:
            for v in pair:v.zero_()

def copy_state(dst,src):
    if isinstance(dst,torch.Tensor):dst.copy_(src)
    else:
        for d,s in zip(dst,src):
            for dv,sv in zip(d,s):dv.copy_(sv)

def measure(net,frames,arch):
    x=frames[:,:1].clone()
    h=(torch.zeros(2,1,64,device='cuda') if arch=='GRU' else [(torch.zeros(1,100,16,device='cuda'),torch.zeros(1,100,3,device='cuda')) for _ in range(2)])
    def invoke():
        # mambapy mutates its list of tuples; keep the captured static tensors stable.
        cache=list(h) if isinstance(h,list) else h
        pred,new=exp.step(net,x,cache);copy_state(h,new)
        return pred
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(20):invoke()
    torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();clear(h)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=invoke()
    clear(h);eager=None;errors=[]
    for i in range(100):
        frame=frames[:,i:i+1];pred,eager=exp.step(net,frame,eager)
        x.copy_(frame);graph.replay();errors.append(float((pred-output).abs().max()))
    assert max(errors)<1e-4
    for i in range(100):x.copy_(frames[:,i:i+1]);graph.replay()
    torch.cuda.synchronize();clear(h);times=[]
    for i in range(frames.shape[1]):
        torch.cuda.synchronize();st=time.perf_counter()
        x.copy_(frames[:,i:i+1]);graph.replay();torch.cuda.synchronize()
        times.append((time.perf_counter()-st)*1000)
    return dict(p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),max_ms=max(times),fraction_below_1ms=float(np.mean(np.array(times)<1)),parity_max_normalized_error=max(errors),latencies_ms=times,state_bytes=exp.state_bytes(h),frames=len(times))

source=int(next(r['streaming']['source'] for r in rs if 'streaming' in r))
t=pq.read_table(repo.parent/'taf'/f'obj{source}.parquet');w=np.array(t['observation.pressure_matrix'].to_pylist(),dtype='float32');ep=np.asarray(t['episode_index']).ravel()
baseline=np.median(w[:30],axis=0);mad=np.median(abs(w[:30]-baseline),axis=0);d=w-baseline;w=np.where(d>3*1.4826*mad,d,0)
s=exp.base.stat(w);ds=np.zeros_like(s);ds[1:]=s[1:]-s[:-1];ds[np.r_[True,ep[1:]!=ep[:-1]]]=0
scalar=np.zeros((len(s),8),dtype='float32');scalar[:,:2]=np.c_[s[:,0],ds[:,0]];features={'Scalar':scalar,'MCF':np.c_[s,ds]}
results=[]
with torch.no_grad():
    for r in rs:
        if 'streaming' not in r:continue
        arch=r['architecture'];cp=torch.load(D/arch/(r['key']+'.pt'),weights_only=False);cls=exp.GRUProbe if arch=='GRU' else exp.MambaProbe
        net=cls().cuda();net.load_state_dict(cp['state_dict']);net.eval();start=r['streaming']['start'];n=r['streaming']['frames'];mu,sd=cp['normalization']
        frames=torch.from_numpy((features[r['name']][start:start+n]-mu)/sd).cuda()[None,:,None,:]
        entry=dict(architecture=arch,key=r['key'],name=r['name'],T=r['T'],source=source,start=start)
        try:entry.update(measure(net,frames,arch));entry['passed']=True
        except Exception as exc:entry.update(passed=False,error=repr(exc))
        results.append(entry);print({k:v for k,v in entry.items() if k!='latencies_ms'},flush=True)
        exp.base.save(D/'cuda_graph_benchmark.json',dict(scope='Secondary same-weight CUDA Graph deployment benchmark; input GPU-to-GPU copy, replay, sync; no feature extraction/H2D; state reset before real contiguous rollout; FP32; same GPU as primary; excludes graph setup',results=results))
        del net
