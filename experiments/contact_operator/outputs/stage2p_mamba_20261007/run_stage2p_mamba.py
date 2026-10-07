"""Offline TaF current-state Mamba/GRU ablation; preserves prior nowcast runs.

Backend: mambapy 1.2.0 Mamba-1, PyTorch parallel selective scan, eager step.
No official fused CUDA scan/step kernels are used. FP32 throughout.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, hashlib, inspect, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from mambapy.mamba import Mamba, MambaConfig
import mambapy.mamba as backend
import mambapy.pscan as scan_backend
import run_stage2p_nowcast as base

GRUProbe = base.Probe

class MambaProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.token=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.pool=nn.Linear(64,32)
        self.project=nn.Linear(32,50)
        self.mamba=Mamba(MambaConfig(d_model=50,n_layers=2,d_state=16,d_conv=4,expand_factor=2,pscan=True,use_cuda=False))
        self.head=nn.Linear(50,4)

    def encode(self,x):
        z=self.token(x)
        return self.project(self.pool(torch.cat([z.mean(2),z.max(2).values],-1)))

    def forward(self,x,h=None):
        assert h is None, 'Use step for stateful inference'
        return self.head(self.mamba(self.encode(x))[:,-1]), None

    def step(self,x,h=None):
        if h is None:
            h=[(None,x.new_zeros((len(x),100,3))) for _ in range(2)]
        z,h=self.mamba.step(self.encode(x)[:,0],h)
        return self.head(z),h

def state_bytes(h):
    if isinstance(h,torch.Tensor):return h.numel()*h.element_size()
    if h is None:return 0
    return sum(state_bytes(v) for v in h)

def step(net,x,h):
    return net.step(x,h) if isinstance(net,MambaProbe) else net(x,h)

def preflight(out):
    torch.manual_seed(2026)
    checks=[]
    for T in [1,50,100]:
        net=MambaProbe().cuda();seq=copy.deepcopy(net);seq.mamba.config.pscan=False
        x=torch.randn(2,T,1,8,device='cuda',requires_grad=True)
        xx=x.detach().clone().requires_grad_(True)
        p=net(x)[0];q=seq(xx)[0];p.square().sum().backward();q.square().sum().backward()
        e=float((p-q).abs().max());ge=max(float((a.grad-b.grad).abs().max()) for a,b in zip(net.parameters(),seq.parameters()))
        xe=float((x.grad-xx.grad).abs().max());assert max(e,ge,xe)<1e-4
        with torch.no_grad():
            h=None
            for i in range(T):r,h=net.step(x[:,i:i+1],h)
            se=float((p-r).abs().max());assert se<1e-4
            prefix=net.mamba(net.encode(x))[:,:max(1,T//2)]
            altered=x.detach().clone();altered[:,max(1,T//2):]+=10
            ce=float((prefix-net.mamba(net.encode(altered))[:,:max(1,T//2)]).abs().max());assert ce<1e-5
        checks.append(dict(T=T,parallel_sequential_error=e,parameter_gradient_error=ge,input_gradient_error=xe,stream_error=se,causality_error=ce,state_bytes_batch2=state_bytes(h)))
    assert len({r['state_bytes_batch2'] for r in checks})==1
    base.save(out/'implementation_checks.json',checks)

def stream_benchmark(net,frames):
    # Real, consecutive frames from one held-out episode, already normalized/on GPU.
    # Report synchronized end-to-end model call latency; do not time repeated static inputs.
    with torch.no_grad():
        h=None
        for i in range(100):_,h=step(net,frames[:,i:i+1],h)
        torch.cuda.synchronize();h=None;times=[];sizes=[]
        for i in range(frames.shape[1]):
            torch.cuda.synchronize();st=time.perf_counter()
            _,h=step(net,frames[:,i:i+1],h);torch.cuda.synchronize()
            times.append((time.perf_counter()-st)*1000);sizes.append(state_bytes(h))
    return dict(frames=len(times),batch=1,p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),max_ms=max(times),mean_ms=float(np.mean(times)),fraction_below_1ms=float(np.mean(np.array(times)<1)),state_bytes=min(sizes),state_bytes_max=max(sizes),latencies_ms=times,scope='eager PyTorch model-only, synchronized wall clock; real consecutive frames; input resident on GPU; 100 warmups, reset then rollout; no fused Mamba kernels')

def window_benchmark(net,x):
    with torch.no_grad():
        for _ in range(10):net(x)
        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            st=time.perf_counter();net(x);torch.cuda.synchronize();times.append((time.perf_counter()-st)*1000)
    return dict(batch=len(x),T=x.shape[1],p50_ms=float(np.median(times)),incremental_peak_bytes=torch.cuda.max_memory_allocated()-baseline)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    protocol=dict(task='Nowcast current ATI [Fx,Fy,Mx,My]',histories=[50,100],representations=['Scalar','MCF'],seeds=base.SEEDS,split=base.SPLIT,parameters={k:sum(p.numel() for p in cls().parameters()) for k,cls in [('GRU',GRUProbe),('Mamba',MambaProbe)]},mamba_config=vars(MambaConfig(50,2)),backend='mambapy 1.2.0 pure PyTorch parallel scan; eager recurrent step; no fused CUDA kernels',precision='FP32, TF32 disabled, deterministic',optimization=dict(lr=.001,weight_decay=.0001,batch=64,max_epochs=40,patience=8,selection='equal-source normalized validation MSE'),data='Inherited nowcast prepare; common subset requires 100-frame same-episode history plus predecessor for differences. Same inherited stride20/filter and end reserve30. Continuous histories; only anchor filtered.',primary='lateral moment xy macro source RMSE; paired source bootstrap of seed-averaged MSE',secondary='tangential force xy; batch1 streaming p50/p95/p99 latency; no guaranteed speed or accuracy gain',gpu=torch.cuda.get_device_name(),torch=torch.__version__,hashes={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,base.__file__,backend.__file__,scan_backend.__file__]})
    base.save(a.out/'protocol.json',protocol);preflight(a.out);start=time.time()
    inherited=a.out/'inherited_grid';inherited.mkdir()
    records,g,t,y,ids,norms=base.prepare(a.cache_root,inherited)
    keep=[]
    for i,(source,anchor) in enumerate(zip(g,t)):
        ep=records[int(source)]['ep'];lo=max(30,int(np.flatnonzero(ep==ep[anchor])[0]))
        if anchor>=lo+100:keep.append(i)
    keep=np.array(keep);g,t,y=g[keep],t[keep],y[keep]
    ids={k:np.flatnonzero(np.isin(g,v)) for k,v in base.SPLIT.items()}
    prior=np.load(inherited/'examples.npz')
    np.savez_compressed(a.out/'examples.npz',group=g,anchor=t,y=y,episode=prior['episode'][keep],dt=prior['dt'][keep],inherited_index=keep,**ids)
    for source,anchor in zip(g,t):assert np.all(records[int(source)]['ep'][anchor-100:anchor+1]==records[int(source)]['ep'][anchor])
    base.save(a.out/'sample_counts.json',{k:dict(windows=len(v),sources=len(np.unique(g[v]))) for k,v in ids.items()})
    base.save(a.out/'zero_reference.json',base.source_metrics(y[ids['test']],np.zeros_like(y[ids['test']]),g[ids['test']]))
    print('COUNTS',{k:len(v) for k,v in ids.items()},flush=True)
    results=[];base.benchmark=window_benchmark
    for arch,cls in [('GRU',GRUProbe),('Mamba',MambaProbe)]:
        folder=a.out/arch;folder.mkdir();base.Probe=cls
        for T in [50,100]:
            for name in ['Scalar','MCF']:
                for seed in base.SEEDS:
                    r=base.fit(records,g,t,y,ids,norms[name],name,T,seed,folder);r['architecture']=arch
                    if seed==7:
                        cp=torch.load(folder/(r['key']+'.pt'),weights_only=False);net=cls().cuda();net.load_state_dict(cp['state_dict']);net.eval()
                        ix=int(ids['test'][0]);source=int(g[ix]);anchor=int(t[ix]);ep=records[source]['ep'];end=min(anchor+1000,int(np.flatnonzero(ep==ep[anchor])[-1])+1)
                        raw=records[source][name][anchor:end];assert len(raw)>=100
                        frames=torch.from_numpy((raw-norms[name][0])/norms[name][1]).cuda()[None,:,None,:]
                        r['streaming']=stream_benchmark(net,frames);r['streaming'].update(source=source,start=anchor)
                        with torch.no_grad():
                            x=base.batch(records,g,t,ids['test'][:4],T,name,norms[name]);pred=net(x)[0];h=None
                            for j in range(T):stream,h=step(net,x[:,j:j+1],h)
                            err=float((pred-stream).abs().max());assert err<1e-4
                        r['trained_stream_parity_max_normalized_error']=err
                        del net
                    results.append(r);base.save(a.out/'results.json',results)
    base.save(a.out/'execution.json',dict(completed=True,models=len(results),seconds=time.time()-start))

if __name__=='__main__':main()
