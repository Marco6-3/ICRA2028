"""CPU software entry latency, exact bounded histories with cached encodings."""
from collections import deque
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from e2_dev import Probe
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ICRA2028_Stage3S_starter'))
from mcf import per_finger_features


def feature(raw,previous,kind):
    if kind=='raw': return raw
    if kind=='scalar':
        total=np.maximum(raw.astype(float),0).sum(axis=(1,2))
        old=total if previous is None else np.maximum(previous.astype(float),0).sum(axis=(1,2))
        return np.stack([total,total-old],axis=1).astype(np.float32).reshape(-1)
    _,mcf=per_finger_features(raw,previous)
    if kind in ('static','s4'): return mcf[:,:4].reshape(-1)
    if kind=='no_cop': return mcf[:,[0,3,4,7]].reshape(-1)
    if kind=='no_area': return mcf[:,[0,1,2,4,5,6]].reshape(-1)
    return mcf.reshape(-1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('directory',type=Path); p.add_argument('--raw-trace',type=Path,required=True)
    args=p.parse_args(); root=args.directory
    scores=json.loads((root/'results.json').read_text(encoding='utf-8'))
    if scores['status']!='completed': raise RuntimeError('Training incomplete')
    torch.set_num_threads(2)
    with np.load(args.raw_trace,allow_pickle=False) as data: raw=data['tactile']
    results={}
    for path in sorted(root.glob('*_seed*.pt')):
        checkpoint=torch.load(path,weights_only=True,map_location='cpu')
        kind=checkpoint['kind']; history=checkpoint['history']
        model=Probe(kind,checkpoint['hidden']); model.load_state_dict(checkpoint['state_dict']); model.eval()
        mean=checkpoint['mean'].numpy(); std=checkpoint['std'].numpy()
        cache=deque(maxlen=history); previous=None; samples=[]; descriptors=[]; networks=[]; max_error=0.
        normalized_window=deque(maxlen=history)
        with torch.inference_mode():
            for i in range(400):
                j=i%len(raw)
                if j==0: cache.clear(); normalized_window.clear(); previous=None
                begin=time.perf_counter_ns()
                values=feature(raw[j],previous,kind); normalized=(values-mean)/std
                if checkpoint.get('input_clip') is not None:
                    normalized=np.clip(normalized,-checkpoint['input_clip'],checkpoint['input_clip'])
                previous=raw[j]
                inp=torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
                middle=time.perf_counter_ns()
                if kind=='raw': encoded=model.encoder(inp).unsqueeze(1)
                else: encoded=inp.unsqueeze(1)
                cache.append(encoded); normalized_window.append(inp)
                if len(cache)<history: continue
                if kind.startswith('static'): output=model.head(model.spatial(encoded[:,0]))
                else:
                    seq=torch.cat(list(cache),dim=1); out,_=model.gru(seq); output=model.head(out[:,-1])
                end=time.perf_counter_ns()
                if i>40 and j>=16:
                    samples.append((end-begin)/1e6); descriptors.append((middle-begin)/1e6); networks.append((end-middle)/1e6)
                if j in (20,40,60):
                    reference=model(torch.stack(list(normalized_window),dim=1))
                    max_error=max(max_error,float((reference-output).abs().max()))
        if max_error>1e-5: raise RuntimeError('Cached encoding differs from exact model window')
        name=path.stem.rsplit('_seed',1)[0]
        results.setdefault(name,[]).append({'checkpoint':path.name,'cached_window_equivalence_max_error':max_error,
            'samples':len(samples),'full_raw_entry_p50_ms':float(np.quantile(samples,.5)),
            'full_raw_entry_p95_ms':float(np.quantile(samples,.95)),
            'descriptor_and_normalization_p95_ms':float(np.quantile(descriptors,.95)),
            'network_and_encoder_p95_ms':float(np.quantile(networks,.95))})
    summaries={name:{k:float(np.mean([r[k] for r in rows])) for k in
        ('full_raw_entry_p50_ms','full_raw_entry_p95_ms','descriptor_and_normalization_p95_ms','network_and_encoder_p95_ms')} for name,rows in results.items()}
    output={'status':'completed','boundary':'CPU software from already available Raw to exact bounded-window neural output. No sensor transport, actuator, simulation or energy. Dense per-frame encoder cached; numerical equivalence checked.',
        'torch':torch.__version__,'threads':2,'history_reset':'trial boundaries only; GRU bounded window reset matches training',
        'runs':results,'summaries':summaries}
    (root/'software_latency.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    scores=scores['summaries']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    primary=[n for n in ['scalar_t16','static_s4','static_mcf','mcf_t16','dense_t16','dense_t17'] if n in scores]
    for name in primary:
        r=scores[name]; latency=summaries[name]['full_raw_entry_p95_ms']
        axes[0].scatter(latency,r['mean_accuracy'],s=80)
        axes[0].annotate(name,(latency,r['mean_accuracy']),xytext=(5,5),textcoords='offset points')
    axes[0].set_xlabel('Raw-entry CPU P95 (ms), bounded history'); axes[0].set_ylabel('Held-out condition macro accuracy')
    axes[0].set_title('Fixed-budget development comparison')
    for pair in [('mcf_half','mcf_t16'),('dense_half','dense_t16')]:
        if not all(n in scores for n in pair): continue
        axes[1].plot([.5,1],[scores[name]['mean_accuracy'] for name in pair],marker='o',label=pair[1])
    axes[1].set_xticks([.5,1]); axes[1].set_xlabel('Supervised frame budget fraction, same conditions')
    axes[1].set_ylabel('Condition macro accuracy')
    if axes[1].lines: axes[1].legend()
    fig.tight_layout(); fig.savefig(root/'cost_accuracy.png',dpi=150); plt.close(fig)
    print('SOFTWARE_LATENCY',summaries)


if __name__=='__main__': main()
