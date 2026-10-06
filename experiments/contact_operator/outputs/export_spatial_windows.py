"""Export complete qualifying real windows and an explicitly illustrative plot."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'/'spatial_contact_screen'
audit=json.loads((OUT/'sequence_audit.json').read_text(encoding='utf-8'))
windows=[];pressure=[];previous_pressure=[];wrench=[];source=[];rows=[]
for meta in audit:
    if not meta['calibration_ok']:continue
    group=int(meta['source_sequence'].split('obj')[-1])
    path=ROOT/'work'/'taf'/f'obj{group}.parquet'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    table=pq.read_table(path)
    raw=np.asarray(table['observation.pressure_matrix'].to_pylist(),dtype=np.float32)
    ft=np.asarray(table['observation.force_torque'].to_pylist(),dtype=np.float64)
    base=np.median(raw[:30],axis=0)
    threshold=3*1.4826*np.median(abs(raw[:30]-base),axis=0)
    delta=raw-base
    weights=np.where(delta>threshold,delta,0)
    ft-=np.median(ft[:30],axis=0)
    for window in meta['stable_windows']:
        if window['max_shear_ratio']>.2:continue
        ix=np.arange(window['start_row'],window['end_row']+1)
        assert len(ix)==15 and np.std(weights[ix].sum((1,2)))/np.mean(weights[ix].sum((1,2)))<=.05+1e-6
        windows.append(window);pressure.append(weights[ix]);previous_pressure.append(weights[ix-1])
        wrench.append(ft[ix]);source.append(group);rows.append(ix)
pressure=np.asarray(pressure);wrench=np.asarray(wrench)
np.savez_compressed(OUT/'stable_load_real_windows.npz',pressure=pressure,
    previous_pressure=np.asarray(previous_pressure),wrench=wrench,
    source_group=np.asarray(source),original_rows=np.asarray(rows))
(OUT/'stable_load_windows_index.json').write_text(json.dumps(windows,indent=2),encoding='utf-8')

# Choose the largest centroid change only to illustrate already-qualified data.
# This post-screen visualization is not an unbiased test or model metric.
best=int(np.argmax([w['centroid_span_normalized'] for w in windows]))
w=pressure[best];ft=wrench[best];metadata=windows[best]
xx,yy=np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,12))
p=w.sum((1,2));normal=abs(ft[:,2]);cx=w.reshape(-1,144)@xx.ravel()/p;cy=w.reshape(-1,144)@yy.ravel()/p
lever=np.c_[-ft[:,4]/ft[:,2],ft[:,3]/ft[:,2]]
time=np.arange(15)/30
fig,axes=plt.subplots(2,2,figsize=(10,7))
axes[0,0].plot(time,p/p.mean(),label='Pressure sum / mean')
axes[0,0].plot(time,normal/normal.mean(),label='ATI |Fz| / mean')
axes[0,0].set(ylabel='Relative load',xlabel='Stored time (s)',title='Approximately stable loads');axes[0,0].legend()
axes[0,1].plot(time,cx,label='cx');axes[0,1].plot(time,cy,label='cy')
axes[0,1].set(ylabel='Normalized sensor coordinate',xlabel='Stored time (s)',title='Observed centroid change');axes[0,1].legend()
axes[1,0].plot(time,lever[:,0],label='-Ty / Fz');axes[1,0].plot(time,lever[:,1],label='Tx / Fz')
axes[1,0].set(ylabel='Stored moment / force units',xlabel='Stored time (s)',title='Independent ATI label');axes[1,0].legend()
axes[1,1].plot(cx,cy,'o-');axes[1,1].scatter(cx[0],cy[0],c='green',label='Start');axes[1,1].scatter(cx[-1],cy[-1],c='red',label='End')
axes[1,1].set(xlabel='cx',ylabel='cy',title='Contact redistribution cue');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle(f"Illustrative qualified window: {metadata['source_sequence']}, rows {metadata['start_row']}..{metadata['end_row']}")
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'stable_window_example.png',dpi=160);plt.close(fig)
(OUT/'window_export_verification.json').write_text(json.dumps(dict(exported_windows=len(windows),
    pressure_shape=list(pressure.shape),all_finite=bool(np.isfinite(pressure).all() and np.isfinite(wrench).all()),
    illustration_selection='Largest centroid span among already independently gated windows; visualization only',
    illustrated_window=metadata),indent=2),encoding='utf-8')
print('EXPORTED',len(windows),pressure.shape)
