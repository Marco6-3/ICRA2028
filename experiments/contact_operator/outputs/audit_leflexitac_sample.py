"""Audit a public LeFlexiTac shard without claiming raw sensor semantics.
Run after validate_contact_operator.py. Requires numpy and pyarrow.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from validate_contact_operator import descriptor

root = Path(__file__).resolve().parent.parent
folder = root/'work'/'leflexitac'
folder.mkdir(parents=True, exist_ok=True)
base = 'https://huggingface.co/datasets/Tna001/tactile_test_tube_pyflexitac'
path = folder/'file-000.parquet'
if not path.exists():
    with urllib.request.urlopen(base+'/resolve/main/data/chunk-000/file-000.parquet', timeout=30) as r:
        path.write_bytes(r.read())
info_path = folder/'info.json'
if not info_path.exists():
    with urllib.request.urlopen(base+'/resolve/main/meta/info.json', timeout=30) as r:
        info_path.write_bytes(r.read())
info = json.loads(info_path.read_text(encoding='utf-8'))
table = pq.read_table(path)
x = np.array(table['observation.tactile.primary'].to_pylist())
t = np.array(table['timestamp'])
episode = np.array(table['episode_index'])
same = np.diff(episode) == 0
dt = np.diff(t)[same]
xx, yy = np.meshgrid(np.linspace(-1,1,32), np.linspace(-1,1,12))
xy = np.c_[xx.ravel(), yy.ravel()]
mass = x.sum((1,2))
pref = float(np.median(mass[(episode == 0)&(mass > 0)]))
static = np.array([descriptor(frame.ravel(), xy, threshold=0, pref=pref) for frame in x])
assert x.shape[1:] == (12,32) and (x >= 0).all()
assert np.isfinite(static).all()
result = dict(dataset=base, metadata_url=base+'/blob/main/meta/info.json',
    metadata_total_episodes=info['total_episodes'], metadata_total_frames=info['total_frames'],
    metadata_fps=info['fps'], shard='data/chunk-000/file-000.parquet',
    shard_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(x),
    shape=list(x.shape), episodes=np.unique(episode).tolist(),
    min=float(x.min()), max=float(x.max()), all_finite=bool(np.isfinite(x).all()),
    zero_frame_fraction=float(np.mean(mass == 0)),
    timestamp_interval_quantiles_s=np.quantile(dt,[0,.5,1]).tolist(),
    identical_adjacent_frame_fraction=float(np.mean(np.all(x[1:]==x[:-1],axis=(1,2))[same])),
    static_descriptor_finite=True, static_descriptor_shape=list(static.shape),
    P_ref_episode_0=pref,
    sensor_metadata={k:info['tactile_sensors']['primary'][k] for k in
                     ['rows','cols','threshold','noise_scale','init_frames','is_calibrated']},
    limits=['One shard, five episodes; no learned model or action prediction.',
            'Recorded values are 0..1; exact preprocessing and sensor timing not verified.',
            'Static 8-coordinate smoke check only; threshold=0 is a diagnostic convention.',
            'Zero mass is filled with zeros, not interpreted as measured central contact.',
            'No validation of onset labels, physical continuity, dynamics or real latency.'])
(root/'outputs'/'leflexitac_sample_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
