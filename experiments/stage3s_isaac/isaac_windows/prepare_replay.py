"""Restore upstream joint trajectories with its published min/max statistics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

root = Path(__file__).resolve().parent
data_dir = Path('C:/isaac_ws/FlexiTac-IsaacSim-Simulation/Isaacsim_tactile_env/data')
dataset_path = data_dir / 'dataset_train.npz'
stats_path = data_dir / 'dataset_normalizer.npz'
stats = torch.load(stats_path, map_location='cpu', weights_only=True)
with np.load(dataset_path, allow_pickle=False) as dataset:
    lengths = np.asarray(dataset['traj_lengths'], dtype=np.int64)
    out = {'traj_lengths': lengths}
    ranges = {}
    for key in ('joint_states', 'actions'):
        x = np.asarray(dataset[key], dtype=np.float32)
        lo = stats[f'stats.{key}.min'].detach().numpy().astype(np.float32)
        hi = stats[f'stats.{key}.max'].detach().numpy().astype(np.float32)
        assert x.shape == (int(lengths.sum()), 16)
        assert lo.shape == hi.shape == (16,)
        assert np.isfinite(x).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
        assert np.all(hi >= lo)
        assert x.min() >= -1.0001 and x.max() <= 1.0001
        raw = (x + 1.0) * 0.5 * (hi - lo) + lo
        out[key] = raw.astype(np.float32)
        ranges[key] = {'min': raw.min(axis=0).tolist(), 'max': raw.max(axis=0).tolist()}
raw_path = root / 'dataset_raw_joint_positions.npz'
np.savez_compressed(raw_path, **out)
report = {'episodes': len(lengths), 'frames': int(lengths.sum()), 'raw_joint_order': 'unchanged upstream DATASET_JOINT_ORDER', 'normalization_formula': '(x+1)*0.5*(max-min)+min', 'ranges': ranges, 'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (dataset_path, stats_path, raw_path)}}
(root / 'dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('PREPARED', raw_path, 'episodes', len(lengths), 'frames', int(lengths.sum()))
