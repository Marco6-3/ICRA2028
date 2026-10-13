"""Rebuild the replay input from authorized upstream trajectories and statistics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--official-repo',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'isaac_windows/dataset_raw_joint_positions.npz')
    a=p.parse_args(); source=a.official_repo/'Isaacsim_tactile_env/data'
    stats=torch.load(source/'dataset_normalizer.npz',weights_only=True,map_location='cpu')
    with np.load(source/'dataset_train.npz',allow_pickle=False) as d:
        lengths=d['traj_lengths'].astype(np.int64); out={'traj_lengths':lengths}
        for key in ('joint_states','actions'):
            x=d[key].astype(np.float32); lo=stats[f'stats.{key}.min'].numpy().astype(np.float32); hi=stats[f'stats.{key}.max'].numpy().astype(np.float32)
            assert x.shape==(int(lengths.sum()),16) and lo.shape==hi.shape==(16,)
            assert np.isfinite(x).all() and np.all(hi>=lo) and x.min()>=-1.0001 and x.max()<=1.0001
            out[key]=((x+1.)*.5*(hi-lo)+lo).astype(np.float32)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    if a.out.exists(): raise FileExistsError('Preserve existing replay input: '+str(a.out))
    np.savez_compressed(a.out,**out)
    sha=hashlib.sha256(a.out.read_bytes()).hexdigest()
    expected=json.loads((Path(__file__).resolve().parent/'DATA_SOURCES.json').read_text(encoding='utf-8'))['generated_input_sha256']
    if sha!=expected: raise RuntimeError('Input differs from recorded replay hash; check upstream files and NumPy version')
    print('REPLAY_INPUT_VERIFIED',len(lengths),'episodes',int(lengths.sum()),'frames',sha)

if __name__=='__main__': main()
