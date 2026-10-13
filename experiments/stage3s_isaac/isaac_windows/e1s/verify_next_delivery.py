"""Checks source snapshots, unchanged source data, model/prediction manifests."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source_candidates={sha(p):str(p.relative_to(BASE)) for p in BASE.glob('*.py')}
verification={}
for name in ('temporal_01','e2_dev_01','e2_history_control_01','e2_static_control_01'):
    root=BASE/'outputs'/name
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    results=json.loads((root/'results.json').read_text(encoding='utf-8'))
    assert results['status']=='completed'
    assert protocol['script_sha256'] in source_candidates, name+' source snapshot absent'
    if name!='temporal_01':
        runs=json.loads((root/'runs.json').read_text(encoding='utf-8'))
        models=list(root.glob('*.pt')); predictions=list(root.glob('*predictions.npz'))
        assert len(models)==len(predictions)==len(runs)==results['runs']
        for path in predictions:
            with np.load(path,allow_pickle=False) as d:
                for key in ('truth','prediction','truth_FxFyMxMy','prediction_FxFyMxMy'):
                    assert np.isfinite(d[key]).all(),str(path)
                assert len(set(d['trial_ids']))==36
        latency=json.loads((root/'software_latency.json').read_text(encoding='utf-8'))
        assert latency['status']=='completed'
        assert all(row['cached_window_equivalence_max_error']<=1e-5 for rows in latency['runs'].values() for row in rows)
        verification[name]={'runs':len(runs),'models':len(models),'finite_predictions':True,
                            'source_snapshot':source_candidates[protocol['script_sha256']]}
    else:
        verification[name]={'status':'completed','source_snapshot':source_candidates[protocol['script_sha256']]}

old=BASE/'outputs/pilot_tangent_02'
manifest=json.loads((old/'manifest.json').read_text())
assert len(manifest['completed_trials'])==135
for tid in manifest['completed_trials']:
    meta=json.loads((old/f'{tid}.json').read_text())
    assert sha(old/f'{tid}.npz')==meta['npz_sha256']
verification['old_pilot']={'unchanged_npz_hashes':135}

for group in ('contact_02','contact_bringup_01','pure_loading_01'):
    root=BASE/'outputs'/group
    paths=[root] if (root/'manifest.json').exists() else sorted(p.parent for p in root.glob('*/manifest.json'))
    count=0
    for path in paths:
        provenance=json.loads((path/'provenance.json').read_text())
        source=next(v for k,v in provenance['source_file_sha256'].items() if k.endswith('contact_pilot.py'))
        assert source in source_candidates, str(path)+' source snapshot absent'
        manifest=json.loads((path/'manifest.json').read_text())
        assert manifest['status']=='completed'
        for tid in manifest['completed_trials']:
            meta=json.loads((path/f'{tid}.json').read_text())
            assert sha(path/f'{tid}.npz')==meta['npz_sha256']
            count+=1
    verification[group]={'loaded_trials_hash_verified':count}

for p in [BASE/'NEXT_DELIVERY.md',BASE/'README.md']+list(BASE.glob('*PROTOCOL.md')):
    text=p.read_text(encoding='utf-8')
    assert '\ufffd' not in text and '??' not in text,str(p)
verification['utf8_readback']='passed'

artifact_paths=[p for p in BASE.rglob('*') if p.is_file() and (
    p.parent==BASE and p.suffix in ('.py','.ps1','.md','.json') or
    any(name in p.parts for name in ('temporal_01','e2_dev_01','e2_history_control_01','e2_static_control_01')))]
artifact_paths=[p for p in artifact_paths if p.name not in ('next_verification.json','next_artifact_manifest.json')]
artifacts={str(p.relative_to(BASE)):sha(p) for p in artifact_paths}
(BASE/'next_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
(BASE/'next_artifact_manifest.json').write_text(json.dumps(artifacts,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
