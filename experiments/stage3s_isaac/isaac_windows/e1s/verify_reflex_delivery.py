"""Validate old-data preservation and every new physical / learned artifact."""
import json,hashlib,py_compile
from pathlib import Path
import numpy as np
import torch
from reflex_policy import ReflexGRU

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
old_count=0
for group in ('pilot_tangent_02','contact_02','contact_bringup_01','pure_loading_01'):
    root=ROOT/'outputs'/group
    paths=[root] if (root/'manifest.json').exists() else sorted(x.parent for x in root.glob('*/manifest.json'))
    for path in paths:
        manifest=json.loads((path/'manifest.json').read_text()); assert manifest['status']=='completed'
        for tid in manifest['completed_trials']:
            meta=json.loads((path/f'{tid}.json').read_text()); assert sha(path/f'{tid}.npz')==meta['npz_sha256']; old_count+=1
source_candidates={sha(p):str(p.relative_to(ROOT)) for p in ROOT.glob('*.py')}
for name in ('e2_dev_01','e2_history_control_01','e2_static_control_01','bounded_original_01'):
    d=json.loads((ROOT/'outputs'/name/'protocol.json').read_text()); assert d['script_sha256'] in source_candidates
new_count=0; steps=0; native=[]
for path in sorted((ROOT/'outputs').glob('reflex_*')):
    if not (path/'manifest.json').exists(): continue
    manifest=json.loads((path/'manifest.json').read_text()); assert manifest['status']=='completed' and not manifest['errors']
    prov=json.loads((path/'provenance.json').read_text())
    for source,digest in prov['source_file_sha256'].items():
        if Path(source).name in ('reflex_sim.py','reflex_policy.py','bounded_scaler.py','disturbance.py'): assert digest in source_candidates,(path,source)
    assert json.loads((path/'force_calibration.json').read_text())['passed']
    for tid in manifest['completed_trials']:
        meta=json.loads((path/f'{tid}.json').read_text()); assert sha(path/f'{tid}.npz')==meta['npz_sha256']
        with np.load(path/f'{tid}.npz') as d:
            assert d['tactile'].shape[1:]==(4,12,32) and d['mcf'].shape[1:]==(4,8)
            assert all(np.isfinite(d[k]).all() for k in d.files)
            assert np.array_equal(d['input_mcf_before'][1:],d['mcf'][:-1])
            steps+=len(d['mcf']); new_count+=1
    native.append(dict(directory=path.name,trials=len(manifest['completed_trials']),force_calibration=True))
model_count=0
for name in ('bounded_pure_01','bounded_original_01','reflex_students_01','reflex_students_stream_01'):
    root=ROOT/'outputs'/name; results=json.loads((root/'results.json').read_text()); assert results['status']=='completed'
    pts=[x for x in root.glob('*.pt') if not x.name.endswith('_selected.pt')]; model_count+=len(pts)
for name in ('reflex_students_01','reflex_students_stream_01'):
    root=ROOT/'outputs'/name; rows=json.loads((root/'results.json').read_text())['runs']
    for kind in ('mcf','scalar'):
        selected=min([r for r in rows if r['kind']==kind],key=lambda r:r['validation_rmse_rad'])
        assert sha(root/f'{kind}_selected.pt')==sha(root/f"{kind}_seed{selected['seed']}.pt")
        ck=torch.load(root/f'{kind}_selected.pt',map_location='cpu',weights_only=False)
        model=ReflexGRU(kind); model.load_state_dict(ck['state_dict']); model.eval()
        if ck.get('streaming'):
            torch.manual_seed(23); x=torch.randn(1,30,32 if kind=='mcf' else 8); h=None; yy=[]
            with torch.inference_mode():
                z,_=model.gru(x); whole=torch.sigmoid(model.head(z))
                for frame in x.unbind(1):
                    z,h=model.gru(frame[:,None],h); yy.append(torch.sigmoid(model.head(z)))
            assert (torch.cat(yy,dim=1)-whole).abs().max()<1e-6
reports=[]
for name in ('reflex_report_01','reflex_report_stream_01'):
    p=ROOT/'outputs'/name; d=json.loads((p/'verification.json').read_text()); assert d['status']=='passed'
    reports.append(dict(report=name,paired_trials=d['trials_checked']))
for p in [ROOT/'README.md',ROOT/'REFLEX_DELIVERY.md',ROOT/'REFLEX_PROTOCOL.md',ROOT/'REFLEX_STREAM_PROTOCOL.md',ROOT/'outputs/reflex_report_stream_01/REPORT.md']:
    t=p.read_text(encoding='utf-8'); assert '\ufffd' not in t and '??' not in t
for name in ('bounded_scaler.py','bounded_probe.py','reflex_policy.py','reflex_sim.py','reflex_metrics.py','train_reflex.py','train_reflex_stream.py','report_reflex.py','run_reflex_pipeline.py'):
    py_compile.compile(str(ROOT/name),doraise=True)
out=dict(status='passed',old_unchanged_npz=old_count,new_native_trials=new_count,new_synchronized_frames=steps,trained_models=model_count,
         native=native,reports=reports,selected_by_validation=True,streaming_equivalence=True,utf8_readback=True,syntax=True)
(ROOT/'reflex_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
files=[p for p in ROOT.glob('*') if p.is_file() and p.suffix in ('.py','.md','.ps1','.json')]
files+=[p for p in (ROOT/'outputs').rglob('*') if p.is_file() and any(s.startswith(('reflex_','bounded_')) for s in p.relative_to(ROOT/'outputs').parts)]
files=[p for p in files if p.name!='reflex_artifact_manifest.json']
(ROOT/'reflex_artifact_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
