import argparse,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser(); p.add_argument('source',type=Path); p.add_argument('--timing',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
freeze=json.loads((ROOT/'phase_matrices/freeze.json').read_text())
assert sha(ROOT/'PHASE_PROTOCOL_INITIAL.md')==freeze['protocol_sha256']
for kind,digest in freeze['frozen_models'].items(): assert sha(ROOT/'outputs/reflex_students_stream_01'/f'{kind}_selected.pt')==digest
for name,digest in freeze['matrix_sha256'].items(): assert sha(ROOT/'phase_matrices'/name)==digest
pipeline=json.loads((a.source/'pipeline.json').read_text()); assert pipeline['status']=='completed'
trials=0; frames=0; init={}; hashes={}; rep=[]
for name in ('mechanism','pulse','nominal','high','mcf','scalar','oracle_fx','timing_intervention'):
    directory=a.timing if name=='timing_intervention' else a.source/name
    manifest=json.loads((directory/'manifest.json').read_text()); assert manifest['status']=='completed' and not manifest['errors']
    prov=json.loads((directory/'provenance.json').read_text())
    assert prov['support']=='removed' and prov['fixed_robot_base']
    assert not any(x['enabled'] for x in prov['support_collision_paths'])
    assert json.loads((directory/'force_calibration.json').read_text())['passed']
    for path,digest in prov['source_file_sha256'].items():
        if Path(path).parent==ROOT: assert sha(Path(path))==digest,(name,path)
    for tid in manifest['completed_trials']:
        meta=json.loads((directory/f'{tid}.json').read_text()); path=directory/f'{tid}.npz'; assert sha(path)==meta['npz_sha256']
        hashes[name+'/'+tid+'.npz']=meta['npz_sha256']
        with np.load(path) as d:
            assert d['tactile'].shape==(180,4,12,32) and d['mcf'].shape==(180,4,8)
            assert all(np.isfinite(d[k]).all() for k in d.files)
            assert np.array_equal(d['input_mcf_before'][1:],d['mcf'][:-1])
            assert np.all((d['controller_delta_rad']>=0)&(d['controller_delta_rad']<=.003))
            assert np.allclose(np.diff(d['observation_time_s']),1/120)
            fields={k:d[k].copy() for k in ('initial_object_pose','initial_robot_body_pose','force_world_n','torque_com_world_nm','step_start_s')}
            if name=='nominal': init[tid]=fields
            if name in ('high','mcf','scalar','oracle_fx'):
                for key,val in fields.items(): assert np.allclose(val,init[tid][key],rtol=0,atol=1e-8),(name,tid,key)
        trials+=1; frames+=180
    rep.append({'group':name,'trials':len(manifest['completed_trials']),'force_calibration':True,'data_hashes':True})
assert trials==644 and frames==115920
old=json.loads((ROOT/'reflex_artifact_manifest.json').read_text())
for path,digest in old.items(): assert sha(ROOT/path)==digest,'Prior frozen artifact changed: '+path
out={'status':'passed','trials':trials,'synchronized_frames':frames,'frozen_models_unchanged':True,'matrix_and_initial_protocol_unchanged':True,
     'prior_reflex_archive_unchanged':True,'groups':rep,'paired_initial_state_and_loads':True,'causal_input':True,'data_hashes':hashes}
(a.report/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('PHASE_VERIFIED',trials,frames,'frames; frozen models / prior artifacts unchanged')
