"""Freeze A/B/C sampling before native outcomes are examined."""
from pathlib import Path
import json, hashlib, copy
root=Path(__file__).resolve().parent
out=root/'alignment_matrices'; out.mkdir(exist_ok=False)
original=json.loads((root/'phase_matrices/phase_all.json').read_text())
anchors=[x for x in original if x['direction'] in ('+Fx','-Fx') and x['amplitude_n']==.20 and x['point_m'][2]==.015]
alignment=copy.deepcopy(original)
for x in anchors:
    x=copy.deepcopy(x); x.update(trial_id=x['trial_id']+'_repeat1',repeat=1); alignment.append(x)
trigger=[copy.deepcopy(original[0])]
trigger[0].update(trial_id='trigger_zero',onset_s=.55)
for direction in ('+Fx','-Fx'):
    for force in (.18,.20,.225,.25):
        for e in (.01,.015,.02):
            x=copy.deepcopy(original[0]); x.update(trial_id=f"trigger_{'p' if direction[0]=='+' else 'n'}Fx_f{force:.3f}_e{e:.3f}",
                direction=direction,amplitude_n=force,point_m=[0.,0.,e],onset_s=.55)
            trigger.append(x)
for x in [x for x in trigger if x['amplitude_n']==.20 and x['point_m'][2]==.015]:
    x=copy.deepcopy(x); x.update(trial_id=x['trial_id']+'_repeat1',repeat=1); trigger.append(x)
for name,matrix in [('alignment',alignment),('trigger',trigger),('smoke',[original[0],*anchors])]:
    (out/f'{name}.json').write_text(json.dumps(matrix,indent=2),encoding='utf-8')
sources=['ALIGNMENT_PROTOCOL.md','aligned_sim.py','aligned_policy.py','run_alignment_suite.py','prepare_alignment_driver.py','prepare_alignment_matrix.py','reflex_policy.py']
paths=[root/x for x in sources]+list(out.glob('*.json'))+[root/'outputs/reflex_students_stream_01'/f'{kind}_selected.pt' for kind in ('scalar','mcf')]
freeze={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
(out/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
print('FROZEN',len(alignment),len(trigger),'formal native trials',len(alignment)*7+len(trigger)*6)
