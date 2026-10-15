from pathlib import Path
import hashlib,json
from disturbance import Trial
ROOT=Path(__file__).resolve().parent
matrix=[Trial(f'delay_d{i}_r{r}',direction='+Fx',amplitude_n=.2,point_m=(0,0,.015),frame='reference',onset_s=.45,duration_s=.25,repeat=r).metadata() for i in range(6) for r in range(2)]
(ROOT/'timing_intervention_matrix.json').write_text(json.dumps(matrix,indent=2),encoding='utf-8')
s=(ROOT/'reflex_sim.py').read_text(encoding='utf-8')
s=s.replace("'nominal','high','oracle','mcf','scalar'","'nominal','high','oracle','mcf','scalar','timed'")
s=s.replace('controller=Controller(args.controller,cfg.physics_dt,args.checkpoint)',"controller=Controller('nominal' if args.controller=='timed' else args.controller,cfg.physics_dt,args.checkpoint)")
s=s.replace('            previous_control_pose=pose.copy()',"            if args.controller=='timed':\n                delay=(-.1,0.,1/120,.025,.05,.1)[int(tr.trial_id.split('_')[1][1:])]\n                begin=tr.onset_s+delay; ending=tr.onset_s+tr.duration_s\n                delta=0. if t<begin else .003 if t<ending else .003*np.exp(-(t-ending)/.25)\n            previous_control_pose=pose.copy()")
(ROOT/'timing_intervention_sim.py').write_text(s,encoding='utf-8')
# Keep the original frozen protocol bytes available after the explicitly recorded
# sensitivity addition; never silently replace the SHA captured before scanning.
p=ROOT/'PHASE_PROTOCOL.md'; protocol=p.read_text(encoding='utf-8'); start=protocol.index('追加诊断说明'); end=protocol.index('## 机制分离与执行器探针')
original=protocol[:start]+protocol[end:]; expected=json.loads((ROOT/'phase_matrices/freeze.json').read_text())['protocol_sha256']
assert hashlib.sha256(original.encode('utf-8')).hexdigest()==expected
(ROOT/'PHASE_PROTOCOL_INITIAL.md').write_bytes(original.encode('utf-8'))
print('Frozen timing intervention and original phase protocol snapshot')
