import json,hashlib
from pathlib import Path
from disturbance import Trial
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'phase_matrices'; OUT.mkdir(exist_ok=False)
directions=['+Fx','-Fx','+Fy','-Fy']; forces=[.10,.125,.15,.175,.20,.225,.25]; offsets=[0.,.005,.010,.015,.020]
phase=[Trial('phase_zero',direction='zero',frame='reference',onset_s=.45).metadata()]
for d,direction in enumerate(directions):
    for f,force in enumerate(forces):
        for e,offset in enumerate(offsets):
            phase.append(Trial(f'phase_d{d}_f{f}_e{e}',direction,amplitude_n=force,point_m=(0,0,offset),frame='reference',onset_s=.45,duration_s=.25).metadata())
mechanism=[]
for repeat in range(2):
    mechanism.append(Trial(f'mech_zero_r{repeat}',direction='zero',frame='reference',repeat=repeat,onset_s=.45).metadata())
    for d,direction in enumerate(directions):
        torque=[0.,0.,0.]; torque[1 if d<2 else 0]=(.003,-.003,-.003,.003)[d]
        for mode in ('force','torque','combined'):
            mechanism.append(Trial(f'mech_{mode}_d{d}_r{repeat}',direction if mode!='torque' else 'zero',
                amplitude_n=.2 if mode!='torque' else 0.,point_m=(0,0,.015 if mode=='combined' else 0.),
                torque_nm=tuple(torque) if mode=='torque' else (0,0,0),frame='reference',repeat=repeat,onset_s=.45,duration_s=.25).metadata())
pulse=[Trial(f'pulse_p{p}_r{r}',direction='zero',preload_delta_rad=0.,frame='reference',repeat=r,onset_s=.45,duration_s=.25).metadata() for p in range(3) for r in range(2)]
for name,matrix in [('phase_all.json',phase),('phase_oracle_fx.json',[x for x in phase if x['direction'] in ('zero','+Fx')]),('mechanism.json',mechanism),('pulse.json',pulse)]:
    (OUT/name).write_text(json.dumps(matrix,indent=2),encoding='utf-8')
s=(ROOT/'reflex_sim.py').read_text(encoding='utf-8')
s=s.replace("'nominal','high','oracle','mcf','scalar'","'nominal','high','oracle','mcf','scalar','pulse'")
s=s.replace('parser.add_argument(\'--checkpoint\',type=Path)',"parser.add_argument('--checkpoint',type=Path)\nparser.add_argument('--pulse-mode',action='store_true')")
s=s.replace('controller=Controller(args.controller,cfg.physics_dt,args.checkpoint)',"controller=Controller('nominal' if args.controller=='pulse' else args.controller,cfg.physics_dt,args.checkpoint)")
s=s.replace('            previous_control_pose=pose.copy()',"            if args.controller=='pulse':\n                pulse_size=(0.,.001,.003)[int(tr.trial_id.split('_')[1][1:])]\n                delta=pulse_size*tr.gain(t)\n            previous_control_pose=pose.copy()")
s=s.replace("'sensor_slots':[", "'sensor_slots':[")
(ROOT/'phase_sim.py').write_text(s,encoding='utf-8')
spec={'forces_n':forces,'offsets_m':offsets,'directions':directions,'frozen_models':{},'matrix_sha256':{},'protocol_sha256':hashlib.sha256((ROOT/'PHASE_PROTOCOL.md').read_bytes()).hexdigest()}
for kind in ('mcf','scalar'):
    p=ROOT/'outputs/reflex_students_stream_01'/f'{kind}_selected.pt'; spec['frozen_models'][kind]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in OUT.glob('*.json'): spec['matrix_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'freeze.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
print('PHASE_FROZEN',len(phase),'grid conditions',len(mechanism),'mechanism trials',len(pulse),'actuator trials')
