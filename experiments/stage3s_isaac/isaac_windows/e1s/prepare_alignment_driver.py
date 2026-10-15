"""Generate an isolated native driver; assert every edit to the frozen driver."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parent
source = root / 'phase_sim.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '91a584a4314dc3d5329ac8a773c9451048c911eee3e37d5775ec8df7e18af639'
s = source.read_text(encoding='utf-8')

def replace(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)

replace("choices=['nominal','high','oracle','mcf','scalar','pulse']", "choices=['fixed','mcf','scalar']")
replace("parser.add_argument('--pulse-mode',action='store_true')", """parser.add_argument('--aligned-total-n',type=float,required=True)
parser.add_argument('--trigger',choices=['none','pulse','latch'],default='none')
parser.add_argument('--release-s',type=float,default=.45)
parser.add_argument('--reference-file',type=Path,required=True)""")
replace('    from reflex_policy import Controller', '    from aligned_policy import AlignedController')
replace("    reference_needs_fixture_update=bool(args.fixture_npz and args.reference_from_finger is not None)", """    reference=np.asarray(json.loads(args.reference_file.read_text(encoding='utf-8'))['reference_rotation_world_from_load'])
    reference_needs_fixture_update=False
    provenance.update(reference_rotation_world_from_load=reference.tolist(),aligned_total_n=args.aligned_total_n,
        trigger=args.trigger,release_s=args.release_s,absolute_extra_close_limits_rad=[0.,.003])
    for filename in ('aligned_policy.py','prepare_alignment_driver.py'):
        path=Path(__file__).with_name(filename)
        provenance['source_file_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()""")
start=s.index('        preload_history=[]; stable=0; close=0.; initial_gate=False')
end=s.index('        if reference_needs_fixture_update:',start)
s=s[:start]+'''        # Calibrate total PHYSICAL normal force, not the SDF response sum.
        # Same deterministic loop and serialized initial state for all policies.
        preload_history=[]; stable=0; close=0.; initial_gate=False
        for attempt in range(240):
            sums,normal=contact_values()
            gate=bool(np.all(sums[slots]>=args.gate_proxy_min) and np.all(normal[slots]>=args.gate_normal_min_n))
            value=float(normal[slots].sum())
            target_ready=abs(value-args.aligned_total_n)<=.01
            stable=stable+1 if gate and target_ready else 0
            preload_history.append({'step':attempt,'proxy':sums.tolist(),'normal_n':normal.tolist(),
                'closing_delta_rad':close,'gate':gate})
            if stable>=12:
                initial_gate=True
                break
            adjustment=np.clip(.0007*(args.aligned_total_n-value),-.0001,.0001)
            new_close=float(np.clip(close+adjustment,0.,.003))
            action[finger_ids]+=np.array([-1,1])*(new_close-close)
            close=new_close
            env.step(action)
        init={'trial_id':tr.trial_id,'mode':'aligned-total-normal-force','gate_passed':initial_gate,
              'initial_configuration':initial,'target_total_n':args.aligned_total_n,
              'feedback_target_reached':target_ready,'final_closing_delta_rad':close,'history':preload_history}
        (args.out/f'{tr.trial_id}_initialization.json').write_text(json.dumps(init,indent=2),encoding='utf-8')
        if not (initial_gate and target_ready):
            raise RuntimeError('Aligned total preload calibration failed: '+str(init))
''' + s[end:]
replace("        controller=Controller('nominal' if args.controller=='pulse' else args.controller,cfg.physics_dt,args.checkpoint)", "        controller=AlignedController(args.controller,cfg.physics_dt,args.checkpoint,close,args.trigger,args.release_s)")
replace("'expert_linear_speed_m_s','expert_angular_speed_rad_s')", "'expert_linear_speed_m_s','expert_angular_speed_rad_s',\n              'raw_policy_delta_rad','pre_policy_delta_rad','absolute_extra_close_rad','trigger_fired','trigger_score')")
start=s.index('            delta,teacher,speed,angular_speed=controller.step')
end=s.index('            previous_control_pose=pose.copy()',start)
s=s[:start]+'''            delta,prediction,fired,score=controller.step(input_mcf,t)
            teacher=0.; speed=0.; angular_speed=0.
''' + s[end:]
replace('values=dict(obs,input_mcf_before=previous_input,', 'values=dict(obs,raw_policy_delta_rad=prediction,pre_policy_delta_rad=controller.pre_prediction,absolute_extra_close_rad=close+delta,trigger_fired=fired,trigger_score=score,input_mcf_before=previous_input,')
replace("'controller':args.controller,'trial':tr.metadata(),", "'controller':args.controller,'trigger':args.trigger,'target_total_preload_n':args.aligned_total_n,'trial':tr.metadata(),")
(root/'aligned_sim.py').write_text(s,encoding='utf-8',newline='\n')
print('GENERATED',len(s),'characters')
