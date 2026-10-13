from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
s=(p/'reflex_policy.py').read_text(encoding='utf-8')
s=s.replace('        self.hidden=None; self.streaming=False\n','').replace("            self.streaming=ck.get('streaming',False)\n",'')
start=s.index('            with torch.inference_mode():\n')
s=s[:start]+"            with torch.inference_mode(): value=float(self.model(torch.tensor(np.array(padded)[None],dtype=torch.float32))[0])*MAX_DELTA\n"+s[s.index('        return float(value)',start):]
for name,source in [('reflex_policy_window_v1.py',s),('reflex_policy_solver_velocity_v0.py',s.replace('.75*max(0.,linear-.00015)+.003*max(0.,angular-.01)','.30*max(0.,linear-.00015)+.002*max(0.,angular-.005)').replace('self.dt/.25','self.dt/.08'))]:
    (p/name).write_bytes(source.encode('utf-8'))
    print(name,hashlib.sha256((p/name).read_bytes()).hexdigest())

sim=(p/'reflex_sim.py').read_text(encoding='utf-8')
sim=sim.replace("'robot_body_names':list(env._robot.body_names),",'').replace('        previous_control_pose=initial_pose.copy()\n','')
begin=sim.index('            # Differentiate realized link poses;')
end=sim.index('            control_ns=',begin)
sim=sim[:begin]+"            delta,teacher,speed,angular_speed=controller.step(input_mcf,obj.data.root_com_vel_w[0].cpu().numpy().copy(),previous_angle)\n"+sim[end:]
(p/'reflex_sim_solver_velocity_v0.py').write_text(sim,encoding='utf-8')
expected=json.loads((p/'outputs/reflex_dev_oracle_01/provenance.json').read_text(encoding='utf-8'))['source_file_sha256']
expected=next(v for k,v in expected.items() if k.endswith('reflex_sim.py'))
assert hashlib.sha256((p/'reflex_sim_solver_velocity_v0.py').read_bytes()).hexdigest()==expected,'Old recorder reconstruction mismatch'

e=(p/'e2_dev.py').read_text(encoding='utf-8')
e=e.replace("    parser.add_argument('--normalization',choices=['legacy_zscore','robust_floor_clip8'],default='legacy_zscore')\n",'').replace("    protocol['normalization']=args.normalization\n",'')
start=e.index("        if args.normalization=='robust_floor_clip8':")
end=e.index('        def windows(rr):',start)
e=e[:start]+"        mean=fit_frames.mean(axis=0); std=np.maximum(fit_frames.std(axis=0),1e-6)\n        normalized={tid:(x-mean)/std for tid,x in frames.items()}\n"+e[end:]
e=e.replace("'mean':torch.tensor(mean),'std':torch.tensor(std),'normalization':args.normalization,\n                        'input_clip':8. if args.normalization=='robust_floor_clip8' else None,", "'mean':torch.tensor(mean),'std':torch.tensor(std),")
expected=json.loads((p/'outputs/e2_static_control_01/protocol.json').read_text(encoding='utf-8'))['script_sha256']
for source in (e,e.replace('\n','\r\n')):
    if hashlib.sha256(source.encode('utf-8')).hexdigest()==expected:
        (p/'e2_dev_static_v4.py').write_bytes(source.encode('utf-8')); break
else: raise RuntimeError('Old E2 source reconstruction mismatch')
print('Historical policy, recorder and E2 source hashes matched')
