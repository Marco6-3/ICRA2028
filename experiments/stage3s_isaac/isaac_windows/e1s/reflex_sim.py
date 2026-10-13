"""E1-S physical loading and synchronized four-finger recording on upstream ALOHA."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from disturbance import Trial, pilot_matrix, rotation_wxyz

parser=argparse.ArgumentParser()
parser.add_argument('--controller',choices=['nominal','high','oracle','mcf','scalar'],default='nominal')
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--official-repo', type=Path, default=Path('C:/isaac_ws/FlexiTac-IsaacSim-Simulation'))
parser.add_argument('--dataset-npz',type=Path, default=Path(__file__).resolve().parents[1]/'dataset_raw_joint_positions.npz')
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--matrix',type=Path)
parser.add_argument('--repeats',type=int,default=3)
parser.add_argument('--limit-trials',type=int)
parser.add_argument('--episode',type=int,default=3)
parser.add_argument('--hold-step',type=int,default=219)
parser.add_argument('--settle-steps',type=int,default=30)
parser.add_argument('--trial-steps',type=int,default=90)
parser.add_argument('--amplitude-n',type=float,default=.2)
parser.add_argument('--load-frame',choices=['world','object','reference'],default='world')
parser.add_argument('--reference-wxyz',type=float,nargs=4,default=[1,0,0,0])
parser.add_argument('--reference-from-finger',type=int,choices=range(4))
parser.add_argument('--target',choices=['plug','socket'],default='plug')
parser.add_argument('--plug-scale',type=float,default=1.06)
parser.add_argument('--normal-offset',type=float,default=.0036)
parser.add_argument('--threshold',type=float,default=0.)
parser.add_argument('--support',choices=['table','removed'],default='table')
parser.add_argument('--preload-mode',choices=['open','proxy','normal-force'],default='open')
parser.add_argument('--preload-target',type=float,default=8.)
parser.add_argument('--normal-target-n',type=float,default=.6)
parser.add_argument('--gate-proxy-min',type=float,default=2.)
parser.add_argument('--gate-normal-min-n',type=float,default=.05)
parser.add_argument('--preload-max-steps',type=int,default=120)
parser.add_argument('--preload-max-close-rad',type=float,default=.004)
parser.add_argument('--fixture-npz',type=Path)
parser.add_argument('--initial-close-rad',type=float,default=0.)
parser.add_argument('--initial-target-offset-z-m',type=float,default=0.)
parser.add_argument('--bringup-grid',action='store_true')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.out.exists():
    raise FileExistsError('Use a new output directory to preserve existing evidence')
if args.repeats<1 or args.settle_steps<1 or args.trial_steps<1 or args.hold_step<0:
    raise ValueError('Invalid counts')
trials=([Trial(**x) for x in json.loads(args.matrix.read_text(encoding='utf-8'))]
        if args.matrix else pilot_matrix(args.repeats))
if not args.matrix:
    for tr in trials:
        tr.amplitude_n=args.amplitude_n
        tr.frame=args.load_frame
if args.limit_trials:
    trials=trials[:args.limit_trials]
initialization_plan={}
if args.bringup_grid:
    trials=[]
    for i,(z,close) in enumerate((z,c) for z in (0.,.02,.04) for c in (0.,.0015,.003)):
        for repeat in range(2):
            tid=f'initial{i:02d}_r{repeat:02d}'
            trials.append(Trial(tid,direction='zero',frame='reference',repeat=repeat))
            initialization_plan[tid]={'target_z_offset_m':z,'initial_close_rad':close}
if not trials or len({tr.trial_id for tr in trials})!=len(trials):
    raise ValueError('Require unique, nonempty trial IDs')
if any(Path(tr.trial_id).name!=tr.trial_id or '/' in tr.trial_id or '\\' in tr.trial_id for tr in trials):
    raise ValueError('Invalid trial ID')
with np.load(args.dataset_npz,allow_pickle=False) as d:
    lengths=d['traj_lengths'].astype(int)
    start=int(lengths[:args.episode].sum())
    targets=np.repeat(d['joint_states'][start:start+lengths[args.episode]],3,axis=0).astype(np.float32)
if args.hold_step>=len(targets):
    raise ValueError('Hold step out of range')
args.out.mkdir(parents=True)
(args.out/'matrix.json').write_text(json.dumps([t.metadata() for t in trials],indent=2),encoding='utf-8')
(args.out/'initialization_plan.json').write_text(json.dumps(initialization_plan,indent=2),encoding='utf-8')
app=AppLauncher(args).app
sys.path.insert(0,str(args.official_repo/'Isaacsim_tactile_env'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ICRA2028_Stage3S_starter'))
env=None
manifest={'status':'running','completed_trials':[], 'rejected_initializations':[], 'errors':[]}
try:
    import torch
    from aloha_tactile_env import AlohaTactileEnv, AlohaTactileEnvCfg
    from mcf import ContactFeatureStream
    from reflex_policy import Controller
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import UsdPhysics, Gf
    class ContactAuditEnv(AlohaTactileEnv):
        def _spawn_basic_world(self,sim_utils):
            super()._spawn_basic_world(sim_utils)
            # Same lower catch floor in BOTH table / removed conditions. The
            # robot's URDF table supports the initial replay in both conditions.
            floor=sim_utils.get_current_stage().GetPrimAtPath('/World/defaultGroundPlane')
            floor.GetAttribute('xformOp:translate').Set(Gf.Vec3d(0.,0.,-.35))

        def _post_spawn_init(self,cfg,sim_utils,target_query_paths):
            from isaaclab.sim.schemas import activate_contact_sensors
            # All topology / instance edits must precede physics tensor views.
            self.set_support(args.support=='table')
            def paths(root):
                return [p.GetPath().pathString for p in sim_utils.get_all_matching_child_prims(
                    root,predicate=lambda p:p.HasAPI(UsdPhysics.RigidBodyAPI),traverse_instance_prims=False)]
            self.audit_sensors=[]; self.audit_paths=[]
            table=paths('/World/Robot/table')[0]
            floor=paths('/World/defaultGroundPlane')[0]
            object_paths=[paths('/World/Plug')[0],paths('/World/Socket')[0]]
            for i,path in enumerate(object_paths):
                activate_contact_sensors(path,threshold=0.)
                filters=[table,floor,object_paths[1-i]]
                self.audit_paths.append({'object':path,'filters':filters})
                self.audit_sensors.append(ContactSensor(ContactSensorCfg(
                    prim_path=path,update_period=0.,history_length=0,
                    filter_prim_paths_expr=filters,track_pose=True)))
            super()._post_spawn_init(cfg,sim_utils,target_query_paths)

        def step(self,action):
            result=super().step(action)
            for sensor in self.audit_sensors:
                sensor.update(dt=self._cfg.physics_dt,force_recompute=True)
            return result

        def set_support(self,enabled):
            from pxr import Usd
            # URDF table colliders are inside an instance. The generic schema
            # helper skips instance proxies, so expand this runtime instance
            # and verify the actual CollisionAPI properties directly.
            root=self._sim_utils.get_current_stage().GetPrimAtPath('/World/Robot/table/collisions')
            if not enabled and root.IsInstanceable():
                root.SetInstanceable(False)
            colliders=[p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
            if not colliders:
                raise RuntimeError('No table collision primitives found')
            if not enabled:
                for p in colliders:
                    UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
            self.support_collision_paths=[{'path':p.GetPath().pathString,
                'enabled':bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())} for p in colliders]
            if any(p['enabled']!=enabled for p in self.support_collision_paths):
                raise RuntimeError('Table support collision property verification failed')

    cfg=AlohaTactileEnvCfg(headless=args.headless,device=args.device,enable_camera=False,
                          plug_scale=args.plug_scale,normal_offset=args.normal_offset,fix_base=True)
    env=ContactAuditEnv(cfg,simulation_app=app)
    env.reset()
    obj=env._plug_obj if args.target=='plug' else env._socket_obj
    if obj is None:
        raise RuntimeError('Target absent')
    hold=targets[args.hold_step].copy()
    if args.fixture_npz:
        with np.load(args.fixture_npz,allow_pickle=False) as fixture:
            state=np.asarray(fixture['joint_pos'][0],dtype=np.float32)
            velocities=np.asarray(fixture['joint_vel'][0],dtype=np.float32)
            hold=fixture['action'][0].astype(np.float32)
            q=env._robot.data.default_joint_pos.clone(); dq=env._robot.data.default_joint_vel.clone()
            q[0,env._dataset_joint_ids]=torch.tensor(state,device=args.device)
            dq[0,env._dataset_joint_ids]=torch.tensor(velocities,device=args.device)
            env._robot.write_joint_state_to_sim(q,dq)
            for o,key in ((env._plug_obj,'plug_pose'),(env._socket_obj,'socket_pose')):
                root=o.data.default_root_state.clone()
                root[0,:7]=torch.tensor(fixture[key][0],device=args.device)
                root[0,7:]=0.
                o.write_root_state_to_sim(root); o.reset()
            env._robot.reset()
        # Shared saved fixture state for BOTH support conditions. These are
        # physical initial conditions, not load-dependent tactile or pose rules.
    else:
        for action in targets[:args.hold_step+1]:
            obs,*_=env.step(action)
    # Shared fixture is restored immediately; candidate placement / aperture
    # must occur before gravity can move an unsupported object out of reach.
    for _ in range(args.settle_steps if not args.fixture_npz else 0):
        obs,*_=env.step(hold)
    obs=env._get_obs()
    snapshot={ 'robot_root':env._robot.data.root_state_w.clone(),
               'robot_q':env._robot.data.joint_pos.clone(),
               'robot_dq':env._robot.data.joint_vel.clone(),
               'objects':[o.data.root_state_w.clone() if o else None for o in (env._plug_obj,env._socket_obj)]}
    material_default=obj.root_physx_view.get_material_properties().clone()
    limits=env._robot.data.joint_pos_limits[0,env._dataset_joint_ids].cpu().numpy()
    source_sha=subprocess.check_output(['git','-C',str(args.official_repo),'rev-parse','HEAD'],text=True).strip()
    provenance={'upstream_commit':source_sha,'source_file_sha256':{},
        'dataset_sha256':hashlib.sha256(args.dataset_npz.read_bytes()).hexdigest(),
        'device':args.device,'physics_dt':cfg.physics_dt,'target':args.target,
        'episode':args.episode,'hold_step':args.hold_step,'settle_steps':args.settle_steps,
        'plug_scale':args.plug_scale,'normal_offset':args.normal_offset,
        'reference_wxyz':args.reference_wxyz,'threshold':args.threshold,
        'robot_body_names':list(env._robot.body_names),'sensor_slots':[{'slot':int(slot),'link':str(link)} for slot,link in zip(env._sensor_slot_order,env._selected_links)],
        'units':{'tactile':'normalized SDF penetration proxy','mcf_fN':'sum of thresholded proxy responses',
                 'mcf_CoP':'grid coordinates [-1,1], NOT world x/y','mcf_A':'active cell fraction',
                 'force':'N, known external excitation submitted to PhysX',
                 'torque':'Nm about current COM, excludes contact reactions and gravity'},
        'boundary':'Rigid contact physics plus SDF readout. No deformable elastomer or calibrated six-axis F/T. SDF gain sweep is not physical stiffness.',
        'initial_hold_per_finger_sum':obs['tactile'].sum(axis=(1,2)).tolist(),
        'python':sys.version,'torch':torch.__version__}
    provenance.update(support=args.support,preload_mode=args.preload_mode,
        preload_target_proxy=args.preload_target,normal_target_n=args.normal_target_n,
        gate_proxy_min=args.gate_proxy_min,gate_normal_min_n=args.gate_normal_min_n,
        fixed_robot_base=True,catch_floor_center_z=-.35,
        object_contact_audit_paths=env.audit_paths,
        support_collision_paths=env.support_collision_paths,
        fixture_npz=str(args.fixture_npz) if args.fixture_npz else None,
        fixture_sha256=hashlib.sha256(args.fixture_npz.read_bytes()).hexdigest() if args.fixture_npz else None,
        controller=args.controller,checkpoint=str(args.checkpoint) if args.checkpoint else None,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        initial_close_rad=args.initial_close_rad,initial_target_offset_z_m=args.initial_target_offset_z_m,
        bringup_grid=args.bringup_grid,
        physics_material_properties=material_default.cpu().numpy().tolist(),
        preload_max_steps=args.preload_max_steps,preload_max_close_rad=args.preload_max_close_rad,
        boundary='Development contact intervention; table / removed both have fixed robot base. No real force calibration or deformable layer.')
    for path in [Path(__file__),Path(__file__).with_name('disturbance.py'),Path(__file__).with_name('reflex_policy.py'),Path(__file__).with_name('bounded_scaler.py'),
                 args.official_repo/'Isaacsim_tactile_env/aloha_tactile_env.py',
                 args.official_repo/'source/isaaclab/isaaclab/sensors/warp_sdf_tactile/warp_sdf_tactile_sensor.py']:
        provenance['source_file_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    initial_sensor_grids=[s.data.tactile_points_w[0,:,:3].cpu().numpy().copy().reshape(12,32,3)
                          for s in env._tactile_sensors]
    def current_com():
        # Derive from current link pose after teleports; upstream's lazy
        # root_com_pose cache is not invalidated by write_root_state_to_sim.
        pose=obj.data.root_link_pose_w[0].cpu().numpy()
        return pose[:3]+rotation_wxyz(pose[3:7])@obj.data.body_com_pos_b[0,0].cpu().numpy()
    # Independent dynamics check: move only the target clear of all contacts,
    # compare engine velocities with m*delta_v / duration. Gravity cancels in
    # paired loaded/zero trials. This checks actual PhysX force execution.
    mass=float(obj.root_physx_view.get_masses()[0,0])
    calibration_states={}
    inertia=obj.root_physx_view.get_inertias()[0].cpu().numpy().reshape(3,3)
    calibration_momentum={}
    calibration_loads=[('zero',[0.,0.,0.],[0.,0.,0.]),
        ('+Fx',[.2,0.,0.],[0.,0.,0.]),('-Fx',[-.2,0.,0.],[0.,0.,0.]),
        ('+Fy',[0.,.2,0.],[0.,0.,0.]),('-Fy',[0.,-.2,0.],[0.,0.,0.]),
        ('eccentric_Fx',[.2,0.,0.],[0.,.003,0.]),
        ('eccentric_Fy',[0.,.2,0.],[-.003,0.,0.])]
    for label,force,torque in calibration_loads:
        isolated=snapshot['objects'][0 if args.target=='plug' else 1].clone()
        isolated[:,:3]+=torch.tensor([2.,0.,2.],device=args.device)
        isolated[:,7:]=0.
        obj.write_root_state_to_sim(isolated)
        obj.reset()
        states=[]
        for _ in range(12):
            com=torch.tensor(current_com(),dtype=torch.float32,device=args.device)
            obj.root_physx_view.apply_forces_and_torques_at_position(
                force_data=torch.tensor([force],dtype=torch.float32,device=args.device),
                torque_data=torch.tensor([torque],dtype=torch.float32,device=args.device),position_data=com[None],
                indices=torch.tensor([0],dtype=torch.int32,device=args.device),is_global=True)
            env.step(hold)
            states.append(obj.data.root_com_vel_w[0].cpu().numpy().copy())
        calibration_states[label]=np.stack(states)
        r=rotation_wxyz(obj.data.root_com_quat_w[0].cpu().numpy())
        calibration_momentum[label]=r@inertia@r.T@states[-1][3:6]
    calibration={'mass_kg':mass,'duration_s':12*cfg.physics_dt,'forces':{}}
    for label,force in [('+Fx',[.2,0.,0.]),('-Fx',[-.2,0.,0.]),('+Fy',[0.,.2,0.]),('-Fy',[0.,-.2,0.])]:
        recovered=mass*(calibration_states[label][-1,:3]-calibration_states['zero'][-1,:3])/(12*cfg.physics_dt)
        calibration['forces'][label]={'requested_n':force,'engine_m_delta_v_over_dt_n':recovered.tolist(),
                                     'absolute_error_n':float(np.linalg.norm(recovered-force))}
    calibration['max_absolute_error_n']=max(v['absolute_error_n'] for v in calibration['forces'].values())
    calibration['torques']={}
    for label,force,torque in calibration_loads[-2:]:
        recovered=(calibration_momentum[label]-calibration_momentum['zero'])/(12*cfg.physics_dt)
        calibration['torques'][label]={'requested_nm':torque,'engine_delta_angular_momentum_over_dt_nm':recovered.tolist(),
                                     'absolute_error_nm':float(np.linalg.norm(recovered-torque))}
    calibration['max_absolute_error_nm']=max(v['absolute_error_nm'] for v in calibration['torques'].values())
    calibration['passed']=calibration['max_absolute_error_n']<.01 and calibration['max_absolute_error_nm']<.00015
    (args.out/'force_calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    np.savez_compressed(args.out/'force_calibration.npz',**calibration_states)
    print('FORCE_CALIBRATION',calibration,flush=True)
    if not calibration['passed']:
        raise RuntimeError('External force engine calibration failed')
    reference=rotation_wxyz(args.reference_wxyz)
    if args.reference_from_finger is not None:
        if args.load_frame!='reference':
            raise ValueError('reference-from-finger requires reference load frame')
        index=env._sensor_slot_order.index(args.reference_from_finger)
        grid=initial_sensor_grids[index].astype(np.float64)
        axis_x=grid[0,-1]-grid[0,0]
        axis_x=axis_x/np.linalg.norm(axis_x)
        axis_y=grid[-1,0]-grid[0,0]
        axis_y=axis_y-axis_x*np.dot(axis_x,axis_y)
        axis_y=axis_y/np.linalg.norm(axis_y)
        reference=np.column_stack([axis_x,axis_y,np.cross(axis_x,axis_y)])
    provenance['reference_rotation_world_from_load']=reference.tolist()
    provenance['reference_from_finger']=args.reference_from_finger
    reference_needs_fixture_update=bool(args.fixture_npz and args.reference_from_finger is not None)
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    for tr in trials:
        initial=initialization_plan.get(tr.trial_id,{'target_z_offset_m':args.initial_target_offset_z_m,
                                                    'initial_close_rad':args.initial_close_rad})
        if not 0<=initial['initial_close_rad']<=.004:
            raise ValueError('Initial close exceeds declared bound')
        # Restore the same physical state. Contact-solver warm-start caches are
        # not serialized; repeat discrepancy is explicitly measured downstream.
        env._robot.write_root_state_to_sim(snapshot['robot_root'])
        initial_q=snapshot['robot_q'].clone()
        finger_ids=[14,15] if args.target=='plug' else [6,7]
        for j,sign in zip(finger_ids,[-1,1]):
            initial_q[0,env._dataset_joint_ids[j]]+=sign*initial['initial_close_rad']
        env._robot.write_joint_state_to_sim(initial_q,torch.zeros_like(snapshot['robot_dq']))
        env._robot.reset()
        for o,state in zip((env._plug_obj,env._socket_obj),snapshot['objects']):
            if o:
                state=state.clone()
                if o is obj: state[:,2]+=initial['target_z_offset_m']
                state[:,7:]=0.
                o.write_root_state_to_sim(state)
                o.reset()
        for sensor in env._tactile_sensors:
            sensor.reset()
            sensor.cfg.stiffness=cfg.stiffness*tr.sdf_gain_scale
        material=material_default.clone()
        material[...,:2]*=tr.friction_scale
        obj.root_physx_view.set_material_properties(material,torch.tensor([0],dtype=torch.int32))
        action=hold.copy()
        finger_ids=[14,15] if args.target=='plug' else [6,7]
        action[finger_ids]+=np.array([1,-1])*tr.preload_delta_rad
        action[finger_ids]+=np.array([-1,1])*initial['initial_close_rad']
        if np.any(action<limits[:,0]) or np.any(action>limits[:,1]):
            raise ValueError(f'{tr.trial_id}: action exceeds joint limits')
        for _ in range(args.settle_steps):
            env.step(action)
        def contact_values():
            sums=env._get_obs()['tactile'].sum(axis=(1,2))
            normal=np.zeros(4)
            for slot,sensor in zip(env._sensor_slot_order,env._tactile_sensors):
                pts=sensor.data.tactile_points_w[0,:,:3].cpu().numpy().reshape(12,32,3)
                direction=np.cross(pts[0,-1]-pts[0,0],pts[-1,0]-pts[0,0])
                direction/=np.linalg.norm(direction)
                force=sensor._pose_sensors[0].data.net_forces_w[0,0].cpu().numpy()
                normal[slot]=abs(float(force@direction))
            return sums,normal
        slots=[2,3] if args.target=='plug' else [0,1]
        preload_history=[]; stable=0; close=0.; initial_gate=False
        for attempt in range(args.preload_max_steps if args.preload_mode!='open' else 1):
            sums,normal=contact_values()
            gate=bool(np.all(sums[slots]>=args.gate_proxy_min) and np.all(normal[slots]>=args.gate_normal_min_n))
            stable=stable+1 if gate else 0
            preload_history.append({'step':attempt,'proxy':sums.tolist(),'normal_n':normal.tolist(),
                'closing_delta_rad':close,'gate':gate})
            value=float(min(sums[slots])) if args.preload_mode=='proxy' else float(min(normal[slots]))
            target=args.preload_target if args.preload_mode=='proxy' else args.normal_target_n
            target_ready=abs(value-target)<=.1*target
            if args.preload_mode=='open' or (stable>=3 and target_ready):
                initial_gate=gate
                break
            step=.00005*(1 if value<target else -1)
            new_close=float(np.clip(close+step,-args.preload_max_close_rad,args.preload_max_close_rad))
            action[finger_ids]+=np.array([-1,1])*(new_close-close)
            close=new_close
            if np.any(action<limits[:,0]) or np.any(action>limits[:,1]):
                raise ValueError('Preload action exceeds joint limits')
            env.step(action)
        init={'trial_id':tr.trial_id,'mode':args.preload_mode,'gate_passed':initial_gate,'initial_configuration':initial,
              'feedback_target_reached':target_ready if args.preload_mode!='open' else None,
              'final_closing_delta_rad':close,'history':preload_history,
              'scope':'Target two fingers gate; four channels observe two separate objects.'}
        (args.out/f'{tr.trial_id}_initialization.json').write_text(json.dumps(init,indent=2),encoding='utf-8')
        if args.preload_mode!='open' and not (initial_gate and target_ready):
            manifest['rejected_initializations'].append(tr.trial_id)
            (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print('INITIALIZATION_REJECTED',tr.trial_id,init['final_closing_delta_rad'],flush=True)
            continue
        if reference_needs_fixture_update:
            index=env._sensor_slot_order.index(args.reference_from_finger)
            grid=env._tactile_sensors[index].data.tactile_points_w[0,:,:3].cpu().numpy().reshape(12,32,3).astype(float)
            x=grid[0,-1]-grid[0,0]; x/=np.linalg.norm(x)
            y=grid[-1,0]-grid[0,0]; y-=x*(x@y); y/=np.linalg.norm(y)
            reference=np.column_stack([x,y,np.cross(x,y)])
            provenance['reference_rotation_world_from_load']=reference.tolist()
            provenance['reference_frozen_after_trial_initialization']=tr.trial_id
            (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
            reference_needs_fixture_update=False
        stream=ContactFeatureStream(threshold=args.threshold)
        # Prime stream with adjacent pre-trial observation, avoiding a spurious
        # first-frame zero difference at onset.
        initial_obs=env._get_obs()
        stream.update(initial_obs['tactile'])
        initial_pose=(obj.data.root_link_pose_w[0].cpu().numpy().copy())
        input_mcf=stream.update(initial_obs['tactile'])[1]
        controller=Controller(args.controller,cfg.physics_dt,args.checkpoint)
        nominal_action=action.copy()
        previous_control_pose=initial_pose.copy()
        initial_robot=env._robot.data.body_link_pose_w[0].cpu().numpy().copy()
        keys=('tactile','scalar','state4','mcf','joint_pos','joint_vel','plug_pose','socket_pose','action',
              'object_state_w','robot_body_pose_w','sensor_points_w','sensor_pose_w','sensor_net_contact_force_world_n',
              'object_net_contact_normal_world_n','object_environment_contact_normal_world_n','force_world_n','torque_com_world_nm',
              'application_point_world_m','com_before_world_m','object_pose_before','applied_force_local_n',
              'applied_torque_local_nm','loading_gain','step_start_s','observation_time_s','global_step',
              'host_time_before_ns','host_time_after_ns','displacement_world_m','orientation_change_rad',
              'input_mcf_before','expert_delta_rad','controller_delta_rad','controller_inference_ns','expert_linear_speed_m_s','expert_angular_speed_rad_s')
        records={k:[] for k in keys}
        for i in range(args.trial_steps):
            if not app.is_running():
                raise RuntimeError('Simulation app closed')
            pose=obj.data.root_link_pose_w[0].cpu().numpy().copy()
            com=current_com().copy()
            t=i*cfg.physics_dt
            control_begin=time.perf_counter_ns()
            previous_angle=2*np.arccos(np.clip(abs(np.dot(pose[3:7],initial_pose[3:7])),0,1))
            # Differentiate realized link poses; solver velocities contain
            # persistent contact stabilization terms even at stationary poses.
            tangent_velocity=reference.T@(pose[:3]-previous_control_pose[:3])/cfg.physics_dt
            tangent_velocity[2]=0.
            qa=pose[3:7].astype(float); qa/=np.linalg.norm(qa)
            qb=previous_control_pose[3:7].astype(float); qb/=np.linalg.norm(qb)
            angular_rate=2*np.arccos(np.clip(abs(qa@qb),0,1))/cfg.physics_dt
            realized_velocity=np.r_[tangent_velocity,angular_rate,0.,0.]
            delta,teacher,speed,angular_speed=controller.step(input_mcf,realized_velocity,previous_angle)
            previous_control_pose=pose.copy()
            control_ns=time.perf_counter_ns()-control_begin
            action=nominal_action.copy()
            action[finger_ids]+=np.array([-1,1])*delta
            if np.any(action<limits[:,0]) or np.any(action>limits[:,1]): raise ValueError('Reflex joint limits')
            previous_input=input_mcf.copy()
            f,m,p=tr.wrench(t,pose,com,reference)
            r=rotation_wxyz(pose[3:7])
            local_f=r.T@f
            local_m=r.T@m
            host_before=time.perf_counter_ns()
            # Apply an equivalent wrench at the actual COM. PhysX receives force
            # and lever-arm moment before evolving the object/contact geometry.
            # No target pose or tactile map is set as a function of disturbance.
            obj.root_physx_view.apply_forces_and_torques_at_position(
                force_data=torch.tensor(f[None],dtype=torch.float32,device=args.device),
                torque_data=torch.tensor(m[None],dtype=torch.float32,device=args.device),
                position_data=torch.tensor(com[None],dtype=torch.float32,device=args.device),
                indices=torch.tensor([0],dtype=torch.int32,device=args.device),is_global=True)
            obs,*_=env.step(action)
            host_after=time.perf_counter_ns()
            scalar,mcf=stream.update(obs['tactile'])
            input_mcf=mcf.copy()
            current_pose=obj.data.root_link_pose_w[0].cpu().numpy().copy()
            angle=2*np.arccos(np.clip(abs(np.dot(current_pose[3:7],initial_pose[3:7])),0,1))
            points=np.zeros((4,12,32,3),dtype=np.float32)
            sensor_pose=np.zeros((4,7),dtype=np.float32)
            contact_force=np.zeros((4,3),dtype=np.float32)
            for slot,sensor in zip(env._sensor_slot_order,env._tactile_sensors):
                points[slot]=sensor.data.tactile_points_w[0,:,:3].cpu().numpy().reshape(12,32,3)
                # World grid points above are the authoritative patch geometry.
                # Store supporting link pose from the robot, not nonexistent
                # fields on WarpSdfTactileSensorData.
                pose_data=sensor._pose_sensors[0].data
                sensor_pose[slot,:3]=pose_data.pos_w[0,0].cpu().numpy()
                sensor_pose[slot,3:]=pose_data.quat_w[0,0].cpu().numpy()
                contact_force[slot]=pose_data.net_forces_w[0,0].cpu().numpy()
            values=dict(obs,input_mcf_before=previous_input,expert_delta_rad=teacher,controller_delta_rad=delta,controller_inference_ns=control_ns,expert_linear_speed_m_s=speed,expert_angular_speed_rad_s=angular_speed,scalar=scalar,state4=mcf[:,:4],mcf=mcf,action=action,
                object_net_contact_normal_world_n=np.stack([s.data.net_forces_w[0,0].cpu().numpy() for s in env.audit_sensors]),
                object_environment_contact_normal_world_n=np.stack([s.data.force_matrix_w[0,0].cpu().numpy() for s in env.audit_sensors]),
                object_state_w=obj.data.root_state_w[0].cpu().numpy(),
                robot_body_pose_w=env._robot.data.body_link_pose_w[0].cpu().numpy(),
                sensor_points_w=points,sensor_pose_w=sensor_pose,sensor_net_contact_force_world_n=contact_force,
                force_world_n=f,torque_com_world_nm=m,application_point_world_m=p,
                com_before_world_m=com,object_pose_before=pose,applied_force_local_n=local_f,
                applied_torque_local_nm=local_m,loading_gain=tr.gain(t),step_start_s=t,
                observation_time_s=(i+1)*cfg.physics_dt,global_step=env._step_count,
                host_time_before_ns=host_before,host_time_after_ns=host_after,
                displacement_world_m=current_pose[:3]-initial_pose[:3],orientation_change_rad=angle)
            for k in records:
                records[k].append(np.asarray(values[k]).copy())
        arrays={k:np.stack(v) for k,v in records.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError(f'{tr.trial_id}: nonfinite recording')
        assert arrays['tactile'].shape==(args.trial_steps,4,12,32)
        assert arrays['mcf'].shape==(args.trial_steps,4,8)
        np.savez_compressed(args.out/f'{tr.trial_id}.npz',**arrays,
                            initial_object_pose=initial_pose,initial_robot_body_pose=initial_robot,
                            initial_tactile=initial_obs['tactile'],
                            physics_dt=cfg.physics_dt)
        summary={'controller':args.controller,'trial':tr.metadata(),'initial_configuration':initial,'initial_gate_passed':initial_gate,'closing_delta_rad':close,
                 'initial_per_finger_sum':initial_obs['tactile'].sum(axis=(1,2)).tolist(),
                 'max_environment_normal_contact_n':np.linalg.norm(arrays['object_environment_contact_normal_world_n'],axis=-1).max(axis=0).tolist(),
                 'nonzero_frames_per_finger':np.any(arrays['tactile']>0,axis=(2,3)).sum(axis=0).tolist(),
                 'max_displacement_m':float(np.linalg.norm(arrays['displacement_world_m'],axis=1).max()),
                 'max_orientation_change_rad':float(arrays['orientation_change_rad'].max()),
                 'npz_sha256':hashlib.sha256((args.out/f'{tr.trial_id}.npz').read_bytes()).hexdigest()}
        (args.out/f'{tr.trial_id}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        manifest['completed_trials'].append(tr.trial_id)
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('E1S_TRIAL',tr.trial_id,'contact',summary['nonzero_frames_per_finger'],
              'displacement',summary['max_displacement_m'],flush=True)
    manifest['status']='completed'
except Exception as exc:
    manifest['status']='failed'
    manifest['errors'].append(repr(exc))
    raise
finally:
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if env:
        env.close()
    app.close()
