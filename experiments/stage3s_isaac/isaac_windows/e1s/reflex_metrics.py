"""Independent relative motion and contact effort from physical recordings."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from disturbance import rotation_wxyz


def read(root,tid):
    meta=json.loads((root/f'{tid}.json').read_text(encoding='utf-8'))
    path=root/f'{tid}.npz'; assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['npz_sha256']
    with np.load(path) as z: d={k:z[k] for k in z.files}
    initial_object=d['initial_object_pose']; relative=[]; initial=[]; force=[]
    # Initial sensor support poses are the initial robot body's actual link poses.
    # Map sensor links by provenance slots rather than assuming body indices.
    prov=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    # Sensor support poses have negligible motion before the first recorded step;
    # reconstruct the initial relation from stored initial object and initial
    # robot link pose. Body names are included in provenance by the driver.
    body_names=prov.get('robot_body_names')
    for finger in (2,3):
        link=next(x['link'] for x in prov['sensor_slots'] if x['slot']==finger).rsplit('/',1)[-1]
        if body_names:
            q=d['initial_robot_body_pose'][body_names.index(link)]
        else: q=d['sensor_pose_w'][0,finger]
        initial.append(rotation_wxyz(q[3:7]).T@(initial_object[:3]-q[:3]))
    for i in range(len(d['mcf'])):
        rel=[]; norms=[]
        for finger in (2,3):
            q=d['sensor_pose_w'][i,finger]
            rel.append(rotation_wxyz(q[3:7]).T@(d['object_state_w'][i,:3]-q[:3]))
            p=d['sensor_points_w'][i,finger]
            n=np.cross(p[0,-1]-p[0,0],p[-1,0]-p[0,0]); n/=np.linalg.norm(n)
            norms.append(abs(d['sensor_net_contact_force_world_n'][i,finger]@n))
        relative.append(rel); force.append(norms)
    slip=np.linalg.norm(np.array(relative)-np.array(initial)[None],axis=2).max(axis=1)*1000
    force=np.array(force); dt=float(d['physics_dt'])
    r0=rotation_wxyz(initial_object[3:7])
    d['tilt_change_rad']=np.array([np.arccos(np.clip((rotation_wxyz(q[3:7])@r0.T)[2,2],-1,1)) for q in d['object_state_w']])
    ending=meta['trial']['onset_s']+meta['trial']['duration_s']
    post=d['observation_time_s']>=ending+.2
    delta=d.get('controller_delta_rad',np.zeros(len(slip)))
    row=dict(id=tid,controller=meta.get('controller','nominal'),trial=meta['trial'],max_slip_mm=float(slip.max()),
        max_tilt_deg=float(np.rad2deg(d['tilt_change_rad']).max()),
        max_orientation_change_deg=float(np.rad2deg(d['orientation_change_rad']).max()),effort_normal_n_s=float(force.sum()*dt),
        peak_normal_n=float(force.sum(axis=1).max()),post_normal_n=float(force[post].sum(axis=1).mean()) if post.any() else None,
        saturation_fraction=float((delta>=.00297).mean()),max_delta_rad=float(delta.max()),
        contact_fraction=float(np.all(force>=.05,axis=1).mean()),
        environment_max_normal_n=float(np.linalg.norm(d['object_environment_contact_normal_world_n'][:,0],axis=-1).max()),
        control_p95_ms=float(np.quantile(d.get('controller_inference_ns',np.zeros(len(slip))),.95)/1e6),
        goal_slip_2mm=bool(slip.max()<=2.))
    row['dropped']=bool((slip[-1]>50 and np.all(force[-12:]<.05)) or
                        np.linalg.norm(d['object_environment_contact_normal_world_n'][-12:,0,1],axis=-1).max()>.0001)
    return row,d,slip,force.sum(axis=1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('roots',nargs='+',type=Path); a=p.parse_args()
    for root in a.roots:
        manifest=json.loads((root/'manifest.json').read_text()); assert manifest['status']=='completed'
        rows=[read(root,tid)[0] for tid in manifest['completed_trials']]
        (root/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(root.name,[(r['id'],round(r['max_slip_mm'],3),round(r['effort_normal_n_s'],3),round(r['post_normal_n'],3)) for r in rows])


if __name__=='__main__': main()
