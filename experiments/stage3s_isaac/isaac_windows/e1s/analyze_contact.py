"""Physical contact validity, initialization failures and candidate feasibility."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from disturbance import rotation_wxyz


def analyze_directory(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed': raise RuntimeError('Incomplete directory: '+str(root))
    rows=[]
    for tid in manifest['completed_trials']:
        meta=json.loads((root/f'{tid}.json').read_text(encoding='utf-8'))
        sha=hashlib.sha256((root/f'{tid}.npz').read_bytes()).hexdigest()
        if sha!=meta['npz_sha256']: raise RuntimeError('Hash mismatch')
        with np.load(root/f'{tid}.npz',allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
        if not all(np.isfinite(v).all() for v in d.values()): raise RuntimeError('Nonfinite data')
        slots=[2,3] if provenance['target']=='plug' else [0,1]
        normal=[]; relative=[]
        for i in range(len(d['tactile'])):
            values=[]; rel=[]
            for finger in slots:
                p=d['sensor_points_w'][i,finger]
                n=np.cross(p[0,-1]-p[0,0],p[-1,0]-p[0,0]); n/=np.linalg.norm(n)
                values.append(abs(d['sensor_net_contact_force_world_n'][i,finger]@n))
                q=d['sensor_pose_w'][i,finger]
                rel.append(rotation_wxyz(q[3:7]).T@(d['object_state_w'][i,:3]-q[:3]))
            normal.append(values); relative.append(rel)
        normal=np.asarray(normal); relative=np.asarray(relative)
        sums=d['state4'][:,:,0][:,slots]
        contact_fraction=float(np.all((normal>=.05)&(sums>=2.),axis=1).mean())
        index=0 if provenance['target']=='plug' else 1
        environment=d['object_environment_contact_normal_world_n'][:,index]
        support_max=np.linalg.norm(environment,axis=-1).max(axis=0)
        relative_max=float(np.linalg.norm(relative-relative[:1],axis=2).max())
        feasible=contact_fraction>=.95 and support_max.max()<.0001 and relative_max<=.005
        rows.append({'id':tid,'initial':meta.get('initial_configuration'),
            'hash_verified':True,'valid_target_contact_fraction':contact_fraction,
            'support_max_n_table_floor_other':support_max.tolist(),
            'max_object_link_relative_displacement_m':relative_max,
            'max_world_displacement_m':meta['max_displacement_m'],
            'zero_load_feasible':bool(feasible),'initial_gate_passed':meta.get('initial_gate_passed'),
            'normal_min_mean_n':float(normal.min(axis=1).mean())})
    initial=[]
    for path in root.glob('*_initialization.json'):
        init=json.loads(path.read_text(encoding='utf-8')); history=init['history']
        initial.append({'id':init['trial_id'],'rejected':init['trial_id'] in manifest.get('rejected_initializations',[]),
            'ever_contact_gate':any(h['gate'] for h in history),
            'feedback_target_reached':init['feedback_target_reached'],
            'final_close_rad':init['final_closing_delta_rad'],'final_proxy':history[-1]['proxy'],
            'final_normal_n':history[-1]['normal_n']})
    return {'name':root.name,'support':provenance['support'],'preload_mode':provenance['preload_mode'],
        'loaded':len(rows),'rejected_initializations':len(manifest.get('rejected_initializations',[])),
        'feasible_trial_count':sum(r['zero_load_feasible'] for r in rows),'rows':rows,'initializations':initial,
        'source_sha256':provenance['source_file_sha256'],'fixture_sha256':provenance.get('fixture_sha256'),
        'collision_paths':provenance.get('support_collision_paths')}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    args=parser.parse_args(); root=args.root
    paths=[root] if (root/'manifest.json').exists() else sorted(p.parent for p in root.glob('*/manifest.json'))
    results=[analyze_directory(p) for p in paths]
    output={'status':'completed','results':results,
            'criteria':'contact_fraction>=.95, max table/floor/other normal contact<.0001 N, max link relative displacement<=.005 m; no formal E1/E3 success claim'}
    (root/'contact_analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    lines=['# 支撑/预紧与夹持初态检查','', '| 组 | 加载试次 | 初始化拒绝 | 满足初态标准的记录 |','|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['name']} | {r['loaded']} | {r['rejected_initializations']} | {r['feasible_trial_count']} |")
    lines+=['', '初态标准同时要求目标两指的 SDF/PhysX 接触、零环境支撑法向力、相对位移≤5 mm。该标准用于开发夹持初态；加载后的满足数不是正式 Anti-Tilt 控制成功率。',
        '', 'SDF proxy 反馈可能达到接触门限但达不到目标响应，在闭合上限处仍被拒绝；这与“完全无接触”不同，分别记录 ever_contact_gate 和 feedback_target_reached。',
        '', '过滤接触力只包含相应法向接触通道，不是完整含摩擦的六轴 wrench；另保存 table collider 关闭状态。未接触的试次或掉到接物地面不计为纯夹持。',
        '', '改变初始放置与闭合量是物理场景初始化，没有写入任何预期压力/CoP 响应。数据、哈希、逐试次失败和全部门限历史均保留。', '']
    feasible=[]
    for r in results:
        for row in r['rows']:
            if row['zero_load_feasible']:
                feasible.append(row)
    if len(results)==1 and feasible:
        lines+=['## 可行候选','']
        for row in feasible:
            lines.append(f"- {row['id']}: {row['initial']}, 目标接触比例 {row['valid_target_contact_fraction']:.1%}，环境接触最大值 {max(row['support_max_n_table_floor_other']):.6g} N，相对位移 {row['max_object_link_relative_displacement_m']:.6g} m。")
    (root/'CONTACT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print([(r['name'],r['loaded'],r['rejected_initializations'],r['feasible_trial_count']) for r in results])


if __name__=='__main__': main()
