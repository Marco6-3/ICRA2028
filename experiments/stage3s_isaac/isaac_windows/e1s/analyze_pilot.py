"""Trial-level E1-S diagnostics. No E2 temporal model or closed-loop claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from disturbance import Trial, rotation_wxyz
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ICRA2028_Stage3S_starter'))
from mcf import ContactFeatureStream


LABELS=['+Fx','-Fx','+Fy','-Fy']


def classify(x,y,train,test):
    mu=x[train].mean(axis=0)
    sd=x[train].std(axis=0)
    sd=np.where(sd>1e-8,sd,1.)
    xx=(x-mu)/sd
    centers=np.stack([xx[train & (y==i)].mean(axis=0) for i in range(4)])
    pred=((xx[test,None]-centers[None])**2).sum(axis=2).argmin(axis=1)
    truth=y[test]
    confusion=np.zeros((4,4),dtype=int)
    for a,b in zip(truth,pred):
        confusion[a,b]+=1
    correct=truth==pred
    rng=np.random.default_rng(20261013)
    boot=correct[rng.integers(0,len(correct),(2000,len(correct)))].mean(axis=1)
    return {'accuracy':float(correct.mean()),'trial_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'confusion_true_rows_pred_cols':confusion.tolist(),'labels':LABELS,
            'test_trial_count':int(test.sum()),'predictions':pred.tolist(),'truth':truth.tolist()}


def ridge_reconstruct(x,target,train,test):
    mu=x[train].mean(axis=0); sd=x[train].std(axis=0)
    sd=np.where(sd>1e-8,sd,1.)
    xx=(x-mu)/sd; yy=target[train]
    mean_y=yy.mean(axis=0); xx_train=xx[train]
    alpha=1.0
    weights=np.linalg.solve(xx_train@xx_train.T+alpha*np.eye(train.sum()),yy-mean_y)
    pred=xx[test]@xx_train.T@weights+mean_y
    rmse=np.sqrt(np.mean((pred-target[test])**2,axis=0))
    return {'targets':['excitation_Fx_N','excitation_Fy_N','excitation_Mx_Nm','excitation_My_Nm'],
            'rmse':rmse.tolist(),'ridge_alpha':alpha,'test_trial_count':int(test.sum()),
            'predictions':pred.tolist(),'truth':target[test].tolist(),
            'boundary':'Known external excitation about COM in frozen load reference, not total contact wrench.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':
        raise RuntimeError('Refusing to report an incomplete pilot as complete')
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    reference=np.asarray(provenance['reference_rotation_world_from_load'])
    summaries=[]; segments=[]
    traces={}
    validation={'shape':True,'finite':True,'causal_features':True,'synchronized_time':True,
                'external_wrench_math':True,'hashes':True}
    max_force_error=max_torque_error=max_feature_error=0.
    for trial_id in manifest['completed_trials']:
        meta=json.loads((root/f'{trial_id}.json').read_text(encoding='utf-8'))
        tr=Trial(**meta['trial'])
        raw_bytes=(root/f'{trial_id}.npz').read_bytes()
        validation['hashes'] &= hashlib.sha256(raw_bytes).hexdigest()==meta['npz_sha256']
        with np.load(root/f'{trial_id}.npz',allow_pickle=False) as data:
            d={k:data[k] for k in data.files}
        traces[trial_id]=d
        n=len(d['tactile']); dt=float(d['physics_dt'])
        validation['shape'] &= d['tactile'].shape==(n,4,12,32) and d['mcf'].shape==(n,4,8)
        validation['finite'] &= all(np.isfinite(v).all() for v in d.values())
        validation['synchronized_time'] &= (np.allclose(d['step_start_s'],np.arange(n)*dt)
            and np.allclose(d['observation_time_s'],(np.arange(n)+1)*dt)
            and np.all(np.diff(d['global_step'])==1)
            and np.all(d['host_time_after_ns']>=d['host_time_before_ns']))
        stream=ContactFeatureStream(threshold=provenance['threshold'])
        if 'initial_tactile' in d:
            stream.update(d['initial_tactile'])
        for i in range(n):
            scalar,mcf=stream.update(d['tactile'][i])
            if i>0 or 'initial_tactile' in d:
                max_feature_error=max(max_feature_error,float(np.max(abs(mcf-d['mcf'][i]))))
            f,m,p=tr.wrench(d['step_start_s'][i],d['object_pose_before'][i],d['com_before_world_m'][i],reference)
            max_force_error=max(max_force_error,float(np.max(abs(f-d['force_world_n'][i]))))
            max_torque_error=max(max_torque_error,float(np.max(abs(m-d['torque_com_world_nm'][i]))))
        # Loaded interval and baseline are fixed by the loading protocol, not
        # chosen by tactile magnitude, object motion or desired predictions.
        baseline=d['step_start_s']<tr.onset_s
        loaded=np.abs(d['loading_gain'])>0
        scalar_summary=d['scalar'][loaded].mean(axis=0)-d['scalar'][baseline].mean(axis=0)
        mcf_summary=d['mcf'][loaded].mean(axis=0)-d['mcf'][baseline].mean(axis=0)
        raw=d['tactile']; normalized=raw/np.maximum(raw.sum(axis=(2,3),keepdims=True),1e-12)
        raw_summary=normalized[loaded].mean(axis=0)-normalized[baseline].mean(axis=0)
        net=d.get('sensor_net_contact_force_world_n')
        relative_positions=np.zeros((n,4,3))
        relative_rotations=np.zeros((n,4,3,3))
        for i in range(n):
            object_rotation=rotation_wxyz(d['object_state_w'][i,3:7])
            for finger in range(4):
                sensor_rotation=rotation_wxyz(d['sensor_pose_w'][i,finger,3:7])
                relative_positions[i,finger]=sensor_rotation.T@(d['object_state_w'][i,:3]-d['sensor_pose_w'][i,finger,:3])
                relative_rotations[i,finger]=sensor_rotation.T@object_rotation
        relative_delta=relative_positions-relative_positions[:1]
        relative_angle=np.arccos(np.clip((np.einsum('tfij,fij->tf',relative_rotations,relative_rotations[0])-1)/2,-1,1))
        normal_projection=None
        if net is not None:
            points=d['sensor_points_w']
            a=points[:,:,0,-1]-points[:,:,0,0]
            b=points[:,:,-1,0]-points[:,:,0,0]
            normals=np.cross(a,b); normals/=np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-12)
            normal_projection=np.abs((net*normals).sum(axis=2)).mean(axis=0).tolist()
        summaries.append({'id':trial_id,'trial':tr.metadata(), 'baseline_proxy_sum':d['state4'][baseline,:,0].mean(axis=0).tolist(),
            'loaded_proxy_sum':d['state4'][loaded,:,0].mean(axis=0).tolist(),
            'contact_normal_projection_abs_mean_n':normal_projection,
            'max_object_link_relative_displacement_per_finger_m':np.linalg.norm(relative_delta,axis=2).max(axis=0).tolist(),
            'max_object_link_relative_orientation_change_per_finger_rad':relative_angle.max(axis=0).tolist(),
            'scalar':scalar_summary,'state4':mcf_summary[:,:4],'mcf':mcf_summary,'raw_normalized':raw_summary,
            'excitation_reference_FxFyMxMy':np.r_[d['force_world_n'][loaded].mean(axis=0)@reference,
                                                d['torque_com_world_nm'][loaded].mean(axis=0)@reference][[0,1,3,4]],
            'max_displacement_m':meta['max_displacement_m'],'max_orientation_change_rad':meta['max_orientation_change_rad'],
            'contact_fraction_per_finger':np.any(raw>0,axis=(2,3)).mean(axis=0).tolist(),
            'physics_plus_sensor_host_ms_p95':float(np.quantile((d['host_time_after_ns']-d['host_time_before_ns'])/1e6,.95))})
        if tr.waveform=='reverse' and tr.repeat==2:
            for sign in (1,-1):
                mask=d['loading_gain']==sign
                # Segment labels come from independent loader gain, not CoP.
                direction=tr.direction if sign==1 else ('-' if tr.direction[0]=='+' else '+')+tr.direction[1:]
                mc=d['mcf'][mask].mean(axis=0)-d['mcf'][baseline].mean(axis=0)
                segments.append({'id':trial_id,'direction':direction,
                    'scalar':d['scalar'][mask].mean(axis=0)-d['scalar'][baseline].mean(axis=0),
                    'state4':mc[:,:4],'mcf':mc,
                    'raw_normalized':normalized[mask].mean(axis=0)-normalized[baseline].mean(axis=0)})
    validation['causal_features']=max_feature_error<1e-6
    validation['external_wrench_math']=max_force_error<1e-8 and max_torque_error<1e-8
    primary=[s for s in summaries if s['trial']['direction']!='zero' and s['trial']['waveform']=='step'
             and s['trial']['preload_delta_rad']==0 and s['trial']['friction_scale']==1 and s['trial']['sdf_gain_scale']==1]
    y=np.array([LABELS.index(s['trial']['direction']) for s in primary])
    repeat=np.array([s['trial']['repeat'] for s in primary])
    scores={}; regression={}; generalization={}; reverse_scores={}; train=repeat<2; test=repeat==2
    if test.any() and train.any() and len(set(y[train]))==4:
        for key in ('scalar','state4','mcf','raw_normalized'):
            x=np.stack([s[key].reshape(-1) for s in primary])
            scores[key+'_four_fingers']=classify(x,y,train,test)
            scores[key+'_four_fingers']['input_dimension']=int(x.shape[1])
            target=np.stack([s['excitation_reference_FxFyMxMy'] for s in primary])
            regression[key]=ridge_reconstruct(x,target,train,test)
            for finger in range(4):
                x=np.stack([s[key][finger].reshape(-1) for s in primary])
                scores[f'{key}_finger{finger}']=classify(x,y,train,test)
                scores[f'{key}_finger{finger}']['input_dimension']=int(x.shape[1])
        training=[s for s,keep in zip(primary,train) if keep]
        for scenario,subset in [
            ('preload',[s for s in summaries if s['trial']['repeat']==2 and s['trial']['preload_delta_rad']!=0]),
            ('friction',[s for s in summaries if s['trial']['repeat']==2 and s['trial']['friction_scale']!=1]),
            ('sdf_gain',[s for s in summaries if s['trial']['repeat']==2 and s['trial']['sdf_gain_scale']!=1]),
            ('ramp',[s for s in summaries if s['trial']['repeat']==2 and s['trial']['waveform']=='ramp'])]:
            if not subset:
                continue
            combined=training+subset
            yy=np.array([LABELS.index(s['trial']['direction']) for s in combined])
            trmask=np.arange(len(combined))<len(training)
            generalization[scenario]={}
            for key in ('scalar','state4','mcf','raw_normalized'):
                xx=np.stack([s[key].reshape(-1) for s in combined])
                generalization[scenario][key]=classify(xx,yy,trmask,~trmask)
                generalization[scenario][key]['test_ids']=[s['id'] for s in subset]
        if segments:
            combined=training+segments
            yy=np.array([LABELS.index(s['trial']['direction']) if 'trial' in s else LABELS.index(s['direction']) for s in combined])
            trmask=np.arange(len(combined))<len(training)
            for key in ('scalar','state4','mcf','raw_normalized'):
                xx=np.stack([s[key].reshape(-1) for s in combined])
                sc=classify(xx,yy,trmask,~trmask)
                # Two intervals from each reversal trial are not independent.
                correct=(np.asarray(sc['truth'])==np.asarray(sc['predictions'])).reshape(-1,2).mean(axis=1)
                rng=np.random.default_rng(20261013)
                boot=correct[rng.integers(0,len(correct),(2000,len(correct)))].mean(axis=1)
                sc['trial_bootstrap_ci95']=np.quantile(boot,[.025,.975]).tolist()
                sc['test_segment_count']=sc.pop('test_trial_count')
                sc['test_trial_count']=len(correct)
                sc['test_ids']=[s['id'] for s in segments]
                reverse_scores[key]=sc
    groups={}
    for s in summaries:
        condition=s['id'].split('_r')[0]
        groups.setdefault(condition,[]).append(s)
    repeat_diagnostics=[]
    for condition,ss in groups.items():
        if len(ss)<2:
            continue
        ref=traces[ss[0]['id']]
        repeat_diagnostics.append({'condition':condition,'repeats':len(ss),
            'max_force_difference_n':max(float(np.max(abs(traces[s['id']]['force_world_n']-ref['force_world_n']))) for s in ss[1:]),
            'max_torque_difference_nm':max(float(np.max(abs(traces[s['id']]['torque_com_world_nm']-ref['torque_com_world_nm']))) for s in ss[1:]),
            'max_object_position_difference_m':max(float(np.max(np.linalg.norm(traces[s['id']]['object_state_w'][:,:3]-ref['object_state_w'][:,:3],axis=1))) for s in ss[1:]),
            'raw_map_rmse':max(float(np.sqrt(np.mean((traces[s['id']]['tactile']-ref['tactile'])**2))) for s in ss[1:])})
    failure_cases=[]
    for key,score in scores.items():
        ids=[s['id'] for s,keep in zip(primary,test) if keep]
        for trial_id,a,b in zip(ids,score['truth'],score['predictions']):
            if a!=b:
                failure_cases.append({'model':key,'trial':trial_id,'truth':LABELS[a],'prediction':LABELS[b]})
    for scenario,models in {**generalization,'reverse':reverse_scores}.items():
        for key,score in models.items():
            for trial_id,a,b in zip(score['test_ids'],score['truth'],score['predictions']):
                if a!=b:
                    failure_cases.append({'model':key,'scenario':scenario,'trial':trial_id,'truth':LABELS[a],'prediction':LABELS[b]})
    zero_by_repeat={s['trial']['repeat']:traces[s['id']] for s in summaries if s['trial']['direction']=='zero'}
    matched_zero=[]
    for s in primary:
        zero=zero_by_repeat.get(s['trial']['repeat'])
        if zero is not None:
            d=traces[s['id']]
            matched_zero.append({'trial':s['id'],'raw_vs_zero_rmse':float(np.sqrt(np.mean((d['tactile']-zero['tactile'])**2))),
                'max_position_difference_vs_zero_m':float(np.linalg.norm(d['object_state_w'][:,:3]-zero['object_state_w'][:,:3],axis=1).max())})
    report={'status':'validated' if all(validation.values()) else 'validation_failed',
        'trials':len(summaries),'frames':sum(len(d['tactile']) for d in traces.values()),
        'validation':{k:bool(v) for k,v in validation.items()},'max_feature_error':max_feature_error,
        'max_force_math_error_n':max_force_error,'max_torque_math_error_nm':max_torque_error,
        'force_calibration':json.loads((root/'force_calibration.json').read_text(encoding='utf-8')),
        'split':'development repeat 0,1; diagnostic repeat 2. Same simulator session, not independent sessions.',
        'classifier':'training-only standardized nearest class centroid; fixed whole loaded interval minus pre-onset mean',
        'statistical_boundary':'Exploratory trial bootstrap only. Deterministic state restores do not establish real-world repeatability or E2 interaction.',
        'scores':scores,'excitation_reconstruction':regression,'held_out_factor_diagnostics':generalization,
        'reversal_segment_diagnostics':reverse_scores,'matched_zero_diagnostics':matched_zero,
        'repeat_diagnostics':repeat_diagnostics,'failure_cases':failure_cases,
        'trial_metrics':[{k:v for k,v in s.items() if k not in ('scalar','state4','mcf','raw_normalized','excitation_reference_FxFyMxMy')} for s in summaries]}
    (root/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# E1-S pilot 数据核验与方向诊断','',f"采集 {report['trials']} 个试次、{report['frames']} 个物理步。核验状态：`{report['status']}`。",
        '', '每指 Raw 为 12×32，MCF 为 8D；四指联合诊断分别为 1536D 和 32D。',
        '', '本次是开发 pilot。重复 0/1 用于拟合均值与标准化，重复 2 用于诊断；没有按帧随机划分。重复来自同一仿真会话与状态恢复，不能视作真实独立实验。',
        '', '## 加载与记录核验','',f"力的无接触动态校验最大误差 {report['force_calibration']['max_absolute_error_n']:.6g} N；偏心力矩校验最大误差 {report['force_calibration']['max_absolute_error_nm']:.6g} Nm。",
        '', '外力/力矩为施加到 PhysX 的已知激励，独立于触觉特征；并非从法向阵列反推出的真值。接触合力另外记录自 PhysX 接触传感器，不能把它与外部激励或完整六轴 F/T 等同。',
        '', 'analysis.json 另外保存每指支撑 link 坐标中的物体相对位置和朝向变化，基准为录制首帧。它是独立引擎运动学读出，不是接触界面的微滑移真值。',
        '',f"因果 MCF 重算最大误差 {max_feature_error:.6g}；同步时间、形状、有限值与逐文件 SHA-256 的核验结果见 analysis.json。",
        '', '## 固定条件方向混淆','', '| 表示（四指联合） | 输入维数 | 方向准确率 | 试次 bootstrap 95% 区间 |','|---|---:|---:|---|']
    for key in ('scalar','state4','mcf','raw_normalized'):
        s=scores.get(key+'_four_fingers')
        if s:
            lines.append(f"| {key} | {s['input_dimension']} | {s['accuracy']:.1%} | {s['trial_bootstrap_ci95']} |")
    lines+=['', 'Raw 先按每指总强度归一化，检查空间分布信息。此处 MCF 使用整个受载区间的均值差；不是 GRU 时序对照，也没有检验 representation × memory 交互。混淆矩阵、逐指结果和全部误判在 analysis.json 中。',
        '', '## 留出因素和反向时段诊断','', '| 条件 | Scalar | 静态状态 | MCF | 归一化 Raw |','|---|---:|---:|---:|---:|']
    for scenario,models in {**generalization,'reverse':reverse_scores}.items():
        if not models:
            continue
        lines.append('| '+scenario+' | '+' | '.join(f"{models[key]['accuracy']:.1%}" for key in ('scalar','state4','mcf','raw_normalized'))+' |')
    lines+=['', '上述模型只在基准阶跃条件重复 0/1 拟合，留出因素不重新训练。反向按两个独立加载符号时段诊断，但 bootstrap 以原 trial 为单位，不能把两个时段当两次独立重复。范围仍限于同一仿真会话。',
        '', '## 独立激励重构误差','', '| 表示 | Fx RMSE (N) | Fy RMSE (N) | Mx RMSE (Nm) | My RMSE (Nm) |','|---|---:|---:|---:|---:|']
    for key,result in regression.items():
        lines.append('| '+key+' | '+' | '.join(f'{v:.6g}' for v in result['rmse'])+' |')
    lines+=['', '固定 ridge α=1，只使用训练 trial 标准化。目标是参考系中已知外部激励的受载均值，既不是接触总 wrench，也不是 Stage 2P 当前状态时序重构；不据此判断 E2 已完成。',
        '', '## 重复性与失败边界','']
    if repeat_diagnostics:
        lines.append(f"同条件最大加载差：{max(x['max_force_difference_n'] for x in repeat_diagnostics):.6g} N / {max(x['max_torque_difference_nm'] for x in repeat_diagnostics):.6g} Nm。")
        lines.append(f"恢复状态后的最大物体位置差 {max(x['max_object_position_difference_m'] for x in repeat_diagnostics):.6g} m；Raw 重复 RMSE 最大值 {max(x['raw_map_rmse'] for x in repeat_diagnostics):.6g}。接触求解器 warm-start 缓存未序列化，因此不保证逐位一致。")
        if max(x['raw_map_rmse'] for x in repeat_diagnostics)==0:
            lines.append('本次恢复重复在保存精度下完全一致，诊断测试集包含训练轨迹的精确重复。这核验工程可重放性，但方向准确率和 bootstrap 区间不能用作独立泛化、统计显著优势或真实重复性证据。')
    support_text=('本组保留原作者机器人/物体几何，固定底座并禁用 table 碰撞、下移接物地面。环境接触是否为零及夹持失败见 CONTACT_REPORT.md。该干预不模拟柔性层刚度或迟滞。'
                  if provenance.get('support')=='removed' else
                  '该环境为原作者装配场景，包含桌面、另一个物体及夹爪支撑。不能把方向分类结果直接归因于柔性层的法向—切向耦合。SDF 增益变化仅改变读出，未实现可变材料刚度或迟滞。')
    lines+=['',f"共保存 {len(failure_cases)} 条逐表示误判记录，包含左右非目标指尖的结果；没有删去弱响应或接触变化试次。",
        '', support_text,
        '', '此报告不判定 MCF 最小充分性、GRU 优势或闭环控制贡献。完整 E1 仍需真实 FlexiTac 标定与独立会话；E2/E3 按 PROTOCOL.md 保留。', '']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(root,traces,report,primary)
    print(json.dumps({k:report[k] for k in ('status','trials','frames','validation')},indent=2))
    print('JOINT_SCORES', {k:v['accuracy'] for k,v in scores.items() if k.endswith('four_fingers')})
    if report['status']!='validated':
        raise RuntimeError('Data validation failed')


def plot(root,traces,report,primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not report['scores']:
        frames=[frame for frame in ('world','object','reference') if any(t['trial']['frame']==frame for t in report['trial_metrics'])]
        fig,axes=plt.subplots(len(frames),2,figsize=(12,3*len(frames)),squeeze=False)
        for row,frame in enumerate(frames):
            for trial in report['trial_metrics']:
                if trial['trial']['frame']!=frame or trial['trial']['repeat']!=0:
                    continue
                d=traces[trial['id']]
                for col,key in enumerate(('force_world_n','torque_com_world_nm')):
                    axes[row,col].plot(d['step_start_s'],d[key][:,0],label=trial['trial']['direction']+' world x')
                    axes[row,col].plot(d['step_start_s'],d[key][:,1],linestyle='--',label=trial['trial']['direction']+' world y')
                    axes[row,col].set_title(frame+': '+key)
                    axes[row,col].set_xlabel('Simulation time (s)')
            for a in axes[row]:
                a.legend(fontsize=7,ncol=2)
        fig.tight_layout(); fig.savefig(root/'diagnostics.png',dpi=150); plt.close(fig)
        return
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for col,key in enumerate(('scalar','state4','mcf','raw_normalized')):
        score=report['scores'].get(key+'_four_fingers')
        if score:
            a=axes[0,col]
            a.imshow(score['confusion_true_rows_pred_cols'],vmin=0,cmap='Blues')
            a.set_xticks(range(4),LABELS); a.set_yticks(range(4),LABELS)
            a.set_title(f'{key}: {score["accuracy"]:.1%}'); a.set_xlabel('Predicted'); a.set_ylabel('True')
            for i,row in enumerate(score['confusion_true_rows_pred_cols']):
                for j,value in enumerate(row):
                    a.text(j,i,str(value),ha='center',va='center')
        a=axes[1,col]
        zero_id=next((tid for tid in traces if tid=='c000_r02'),None)
        if zero_id:
            d=traces[zero_id]
            a.plot(d['observation_time_s'],d['state4'][:,2,col],color='black',linestyle='--',label='Zero load')
        for label in LABELS:
            selected=[s for s in primary if s['trial']['direction']==label and s['trial']['point_m']==[0.,0.,0.] and s['trial']['repeat']==2]
            if selected:
                d=traces[selected[0]['id']]
                a.plot(d['observation_time_s'],d['state4'][:,2,col],label=label)
        a.set_title(f'Finger 2: {("proxy sum","CoP grid x","CoP grid y","area fraction")[col]}')
        a.set_xlabel('Simulation time (s)'); a.legend(fontsize=8)
    fig.suptitle('E1-S development pilot: one session, no deformable elastomer / control claim')
    fig.tight_layout(); fig.savefig(root/'diagnostics.png',dpi=150); plt.close(fig)


if __name__=='__main__':
    main()
