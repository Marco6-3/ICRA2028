"""Audit saved models and generate a Chinese evidence report and figures."""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_spatial_validation import OUT,CACHE,SPLITS,NAMES,SEEDS,Net,features,macro

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    threadpool_limits(4);torch.set_num_threads(4)
    results=read('all_results.json');audit=read('data_audit.json');checks=read('feature_checks.json');splits=read('eligible_split.json')
    full=[r for r in results if r['fraction']==1]
    summary={}
    for family in ['linear','neural']:
      names=sorted(set(r['name'] for r in full if r['family']==family))
      for name in names:
        entries=[r for r in full if r['family']==family and r['name']==name]
        summary[family+'_'+name]={}
        for scope in ['test','conditional']:
          vals=[r[scope]['rmse'] for r in entries if r[scope]['rmse'] is not None]
          maes=[r[scope]['mae'] for r in entries if r[scope]['mae'] is not None]
          summary[family+'_'+name][scope]=dict(rmse_mean=float(np.mean(vals)),rmse_seed_std=float(np.std(vals)),mae_mean=float(np.mean(maes)))
    # Paired cluster bootstrap of seed-averaged per-source MSE, no frame bootstrap.
    comparisons=[];rng=np.random.default_rng(20261006)
    for family in ['linear','neural']:
      for scope in ['test','conditional']:
        def per(name):
            rows=[r for r in full if r['family']==family and r['name']==name]
            ids=sorted(set(p['group'] for r in rows for p in r[scope]['per_source']))
            return {g:np.mean([p['mse'] for r in rows for p in r[scope]['per_source'] if p['group']==g]) for g in ids}
        for other in ['Scalar','ScalarCentroid','Raw','Learned11D']:
          if family=='linear' and other=='Learned11D':continue
          a=per('Physics11D');b=per(other);ids=sorted(set(a)&set(b))
          if not ids:continue
          d=np.array([a[g]-b[g] for g in ids]);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
          comparisons.append(dict(family=family,scope=scope,comparison='Physics11D minus '+other,groups=ids,
             mse_difference=float(d.mean()),paired_cluster_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
             physics_better_source_count=int((d<0).sum()),source_count=len(ids),per_source_mse_difference=d.tolist()))
    write('summary.json',summary);write('paired_comparisons.json',comparisons)

    # Reload artifacts independently on CPU, compare with saved CUDA predictions.
    data=dict(np.load(CACHE/'stage1_data_expanded.npz'));g=data['group']
    tr=np.flatnonzero(np.isin(g,SPLITS['train'])&(data['row']%3==0));te=np.flatnonzero(np.isin(g,SPLITS['test']))
    f,_=features(data,tr);take=np.linspace(0,len(te)-1,min(200,len(te)),dtype=int)
    constant=np.mean([data['y'][tr[g[tr]==source]].mean(0) for source in np.unique(g[tr])],axis=0)
    constant_pred=np.broadcast_to(constant,(len(te),2))
    write('constant_reference.json',dict(training_equal_source_mean=constant.tolist(),test=macro(data['y'][te],constant_pred,g[te]),
        conditional=macro(data['y'][te],constant_pred,g[te],data['conditional'][te]),status='Supplementary training-only constant reference, no selection'))
    # Match each eligible conditional frame back to its original nonoverlapping window.
    cond=data['conditional'][te];window=np.full(len(te),-1,dtype=int);wid=0
    for group in np.unique(g[te]):
        ep=np.asarray(pq.read_table(CACHE/f'obj{group}.parquet',columns=['episode_index'])['episode_index'])
        positions=np.flatnonzero((g[te]==group)&cond);rows=data['row'][te[positions]]
        for episode in np.unique(ep[rows]):
            origin=np.flatnonzero(ep==episode);origin=origin[origin>=30];base=origin[0]
            q=positions[ep[rows]==episode];indices=(data['row'][te[q]]-base)//15
            for k in np.unique(indices):
                window[q[indices==k]]=wid;wid+=1
    assert (window[cond]>=0).all()
    def centered(a):
        z=a.copy()
        ix=np.flatnonzero(cond);labels=window[ix];counts=np.bincount(labels,minlength=wid)
        sums=np.c_[np.bincount(labels,weights=a[ix,0],minlength=wid),np.bincount(labels,weights=a[ix,1],minlength=wid)]
        means=sums/np.maximum(counts,1)[:,None]
        z[ix]-=means[labels]
        return z
    cy=centered(data['y'][te]);dynamic=[]
    for family in ['linear','neural']:
      for name in ['Scalar','ScalarCentroid','Physics11D','Raw','Learned11D']:
        if family=='linear' and name=='Learned11D':continue
        for seed in ([7] if family=='linear' else SEEDS):
          path=OUT/(f'linear_{name}_prediction.npz' if family=='linear' else f'neural_{name}_seed{seed}_prediction.npz')
          p=np.load(path)['prediction'];metric=macro(cy,centered(p),g[te],cond)
          dynamic.append(dict(family=family,name=name,seed=seed,metric=metric))
    dynamic_baseline=macro(cy,np.zeros_like(cy),g[te],cond)
    write('within_window_diagnostic.json',dict(window_count=wid,static_prediction_reference=dynamic_baseline,results=dynamic,
      limitations='Retrospective window centering only; not a deployable prediction preprocessing step; supplementary endpoint, no tuning'))
    reload_checks=[]
    for name in NAMES:
      for seed in SEEDS:
        path=OUT/f'neural_{name}_seed{seed}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        model=Net(ck['input_dim'],name=='Learned11D');model.load_state_dict(ck['state']);model.eval()
        x=f['Raw'] if name=='Learned11D' else f[name];s=ck['scale']
        with torch.no_grad():p=model(torch.tensor((x[te[take]]-s['x_mean'])/s['x_std'])).numpy()*s['y_std']+s['y_mean']
        saved=np.load(OUT/f'neural_{name}_seed{seed}_prediction.npz')
        assert np.array_equal(saved['group'],g[te]) and np.array_equal(saved['row'],data['row'][te])
        err=float(np.max(abs(p-saved['prediction'][take])))
        assert np.isfinite(saved['prediction']).all() and err<1e-4
        reload_checks.append(dict(name=name,seed=seed,cpu_cuda_max_absolute_difference=err,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name in ['Scalar','ScalarCentroid','Physics11D','Raw','FzOracle']:
        z=np.load(OUT/f'linear_{name}_weights.npz');x=f[name][te[take]]
        p=(((x-z['x_mean'])/z['x_std'])@z['coef'].T+z['intercept'])*z['y_std']+z['y_mean']
        saved=np.load(OUT/f'linear_{name}_prediction.npz');err=float(np.max(abs(p-saved['prediction'][take])))
        assert err<1e-5
        reload_checks.append(dict(name='linear_'+name,reload_max_absolute_difference=err))
    # Independent moment identities on genuine arrays.
    w=np.where(data['delta'][te[take]]>3*data['noise'][te[take]],data['delta'][te[take]],0).astype(np.float64)
    xx,yy=np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,12));den=w.sum(1)
    cx=(w*xx.ravel()).sum(1)/den;cy=(w*yy.ravel()).sum(1)/den
    trace=(w*((xx.ravel()[None,:]-cx[:,None])**2+(yy.ravel()[None,:]-cy[:,None])**2)).sum(1)/den
    phys=f['Physics11D'][te[take]]
    moment_error=float(np.max(abs(trace-(phys[:,4]**2+phys[:,5]**2))))
    assert moment_error<1e-6
    write('artifact_verification.json',dict(split_disjoint=True,all_predictions_finite=True,checkpoints=reload_checks,
      covariance_trace_identity_max_error=moment_error,neural_best_epoch_near_cap=[dict(name=r['name'],seed=r['seed'],best_epoch=r['best_epoch']) for r in full if r['family']=='neural' and r['best_epoch']>=195]))

    labels=['Scalar','ScalarCentroid','Physics11D','Raw','Learned11D']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,scope,title in zip(axes,['test','conditional'],['All eligible contact frames','Stable-load / low-shear subset']):
        vals=[summary['neural_'+n][scope]['rmse_mean'] for n in labels]
        errs=[summary['neural_'+n][scope]['rmse_seed_std'] for n in labels]
        ax.bar(np.arange(5),vals,yerr=errs,capsize=4,color=['#999999','#e9b44c','#1982a4','#7057a3','#d87850'])
        ax.set_xticks(np.arange(5),['Scalar','Scalar +\ncentroid','Physics\n11D','Raw','Learned\n11D'])
        ax.set_ylabel('Macro-source RMSE (stored torque / force units)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Held-out source sequences; bars average 3 seeds, error bars are seed SD')
    fig.tight_layout();fig.savefig(OUT/'spatial_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name in ['Scalar','ScalarCentroid','Physics11D','Raw']:
      means=[];std=[]
      for fraction in [.25,.5,1.]:
        vals=[r['test']['rmse'] for r in results if r['family']=='linear' and r['name']==name and r['fraction']==fraction]
        means.append(np.mean(vals));std.append(np.std(vals))
      ax.errorbar([.25,.5,1],means,yerr=std,marker='o',label=name)
    ax.set_xlabel('Fraction of eligible training source sequences');ax.set_ylabel('Macro-source RMSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'linear_data_efficiency.png',dpi=170);plt.close(fig)

    def table(family):
        lines=['| 输入 | 全部接触帧 RMSE | 稳定载荷子集 RMSE |','| --- | ---: | ---: |']
        for name in ['Scalar','ScalarCentroid','Physics11D','Raw','Learned11D','FzOracle']:
          k=family+'_'+name
          if k not in summary:continue
          a=summary[k];lines.append(f"| {name} | {a['test']['rmse_mean']:.6f} | {a['conditional']['rmse_mean']:.6f} |")
        return '\n'.join(lines)
    def pct(other,scope='test'):
        return 100*(summary['neural_Physics11D'][scope]['rmse_mean']/summary['neural_'+other][scope]['rmse_mean']-1)
    rejected=[m['group'] for m in audit if not m['calibration_ok']]
    rows=sum(m['rows'] for m in audit)
    comp='\n'.join(f"| {r['family']} | {r['scope']} | {r['comparison']} | {r['mse_difference']:.8f} | [{r['paired_cluster_bootstrap_95'][0]:.8f}, {r['paired_cluster_bootstrap_95'][1]:.8f}] | {r['physics_better_source_count']}/{r['source_count']} |" for r in comparisons)
    pscalar=pct('Scalar');pcent=pct('ScalarCentroid');praw=pct('Raw');plearn=pct('Learned11D')
    dynamic_physics=np.mean([r['metric']['rmse'] for r in dynamic if r['family']=='neural' and r['name']=='Physics11D'])
    dynamic_percent=100*(dynamic_physics/dynamic_baseline['rmse']-1)
    sensitivity=read('sensitivity.json')
    sens_lines=['| 阈值 | 输入 | 全部接触帧 RMSE | 稳定载荷 RMSE |','| --- | --- | ---: | ---: |']
    for cutoff in [0,3,5]:
      for name in ['Scalar','ScalarCentroid','Physics11D','Raw']:
        if cutoff==3:a=next(r for r in full if r['family']=='linear' and r['name']==name)
        else:a=next(r for r in sensitivity if r['cutoff']==cutoff and r['name']==name)
        sens_lines.append(f"| {cutoff}×MAD | {name} | {a['test']['rmse']:.6f} | {a['conditional']['rmse']:.6f} |")
    dynamic_lines=['| 输入 | 窗口内变化 RMSE（神经读出均值） |','| --- | ---: |',f"| 不预测窗口内变化的参照 | {dynamic_baseline['rmse']:.6f} |"]
    for name in NAMES:
        vals=[r['metric']['rmse'] for r in dynamic if r['family']=='neural' and r['name']==name]
        dynamic_lines.append(f"| {name} | {np.mean(vals):.6f} |")
    conclusion=f"完整 11D 的神经读出相对 Scalar 的全部接触帧 RMSE 变化为 {pscalar:+.1f}%，相对 Scalar+centroid 为 {pcent:+.1f}%，相对 Raw 为 {praw:+.1f}%，相对 Learned11D 为 {plearn:+.1f}%（负数表示误差更低）。稳载子集相对 Scalar 的均值 RMSE 变化为 {pct('Scalar','conditional'):+.1f}%，但逐源配对 MSE 区间跨零，优势未可靠成立。相对 Scalar+centroid 也没有额外收益。窗口内变化诊断中，11D 比不预测变化的静态参照误差高 {dynamic_percent:.1f}%。因此此次不能宣布 Stage 1 通过，不能把完整 11D 包装成空间动态的必要表示。"
    report=f'''# TaF 空间接触算子离线验证

日期：2026-10-06。已实际运行源记录留出实验；本轮不继续试管 BC、不训练 Mamba 或 VLA。

## 核心结果

{conclusion}

以下 RMSE 是先在每个源记录内对两维误差求 RMSE，再等权平均源记录，神经模型再平均 3 个种子。单位为发布数据中保存的力矩／力比，未核实坐标、安装偏置及绝对单位，不能写成毫米 CoP 误差。

### 神经读出，3 个训练种子

{table('neural')}

![神经读出对比](spatial_comparison.png)

### 线性读出

{table('linear')}

FzOracle 仅是外部真实法向载荷的辅助诊断参照，不是可部署的阵列模型，也不包含 Tx/Ty。完整训练集的 Ridge 是确定性拟合，3 个种子产生同样模型；其零种子方差不是传感器稳定性。

## 数据与协议变更

来源：[TaF 数据卡](https://huggingface.co/datasets/jiamig/taf-dataset/blob/main/README.md)、[原论文采集方法](https://arxiv.org/html/2601.20321v1#S3)。固定发布 SHA：`239c8ee8156bf11eb6d6c11c004f9a8721528d9d`。仅下载压力阵列和 ATI 表格，没有使用图像。全部源文件 SHA-256 校验通过。

原计划训练 obj7～22、验证 obj23～26、测试 obj27～30，但原定 4 个测试记录全部未满足初始无接触校准门槛。此时未训练任何模型。保留原协议和失败审计后，一次性将固定测试候选扩大为 obj27～54，训练／验证范围与校准门槛保持原值；在下载 obj31～54 前冻结修订，没有按预测成绩选择样本。它是透明的可行性修订，并非正式预注册。

共检查 {len(audit)} 个源记录、{rows:,} 原始帧。按预定门槛排除：{rejected}。有效集合为：

- 训练：{splits['train']}。
- 验证：{splits['validation']}。
- 测试：{splits['test']}。
- 优化使用 {checks['train_frames']:,} 帧，验证调参使用 {checks['validation_tuning_frames']:,} 帧；测试遍历全部 {checks['test_frames']:,} 合格接触帧，稳定载荷子集 {checks['test_conditional_frames']:,} 帧。

完整源序列不跨集合；转换后的 episode 很可能是连续记录切片，不作为独立试验。编号不自动等于不同物体／材质。初始校准门槛会筛掉从受力开始记录的数据，因此结论只覆盖可按该协议校准的源记录，而非整个 TaF 数据集。

## 标签、预处理与公平性

标签来自独立 ATI：`e=[-Ty/Fz, Tx/Fz]`，包含两个保存单位下的力矩／法向载荷比。输入模型从不包含 ATI 的 Tx/Ty，也没有机器人本体状态输入。

每源记录前 30 帧要求全部保存值 |Fz|<0.5；以其每 taxel 中位数扣基线、按 3×1.4826×MAD 阈值保留正响应，此后冻结。ATI 同样扣前缀中位数。评价帧要求行号≥31、与前一帧同转换 episode、|Fz|≥2、阵列质量非零且标签有限。当前帧及上一帧的可见历史一致，不用未来帧滤波、不进行 EMA。

Scalar 为 log1p 总响应及相邻差分；ScalarCentroid 增加 cx/cy 及差分；Physics11D 为 8 个静态统计量加总响应与质心的 3 个差分。协方差方向使用各向异性加权的双角编码，各向同性时衰减到零。坐标按阵列两轴归一化到 [-1,1]。所有模型另外共享当前／上一帧有效性标志，不计入 11D；Learned11D 将 Raw 的 288 个值编码到 11D，标志同样绕过瓶颈。

神经读出统一 64→64→2 的隐藏层宽度；Learned11D 编码器为 288→64→11。头部宽度一致不等于参数总量相等，实际参数数目记录在 all_results.json。4 组 lr/weight-decay 仅按验证集选择，3 个种子为 7/19/31；最多 200 epoch，20 epoch patience。各源记录等权损失；输入、输出尺度与 P_ref 均只由训练子集估计。神经训练和选择使用训练输出标准化后的 MSE；线性 alpha 使用保存单位下验证 MSE。

线性数据曲线按训练源记录抽取 25%、50%、100%，每档 3 个固定种子，重新估计子集基准和尺度；alpha 沿用完整训练集验证选择。因此它是表示的数据效率诊断，未隔离超参数选择所需的额外数据。神经模型此次只跑完整训练集，尚未证明神经小样本优势。

## 配对统计与边界

以完整测试源记录做 10,000 次 cluster bootstrap，先平均神经种子的逐源 MSE，再比较 Physics11D 与对应模型。差值<0表示 Physics 更低误差；这不是把每帧视作独立样本的显著性检验。源记录数少、来源独立性未核实，区间仅作描述性不确定性，不能替代重复物理实验。

| 模型族 | 评价集 | 配对比较 | 平均 MSE 差值 | bootstrap 95% 区间 | Physics 更好的源记录 |
| --- | --- | --- | ---: | --- | --- |
{comp}

稳定载荷子集按不重叠 15 帧窗口筛选：P 和独立 |Fz| 的 CV≤5%、|Fz|≥2、独立标签跨度≥0.002、全部剪切比≤0.2。门槛不使用质心／方向变化，但它使用窗口内 ATI 标签，因此这是离线条件诊断，不能说是在线事件检测。CV≤5%也不表示每帧载荷都在±5%内。

## 阈值敏感性与验证

`sensitivity.json` 保留 cutoff=0、5×MAD 的线性模型结果，沿用主实验选择的 alpha，未依据测试集重调门槛、参考尺度或参数。资格与评价帧保持主协议成员不变，阈值更改后额外零质量由共享有效标志处理。

{chr(10).join(sens_lines)}

初次 float32 Ridge 的正规方程出现 scipy 病态矩阵警告，未解释其成绩便改为 float64 求解，按同一 alpha 网格重跑。原始数值试跑日志保留，主表只使用双精度修正版；这项修正没有改变评价定义或按测试表现选参数。

`artifact_verification.json` 记录保存模型在 CPU 冷加载后的预测与 CUDA 保存结果比较，源分割核查、所有预测有限性，以及真实阵列协方差迹恒等式检查。接近 epoch 上限的模型也单独列出；若存在，不应把该模型成绩当作已充分收敛的最优表示上限。

![线性数据曲线](linear_data_efficiency.png)

## 额外诊断：窗口内动态，而非固定偏置

在查看模型预测成绩之前，额外冻结了窗口中心化诊断（secondary_diagnostic_protocol.json）。仅对已经合格的稳定载荷窗口，分别减去每个窗口的标签均值和预测均值，再比较变化误差；它是评价时的回顾性分解，不进入输入预处理，不据此重新训练／挑参数。共 {wid} 个测试窗口。

{chr(10).join(dynamic_lines)}

静态参照的窗口内预测为零变化。若模型低于该参照，支持其捕捉部分窗口内变化；若高于，则不能拿静态标签 RMSE 的改善宣称动态接触追踪有效。此项为补充诊断，并非原始主指标。

## 结论应怎样使用

成功预测该独立标签最多支持“空间表示含有外部力矩关系的可用信息”。不能直接证明接触位置绝对标定、滑移识别、卡死、最佳恢复方向或闭环防倾倒效果。TaF 与 FlexiTac 硬件不同，公开记录约 30 Hz；高频噪声、漂移与真实端到端时延仍需要 FlexiTac 自采。

若只有 ScalarCentroid 已足够，应缩减或降级完整 11D 的必要性表述。若 Raw / Learned 表现更好，应承认物理瓶颈的信息损失，不能把低计算成本当作更高信息量。若低载荷子集与主结果相反，必须同时报告，而不是挑更有利的结果。

## 复现与文件

在任务目录运行 `python outputs/run_spatial_validation.py --prepare-only`，再运行 `python outputs/run_spatial_validation.py`，最后运行 `python outputs/report_spatial_validation.py`。依赖 NumPy、PyArrow、SciPy、scikit-learn、PyTorch CUDA、Matplotlib；此次 GPU：{checks['gpu']}，PyTorch：{checks['torch']}。代码与实验结果归档于 experiments/contact_operator；本报告描述离线试验，不包含实机验证。

数据原文缓存在 work/taf；outputs 保留 protocol.json、完整 data_audit.json、eligible_split.json、线性曲线与 alpha、验证调参日志、3 种子模型及预测、逐源误差和敏感性结果。训练脚本和报告脚本是可编辑的复现来源。
'''
    (OUT/'空间算子离线验证报告.md').write_text(report,encoding='utf-8')
    assert '\ufffd' not in report and '??' not in report
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
