import hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/'outputs/stage2p_mamba'
repo=Path(r'C:\Users\mingzhe Liu\Documents\Codex\2026-10-06\https-github-com-marco6-3-icra2028\work\ICRA2028')
r=json.loads((D/'results.json').read_text());assert len(r)==24
assert json.loads((D/'execution.json').read_text())['completed']
e=np.load(D/'examples.npz');old=np.load(ROOT/'outputs/stage2p_nowcast_fp32/examples.npz')
idx=e['inherited_index']
for key in ['group','anchor','y','episode','dt']:np.testing.assert_array_equal(e[key],old[key][idx])
te=e['test'];y=e['y'][te];g=e['group'][te];sources=np.unique(g);checks=[]
for a in r:
    pred=np.load(D/a['architecture']/(a['key']+'_pred.npy'))
    assert pred.shape==y.shape and np.isfinite(pred).all()
    err=(pred-y).reshape(-1,2,2)
    mse=np.array([np.mean(err[g==s]**2,axis=(0,2)) for s in sources])
    np.testing.assert_allclose(np.sqrt(mse).mean(0),a['macro']['rmse'],rtol=1e-6)
    np.testing.assert_allclose(mse.mean(0),a['macro']['mse'],rtol=1e-6)
    checks.append(dict(architecture=a['architecture'],key=a['key'],metrics_recomputed=True))
def selected(arch,name,T):return [a for a in r if (a['architecture'],a['name'],a['T'])==(arch,name,T)]
def scores(arch,name,T):return np.mean([a['macro']['rmse'] for a in selected(arch,name,T)],axis=0)
def per(arch,name,T):return np.mean([[p['mse'] for p in a['per_source']] for a in selected(arch,name,T)],axis=0)
def paired(d):
    rng=np.random.default_rng(20261007);b=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    return dict(mean=d.mean(0).tolist(),ci95=np.quantile(b,[.025,.975],axis=0).T.tolist())
rows=[];comparisons=[]
for T in [50,100]:
    for name in ['Scalar','MCF']:
        gr=scores('GRU',name,T);ma=scores('Mamba',name,T)
        rows.append(dict(T=T,name=name,GRU=gr.tolist(),Mamba=ma.tolist(),Mamba_reduction_percent=(100*(1-ma/gr)).tolist()))
        comparisons.append(dict(comparison=f'Mamba-GRU {name} T{T}',**paired(per('Mamba',name,T)-per('GRU',name,T))))
for arch in ['GRU','Mamba']:
    for name in ['Scalar','MCF']:
        comparisons.append(dict(comparison=f'{arch} {name} T100-T50',rmse_reduction_percent=(100*(1-scores(arch,name,100)/scores(arch,name,50))).tolist(),**paired(per(arch,name,100)-per(arch,name,50))))
    for T in [50,100]:
        comparisons.append(dict(comparison=f'{arch} T{T} MCF-Scalar',**paired(per(arch,'MCF',T)-per(arch,'Scalar',T))))
summary=dict(metric_order=['force_xy','moment_xy'],rows=rows,comparisons=comparisons,models=24,bootstrap='10000 paired source draws; average three seeds first; source-level uncertainty, not independent frame or seed replicates')
(D/'summary.json').write_text(json.dumps(summary,indent=2))
(D/'artifact_verification.json').write_text(json.dumps(dict(common_labels_exactly_match_prior=True,checks=checks),indent=2))
counts=json.loads((D/'sample_counts.json').read_text());protocol=json.loads((D/'protocol.json').read_text())
lines=['# Stage 2P：Mamba 与 GRU 离线 Now-casting 对照','','2026-10-07。24 组训练完成。Stage 1 和此前 GRU 实验均保留。','','## 实验设置','',
'- 输入 Scalar `[f,Δf]` / MCF8；标签为当前 ATI `[Fx,Fy,Mx,My]`，不预测未来。历史 T=50/100；种子7/19/31。',
'- GRU：两层 hidden64，47,460 参数。Mamba-1：两层 residual + RMSNorm block，d_model50、expand2、state16、causal conv4，47,978 参数（多1.09%）。原 token MLP 和 pool 前端相同；Mamba 增加32→50投影并使用50→4输出。是参数量级匹配，不是逐参数完全相等。',
'- Mamba 后端为 `mambapy==1.2.0`，PyTorch 并行 selective scan 和逐帧 step；没有使用官方融合 CUDA 内核。具有输入相关 Δ/B/C、门控、因果卷积和 SSM 递推。',
'- FP32，matmul/cuDNN TF32 均关闭；相同 AdamW lr0.001 / wd0.0001、batch64、最多40epoch、patience8。按等权源的归一化验证MSE选择epoch。没有按测试分数调参。',
f"- T100公共样本：训练 {counts['train']['windows']}、验证 {counts['validation']['windows']}、测试 {counts['test']['windows']}；源数量分别 {counts['train']['sources']}/{counts['validation']['sources']}/{counts['test']['sources']}。T50和T100、GRU和Mamba使用完全相同锚点。",
'- 从此前完整锚点网格取有100帧连续历史的子集；保留stride20、接触活跃阈值和末端30帧余量。差分还使用窗口首帧的前一帧；检查同episode。训练源输入归一化、每源空载首30帧ATI零偏处理不变。',
'- 力、力矩分开评分，单位沿用数据保存单位。先计算每个源RMSE，再等权平均源和三个种子。','','## RMSE','',
'| T | 输入 | GRU 剪切力 | Mamba 剪切力 | GRU 侧向力矩 | Mamba 侧向力矩 | Mamba 力矩降幅 |',
'|---:|---|---:|---:|---:|---:|---:|']
for a in rows:lines.append(f"| {a['T']} | {a['name']} | {a['GRU'][0]:.7f} | {a['Mamba'][0]:.7f} | {a['GRU'][1]:.7f} | {a['Mamba'][1]:.7f} | {a['Mamba_reduction_percent'][1]:.2f}% |")
lines+=['','正降幅表示Mamba误差更低。新表的GRU也在公共子集上重跑，不能将旧T50结果直接拼入本表。','','## 配对不确定性','','三个种子的每源MSE先平均，再对测试源进行10,000次配对bootstrap；以下是差值的95%区间。负数有利于左项。未把相邻帧视作独立重复；这是固定种子集合下的源级区间，未校正多重比较。','', '| 比较 | 剪切力 MSE CI | 侧向力矩 MSE CI |','|---|---|---|']
for a in comparisons:
    c=a['ci95'];lines.append(f"| {a['comparison']} | [{c[0][0]:.5g}, {c[0][1]:.5g}] | [{c[1][0]:.5g}, {c[1][1]:.5g}] |")
lines+=['','## batch=1 流式延迟','','seed7模型，真实测试episode连续最多1,000帧；100帧预热后重置状态。每帧同步GPU测量墙钟时间，输入已在GPU。包含模型前端与输出头；不包含传感器、特征提取和传输。原生eager后端，不使用CUDA Graph或融合Mamba内核。','', '| 架构 | 输入 | 训练T | P50 ms | P95 ms | P99 ms | 最大 ms | <1ms比例 | 状态字节 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for a in r:
    if 'streaming' not in a:continue
    b=a['streaming'];assert b['state_bytes']==b['state_bytes_max']
    lines.append(f"| {a['architecture']} | {a['name']} | {a['T']} | {b['p50_ms']:.3f} | {b['p95_ms']:.3f} | {b['p99_ms']:.3f} | {b['max_ms']:.3f} | {100*b['fraction_below_1ms']:.1f}% | {b['state_bytes']} |")
lines+=['','这是本机本后端的实测，不能等同于官方融合内核的速度。缓存大小与累计历史长度无关；GRU同样具有固定状态递推。窗口RMSE对应每窗口清零状态；连续流式测试用于延迟，不能自动等同于无限历史在线RMSE。','','## 核验与产物','',
'- 并行扫描/顺序扫描前向和参数、输入梯度一致；T=1/50/100流式与整段输出一致，因果检查通过。',
'- 独立对照 Transformers 4.57.6 的 MambaMixer.slow_forward，复制相同权重后核验前向与梯度。详见 hf_reference_checks.json。',
'- 24组预测指标独立复算；8组seed7模型CPU冷加载和训练后流式一致性检查通过。公共子集的ATI标签、episode、时间戳与旧nowcast逐元素一致。',
'- 模型权重、每个测试窗口预测、训练损失轨迹、最佳epoch、样本索引、协议及源码哈希均保存。',
'- 本轮验证的是该规模Mamba对当前ATI重构的表现，不直接标定微滑移、剪切应变或确定GRU长历史瓶颈的物理原因。',
'', '[结构化汇总](summary.json) · [完整模型结果](results.json) · [训练协议](protocol.json) · [复现说明](README.md)',
'', '实现来源：[Mamba 官方](https://github.com/state-spaces/mamba)、[mamba.py 后端](https://github.com/alxndrTL/mamba.py)。']
lines+=['','## 训练选择记录','','| 架构 | 输入 | T | 种子 | 最佳 epoch | 验证损失 | 训练秒 |','|---|---|---:|---:|---:|---:|---:|']
for a in r:lines.append(f"| {a['architecture']} | {a['name']} | {a['T']} | {a['seed']} | {a['best_epoch']} | {a['validation_loss']:.5f} | {a['train_seconds']:.1f} |")
(D/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for name in ['run_stage2p_mamba.py','run_stage2p_nowcast.py']:
    shutil.copy2(repo/'experiments/contact_operator'/name,D/name)
    assert hashlib.sha256((D/name).read_bytes()).hexdigest()==protocol['hashes'][name]
shutil.copy2(Path(__file__),D/'report_mamba.py');shutil.copy2(ROOT/'work/check_mamba_reference.py',D/'check_mamba_reference.py')
(D/'README.md').write_text('''# TaF Mamba / GRU 离线对照复现

完整结果见 `实验报告.md`，原始指标见 `results.json`，配对区间见 `summary.json`。

依赖：Python3.11、torch2.11.0+cu128、numpy、pyarrow、threadpoolctl、mambapy1.2.0；独立参考检查另需transformers4.57.6。训练设备为RTX5060 Laptop GPU，FP32且TF32关闭。

```powershell
python run_stage2p_mamba.py --cache-root '<含taf目录的数据根目录>' --out '<新的空输出目录>'
```

数据需包含obj7至obj54 parquet以及对应objN_listing.json，脚本逐文件验证LFS SHA256。必须选择新目录，防止覆盖旧实验。

`GRU/` 和 `Mamba/` 内每组保存 `.pt` 与 `_pred.npy`。预测行顺序对应 `examples.npz` 的 `test` 索引。加载自有检查点时使用 `torch.load(..., weights_only=False)`；根据架构选择脚本中的 `GRUProbe` 或 `MambaProbe`，权重为 `state_dict`。归一化和反归一化参数保存在检查点。

`protocol.json` 记录脚本和后端SHA256；`implementation_checks.json`、`hf_reference_checks.json`、`artifact_verification.json` 记录数值核验。流式单步调用 `step(net, frame, state)` 并在episode开始清空state。不能在独立窗口间误携带状态。
''',encoding='utf-8')
print(json.dumps(summary,indent=2))
