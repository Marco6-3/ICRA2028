import json,hashlib,shutil
from pathlib import Path
D=Path('outputs/stage2p_mamba');repo=Path(r'C:\Users\mingzhe Liu\Documents\Codex\2026-10-06\https-github-com-marco6-3-icra2028\work\ICRA2028')
s=json.loads((D/'summary.json').read_text());graph=json.loads((D/'cuda_graph_benchmark.json').read_text());r=json.loads((D/'results.json').read_text())
extra=['','## 同权重 CUDA Graph 部署补测','','此补测在全部训练结束后进行，不重新训练，也不修改窗口预测或选模。两种架构使用相同的Graph优化方式；固定输入/状态缓冲，重放同一组FP32计算。测量包含GPU内输入拷贝、Graph replay及同步，不包含Graph构建、传感器和特征计算。','', '| 架构 | 输入 | T | P50 ms | P95 ms | P99 ms | 最大 ms | <1ms比例 | 与eager误差 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for a in graph['results']:
    if a['passed']:extra.append(f"| {a['architecture']} | {a['name']} | {a['T']} | {a['p50_ms']:.3f} | {a['p95_ms']:.3f} | {a['p99_ms']:.3f} | {a['max_ms']:.3f} | {100*a['fraction_below_1ms']:.1f}% | {a['parity_max_normalized_error']:.3g} |")
    else:extra.append(f"- {a['architecture']} {a['key']} Graph补测失败：{a['error']}")
extra+=['','完整逐帧延迟在 `cuda_graph_benchmark.json`。Graph优化结果和原生eager结果分别报告；即使P95低于1ms，也不等于所有帧严格小于1ms。','', '![侧向力矩RMSE](moment_rmse.png)']
summary=['## 实测结论','']
for a in s['rows']:
    if a['name']=='MCF':
        ci=next(v for v in s['comparisons'] if v['comparison']==f"Mamba-GRU MCF T{a['T']}")['ci95'][1]
        summary.append(f"- MCF T={a['T']}：GRU力矩RMSE {a['GRU'][1]:.7f}，Mamba {a['Mamba'][1]:.7f}；Mamba降幅 {a['Mamba_reduction_percent'][1]:.2f}%。Mamba−GRU MSE的95%源配对区间 [{ci[0]:.6g}, {ci[1]:.6g}]。")
v=next(a for a in s['comparisons'] if a['comparison']=='Mamba MCF T100-T50')
summary.append(f"- MCF-Mamba从T50增加到T100：侧向力矩RMSE降幅 {v['rmse_reduction_percent'][1]:.2f}%；MSE差的95%区间 {v['ci95'][1]}。")
summary+=['- 这些结果对应固定小模型、既定数据划分和沿用的GRU优化超参数；本轮完成架构替换对照，不是Mamba超参数最优性搜索。','']
mb=[a for a in graph['results'] if a['architecture']=='Mamba' and a['passed']]
eg=[a['streaming']['p95_ms'] for a in r if a['architecture']=='Mamba' and 'streaming' in a]
summary.insert(-1,f"- 原生Mamba eager P95为 {min(eg):.3f}–{max(eg):.3f} ms；同权重CUDA Graph P95为 {min(a['p95_ms'] for a in mb):.3f}–{max(a['p95_ms'] for a in mb):.3f} ms。四组各1,000帧Graph测量均低于1ms，测量范围是GPU驻留输入的模型推理；GRU同样优化后更快，未得到Mamba速度优于GRU的结论。")
text=(D/'实验报告.md').read_text(encoding='utf-8');first,rest=text.split('\n',1)
(D/'实验报告.md').write_text(first+'\n\n'+'\n'.join(summary)+rest+'\n'.join(extra)+'\n',encoding='utf-8')
for name in ['benchmark_mamba_graph.py','plot_mamba.py','finalize_mamba.py']:shutil.copy2(Path('work')/name,D/name)
shutil.copy2(Path('work/mamba_run.log'),D/'training.log')
env=json.loads((D/'environment.json').read_text())
(D/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in env['packages'].items())+'\n')
dest=repo/'experiments/contact_operator/outputs/stage2p_mamba_20261007';dest.mkdir(exist_ok=True)
for p in D.iterdir():
    if p.is_file() and p.suffix in ['.md','.json','.png','.pdf','.py','.txt']:shutil.copy2(p,dest/p.name)
doc=repo/'research/STAGE2P_MAMBA.md';old=doc.read_text(encoding='utf-8');old=old.replace('实际结果在完整24组训练与复算后追加。','24组训练和指标复算已完成。')
doc.write_text(old+'\n'+'\n'.join(summary)+'\n完整报告：[实验报告](../experiments/contact_operator/outputs/stage2p_mamba_20261007/实验报告.md)。模型权重和逐窗口预测保存在本地交付目录的GRU、Mamba子目录。\n',encoding='utf-8')
offline=repo/'research/OFFLINE_RESULTS.md'
offline.write_text(offline.read_text(encoding='utf-8')+'\n## 2026-10-07 Mamba / GRU 当前状态重构\n\n24组离线模型完成，公共锚点、FP32、三种子，源配对统计与流式延迟（eager / CUDA Graph）分开报告。详见 [Mamba实验](STAGE2P_MAMBA.md)。\n\n'+'\n'.join(summary)+'\n',encoding='utf-8')
manifest=[dict(path=str(p.relative_to(D)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in D.rglob('*') if p.is_file() and p.name!='manifest.json']
(D/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('\n'.join(summary));print('Artifacts',len(manifest))
