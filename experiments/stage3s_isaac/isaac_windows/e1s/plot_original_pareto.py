"""Cost/safety view of the already frozen phase grid; no model changes."""
from pathlib import Path
import json, hashlib
from analyze_alignment import plot_pareto
root=Path(__file__).resolve().parent
source=root/'outputs/phase_report_20261014_01'
out=root/'outputs/pareto_original_20261014_01'; out.mkdir(exist_ok=True)
findings=json.loads((source/'findings.json').read_text())
phase=json.loads((source/'phase_results.json').read_text())
points={}
for name in ('nominal','high','scalar','mcf'):
    selected=[r for r in phase['rows'] if r['group']==name]
    selected=[r for r in selected if r.get('direction')]
    points[name]=dict(safe_cells=findings['safe_cells'][name],total_cells=140,
        safe_percent=100*findings['safe_cells'][name]/140,
        pre_normal_n=findings['predisturbance_grip'][name]['physical_normal_n'],
        mean_effort_n_s=findings['mean_effort'][name])
plot_pareto(points,out/'pareto','Frozen original policies | 140 identical sampled loads',{'nominal','high'})
result=dict(points=points,common_safe_cells=findings['common_safe_cells'],
    common_safe_effort=findings['common_safe_effort'],
    mcf_initial_force_reduction_vs_scalar_percent=100*(1-points['mcf']['pre_normal_n']/points['scalar']['pre_normal_n']),
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'findings.json',source/'phase_results.json')})
(out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
