"""Re-render figures from verified metrics without repeating native trials."""
from pathlib import Path
import json
from analyze_alignment import plot_pareto,phase_figures,plot_anchor
root=Path(__file__).resolve().parent
out=root/'outputs/alignment_report_20261014_01'
assert json.loads((out/'verification.json').read_text())['status']=='passed'
data=json.loads((out/'results.json').read_text()); summaries=data['summaries']
plot_pareto(data['original_summaries'],out/'pareto_original','Original frozen policies | 140 identical sampled loads',{'nominal','high'})
aligned={name:row for name,row in summaries.items() if not name.startswith('trigger_')}
plot_pareto(aligned,out/'pareto_aligned','Pre-load alignment interventions | 140 identical sampled loads',{'low_fixed','high_fixed','max_fixed'})
phase_figures(data['rows'],summaries,out)
plot_anchor(root/'outputs/alignment_20261014_01',root/'outputs/trigger_static_reference_20261014_01',out,summaries)
print('FIGURES_RENDERED')
