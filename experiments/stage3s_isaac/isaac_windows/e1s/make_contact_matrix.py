import json
from pathlib import Path
from disturbance import Trial
specs=[dict(direction='zero')]
for delta in (0.,.001):
    for direction in ('+Fx','-Fx','+Fy','-Fy'):
        specs.append(dict(direction=direction,point_m=(0.,0.,.015),preload_delta_rad=delta))
trials=[Trial(trial_id=f'c{i:03d}_r{r:02d}',repeat=r,frame='reference',**spec).metadata()
        for r in range(2) for i,spec in enumerate(specs)]
Path(__file__).with_name('contact_matrix.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
