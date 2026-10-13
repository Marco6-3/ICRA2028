"""Additional physical integration coverage for configurable load frames."""
import json
from pathlib import Path
from disturbance import Trial

trials=[]
for frame in ('world','object','reference'):
    for direction in ('+Fx','-Fx','+Fy','-Fy'):
        for repeat in range(2):
            trials.append(Trial(trial_id=f'{frame}_{direction[0].replace("+","pos").replace("-","neg")}{direction[-1]}_r{repeat:02d}',
                                frame=frame,direction=direction,point_m=(0.,0.,.015),repeat=repeat).metadata())
Path(__file__).with_name('coordinate_matrix.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
