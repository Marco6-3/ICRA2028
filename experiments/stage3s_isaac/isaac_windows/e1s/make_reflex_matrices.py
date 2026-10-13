import json
from pathlib import Path
from disturbance import Trial
root=Path(__file__).resolve().parent
train=[]
for j,(amplitude,wave) in enumerate((a,w) for a in (.12,.20,.28) for w in ('step','ramp')):
    for i,direction in enumerate(('+Fx','-Fx','+Fy','-Fy')):
        # Middle-amplitude ramp is validation; no identical replay in training.
        split='val' if amplitude==.20 and wave=='ramp' else 'train'
        tr=Trial(f'{split}_{j:02d}_{i}',direction,amplitude_n=amplitude,point_m=(0,0,.015),frame='reference',waveform=wave,
                 friction_scale=(.8,1.,1.2)[(j+i)%3],sdf_gain_scale=(.8,1.,1.2)[(2*j+i)%3],onset_s=.25,duration_s=.25)
        train.append(tr.metadata())
train.append(Trial('train_zero',direction='zero',frame='reference').metadata())
test=[]
for j,a in enumerate((.16,.20)):
    for i,d in enumerate(('+Fx','-Fx','+Fy','-Fy')):
        test.append(Trial(f'test_{j:02d}_{i}',d,amplitude_n=a,point_m=(0,0,.015),frame='reference',onset_s=.35,duration_s=.25).metadata())
for name,matrix in [('reflex_teacher_matrix.json',train),('reflex_test_matrix.json',test)]:
    (root/name).write_text(json.dumps(matrix,indent=2),encoding='utf-8')
print('Frozen teacher train/validation and heldout-timing evaluation matrices')
