import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('outputs/stage2p_mamba');r=json.loads((D/'results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained',sharey=True)
colors={'GRU':'#386cb0','Mamba':'#d95f02'}
for j,name in enumerate(['Scalar','MCF']):
    ax=axes[j]
    for arch in ['GRU','Mamba']:
        values=np.array([[a['macro']['rmse'][1] for a in r if (a['architecture'],a['name'],a['T'])==(arch,name,T)] for T in [50,100]])
        ax.plot([50,100],values.mean(1),'-o',label=arch,color=colors[arch])
        for i,T in enumerate([50,100]):ax.scatter([T]*3,values[i],s=20,alpha=.45,color=colors[arch])
    ax.set(title=name,xlabel='History (frames)',ylabel='Lateral moment RMSE (stored units)',xticks=[50,100]);ax.grid(alpha=.2);ax.legend()
fig.suptitle('TaF current-state reconstruction: 17 test sources, 3 seeds')
fig.savefig(D/'moment_rmse.png',dpi=180);fig.savefig(D/'moment_rmse.pdf');plt.close(fig)
