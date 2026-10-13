"""Causal single-aperture expert and tactile-only bounded GRU student."""
import numpy as np
import torch
from torch import nn
from bounded_scaler import transform

MAX_DELTA=.003


class ReflexGRU(nn.Module):
    def __init__(self,kind='mcf'):
        super().__init__(); self.kind=kind
        self.gru=nn.GRU(32 if kind=='mcf' else 8,64,num_layers=2,batch_first=True)
        self.head=nn.Linear(64,1)

    def forward(self,x):
        z,_=self.gru(x)
        # Positive incremental closure; bounded by the declared action authority.
        return torch.sigmoid(self.head(z[:,-1]))[:,0]


class Controller:
    def __init__(self,kind,dt,checkpoint=None):
        self.kind=kind; self.dt=dt; self.memory=0.; self.history=[]
        self.hidden=None; self.streaming=False
        torch.set_num_threads(2)
        if kind in ('mcf','scalar'):
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert ck['kind']==kind and ck['max_delta_rad']==MAX_DELTA
            self.model=ReflexGRU(kind); self.model.load_state_dict(ck['state_dict']); self.model.eval()
            self.center=ck['center']; self.scale=ck['scale']; self.window=ck['history']
            self.streaming=ck.get('streaming',False)

    def step(self,mcf,velocity,angle):
        # State at t before applying the t disturbance: no future load or clock.
        linear=float(np.linalg.norm(velocity[:3])); angular=float(np.linalg.norm(velocity[3:]))
        requested=np.clip(.75*max(0.,linear-.00015)+.003*max(0.,angular-.01),0.,MAX_DELTA)
        self.memory=max(float(requested),self.memory*np.exp(-self.dt/.25))
        teacher=self.memory
        if self.kind=='nominal': value=0.
        elif self.kind=='high': value=MAX_DELTA
        elif self.kind=='oracle': value=teacher
        else:
            feature=mcf.reshape(-1) if self.kind=='mcf' else mcf[:,[0,4]].reshape(-1)
            z=transform(feature,self.center,self.scale); self.history.append(z)
            self.history=self.history[-self.window:]
            padded=[self.history[0]]*(self.window-len(self.history))+self.history
            with torch.inference_mode():
                if self.streaming:
                    out,self.hidden=self.model.gru(torch.tensor(z[None,None],dtype=torch.float32),self.hidden)
                    value=float(torch.sigmoid(self.model.head(out[:,-1]))[0,0])*MAX_DELTA
                else:
                    value=float(self.model(torch.tensor(np.array(padded)[None],dtype=torch.float32))[0])*MAX_DELTA
        return float(value),float(teacher),linear,angular
