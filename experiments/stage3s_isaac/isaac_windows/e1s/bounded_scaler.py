"""Training-only robust scales with physical channel floors and saturation."""
import numpy as np


def floors(kind):
    # Proxy force is a sum of 384 normalized SDF responses, NOT newtons.
    f=np.array([1.,.02,.02,.02,.05,.002,.002,.002],np.float32)
    if kind=='scalar': return np.tile(f[[0,4]],4)
    if kind in ('static','s4'): return np.tile(f[:4],4)
    if kind=='no_cop': return np.tile(f[[0,3,4,7]],4)
    if kind=='no_area': return np.tile(f[[0,1,2,4,5,6]],4)
    if kind=='raw': return np.full((4,12,32),.02,np.float32)
    return np.tile(f,4)


def fit(x,kind):
    center=np.median(x,axis=0)
    scale=np.maximum(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0),floors(kind))
    return center.astype(np.float32),scale.astype(np.float32)


def transform(x,center,scale):
    return np.clip((x-center)/scale,-8.,8.).astype(np.float32)
