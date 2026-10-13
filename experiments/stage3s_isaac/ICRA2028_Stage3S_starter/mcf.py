"""Normal-pressure-only Scalar and Minimalist Contact Flow for tactile arrays.

This is an observation adapter, NOT a six-axis force sensor, calibrated FlexiTac
model, or closed-loop controller.
"""
from __future__ import annotations
import numpy as np


def per_finger_features(pressure: np.ndarray, previous: np.ndarray | None = None, *,
                        threshold: float = 0.0):
    """Compute Scalar 2D and MCF 8D per finger.

    pressure shape [fingers, rows, cols] or [rows, cols]. Positive values mean
    normal response. One image/frame at a time; previous MUST be adjacent in
    the same episode. Coordinates are normalized to [-1,1] on each axis.

    Returns (scalar [fingers,2], mcf [fingers,8]). No cross-finger mixing.
    """
    a = np.asarray(pressure, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.ndim != 3 or a.shape[1] < 2 or a.shape[2] < 2:
        raise ValueError('pressure must be [F,H,W], H,W>=2')
    if not np.isfinite(a).all() or threshold < 0:
        raise ValueError('non-finite reading or negative threshold')
    if np.any(a < -1e-6):
        raise ValueError('negative readings: verify sensor sign convention before clipping')
    f = np.where(a > threshold, a, 0.)
    y, x = np.meshgrid(np.linspace(-1.,1.,a.shape[1]), np.linspace(-1.,1.,a.shape[2]), indexing='ij')
    total = f.sum(axis=(1, 2))
    denom = np.maximum(total, 1e-12)
    cx = (f*x).sum(axis=(1, 2))/denom
    cy = (f*y).sum(axis=(1, 2))/denom
    area = (f > 0.).mean(axis=(1, 2))
    s = np.stack([total,cx,cy,area],axis=1)
    if previous is None:
        ds = np.zeros_like(s)
    else:
        p = np.asarray(previous)
        if p.ndim == 2:
            p = p[None,:,:]
        if p.shape != a.shape:
            raise ValueError('previous/current tactile map shapes differ')
        # Recursive evaluation uses exactly same pressure preprocessing.
        _, old = per_finger_features(p,threshold=threshold)
        ds = s - old[:,:4]
    scalar = np.stack([s[:,0],ds[:,0]],axis=1)
    return scalar.astype(np.float32), np.concatenate([s,ds],axis=1).astype(np.float32)


class ContactFeatureStream:
    def __init__(self, *, threshold: float = 0.0):
        self.previous = None
        self.threshold = threshold

    def reset(self):
        self.previous = None

    def update(self, pressure: np.ndarray):
        scalar, mcf = per_finger_features(pressure, self.previous, threshold=self.threshold)
        self.previous = np.asarray(pressure, dtype=np.float32).copy()
        return scalar, mcf
