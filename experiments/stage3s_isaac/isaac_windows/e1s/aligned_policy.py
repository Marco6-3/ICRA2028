"""Preload-subtracted frozen students and causal CoP trigger interventions."""
import numpy as np
from reflex_policy import Controller, MAX_DELTA


class AlignedController:
    def __init__(self, kind, dt, checkpoint, base_close, trigger='none', release_s=.45):
        self.kind, self.dt, self.base_close = kind, dt, base_close
        self.student = Controller(kind if kind != 'fixed' else 'nominal', dt, checkpoint)
        self.trigger, self.release_s = trigger, release_s
        self.pre_prediction = 0.
        self.samples = []
        self.center = self.threshold = None
        self.consecutive = 0
        self.triggered_at = None
        self.fired = False

    def step(self, mcf, t):
        # Only tactile input enters the frozen student; zero debug truth.
        prediction, _, _, _ = self.student.step(mcf, np.zeros(6), 0.)
        cop = np.asarray(mcf[2:4, 1:3], dtype=float)
        if t < self.release_s - 1e-9:
            self.pre_prediction = prediction
            if t >= .20:
                self.samples.append(cop.copy())
            return 0., prediction, False, 0.
        if self.center is None:
            samples = np.stack(self.samples)
            self.center = np.median(samples, axis=0)
            scales = 1.4826 * np.median(abs(samples - self.center), axis=0)
            self.threshold = np.maximum(1e-6, 6 * np.linalg.norm(scales, axis=1))
        score = float(np.max(np.linalg.norm(cop - self.center, axis=1) / self.threshold))
        crossed = score > 1.
        self.consecutive = self.consecutive + 1 if crossed else 0
        required = 1 if self.trigger == 'pulse' else 3
        # One trigger per episode prevents unbounded retriggering / silent tuning.
        if self.trigger != 'none' and not self.fired and self.consecutive >= required:
            self.triggered_at = t
            self.fired = True
        residual = prediction - self.pre_prediction if self.kind != 'fixed' else 0.
        if self.triggered_at is not None:
            elapsed = t - self.triggered_at
            hold = .025 if self.trigger == 'pulse' else .25
            # Pulse returns to the student; latch has a fixed causal decay.
            pulse = (MAX_DELTA - self.base_close) if elapsed < hold - 1e-9 else 0.
            if self.trigger == 'latch' and elapsed >= hold:
                pulse = (MAX_DELTA - self.base_close) * np.exp(-(elapsed - hold) / .15)
            residual = max(residual, pulse)
        residual = float(np.clip(residual, -self.base_close, MAX_DELTA - self.base_close))
        return residual, prediction, self.triggered_at is not None, score
