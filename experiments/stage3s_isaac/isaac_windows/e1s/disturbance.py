"""Physical excitation math only. No mapping from loads to tactile features."""
from dataclasses import dataclass, asdict
import numpy as np


def rotation_wxyz(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass
class Trial:
    trial_id: str
    direction: str = '+Fx'
    amplitude_n: float = 0.2
    point_m: tuple = (0., 0., 0.)
    frame: str = 'world'
    waveform: str = 'step'
    repeat: int = 0
    preload_delta_rad: float = 0.
    friction_scale: float = 1.
    sdf_gain_scale: float = 1.
    onset_s: float = 0.25
    duration_s: float = 0.25
    torque_nm: tuple = (0., 0., 0.)

    def __post_init__(self):
        if self.direction not in ('+Fx', '-Fx', '+Fy', '-Fy', 'zero'):
            raise ValueError('Unknown direction')
        if self.frame not in ('world', 'object', 'reference'):
            raise ValueError('frame must be world, object or reference')
        if self.waveform not in ('step', 'ramp', 'reverse'):
            raise ValueError('Unknown waveform')
        if self.amplitude_n < 0 or self.onset_s < 0 or self.duration_s <= 0:
            raise ValueError('Invalid loading range')
        if self.friction_scale <= 0 or self.sdf_gain_scale <= 0:
            raise ValueError('Invalid material / gain scale')
        if not np.isfinite(list(self.point_m)+list(self.torque_nm)+[
                self.amplitude_n, self.preload_delta_rad, self.friction_scale,
                self.sdf_gain_scale, self.onset_s, self.duration_s]).all():
            raise ValueError('Nonfinite trial')

    def gain(self, t):
        u = (t-self.onset_s)/self.duration_s
        if not 0 <= u < 1:
            return 0.
        if self.waveform == 'step':
            return 1.
        if self.waveform == 'ramp':
            return float(u)
        return 1. if u < .5 else -1.

    def wrench(self, t, link_pose, com_world, reference_rotation=None):
        """Point is an offset from COM in chosen frame; moment is about COM.

        World is inertial. Object follows current link orientation. Reference is
        a frozen, independently specified orthonormal coordinate system.
        """
        if self.frame == 'object':
            r = rotation_wxyz(link_pose[3:7])
        elif self.frame == 'reference':
            r = np.asarray(reference_rotation, dtype=float)
            if r.shape != (3,3) or not np.allclose(r.T@r, np.eye(3)) or not np.isclose(np.linalg.det(r),1):
                raise ValueError('Invalid reference rotation')
        else:
            r = np.eye(3)
        f = np.zeros(3)
        if self.direction != 'zero':
            f[0 if self.direction.endswith('x') else 1] = (1 if self.direction[0]=='+' else -1)*self.amplitude_n
        gain = self.gain(t)
        f_world = r@f*gain
        lever = r@np.asarray(self.point_m)
        moment = np.cross(lever, f_world) + r@np.asarray(self.torque_nm)*gain
        return f_world, moment, np.asarray(com_world)+lever

    def metadata(self):
        return asdict(self)


def pilot_matrix(repeats=3):
    """Sparse development pilot, not the formal full factorial experiment."""
    specs = [dict(direction='zero')]
    for direction in ('+Fx','-Fx','+Fy','-Fy'):
        for z in (-.015, 0., .015):
            specs.append(dict(direction=direction, point_m=(0.,0.,z)))
        for waveform in ('ramp','reverse'):
            specs.append(dict(direction=direction, waveform=waveform, point_m=(0.,0.,.015)))
    # One factor at a time, signs both retained. Gains are sensor-proxy gains,
    # explicitly NOT a deformable material stiffness intervention.
    for direction in ('+Fx','-Fx','+Fy','-Fy'):
        for key, values in (('preload_delta_rad',(-.001,.001)),
                            ('friction_scale',(.5,1.5)), ('sdf_gain_scale',(.5,1.5))):
            for value in values:
                specs.append(dict(direction=direction, **{key:value}))
    return [Trial(trial_id=f'c{c:03d}_r{r:02d}', repeat=r, **spec)
            for r in range(repeats) for c,spec in enumerate(specs)]
