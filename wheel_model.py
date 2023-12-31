import numpy as np
import sys

from matplotlib import pyplot as plt

class WheelModel:
    def v(self):
        assert False, 'Not implemented'
    #:

    def radius_at(self, φ: float) -> float:
        assert False, 'Not implemented'
    #:

    def write_to_file(self, name: str, file=sys.stdout) -> None:
        assert False, 'Not implemented'
    #:
#:WheelModel

class SimpleWheel(WheelModel):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.R = radius
    #:

    def v(self, φ: float, φdot_rads_per_sec: float) -> float:
        return self.R * φdot_rads_per_sec
    #:

    def radius_at(self, φ: float) -> float:
        return self.R
    #:

    def write_to_file(self, name: str, file=sys.stdout) -> None:
        print(f'{name}:', file=file)
        print(f'    type: simple', file=file)
        print(f'    radius: self.R', file=file)
    #:
#:SimpleWheel

class NoisyWheel(WheelModel):
    def __init__(self, radius: float, perturbations: list) -> None:
        super().__init__()
        # each perturbation is a record with entries
        # - scale: fraction relative to (nonminal) radius
        # - angular_position (deg)
        # - angular_extent (deg)
        self.R = radius # for writing out later
        φs_deg = []
        Rs = []
        φ_deg = 0.0
        for i, p in enumerate(perturbations):
            if p['angular_position_deg'] > φ_deg:
                φs_deg.append(φ_deg)
                Rs.append(radius)
            #:if
            φ_deg = p['angular_position_deg']
            φs_deg.append(φ_deg)
            Rs.append(radius * p['scale'])
            φ_deg += p['angular_extent_deg']
        #: for
        if φ_deg < 360:
            φs_deg.append(φ_deg)
            Rs.append(radius)
        #:if
        self.φs_deg = np.array(φs_deg)
        self.Rs = np.array(Rs)
    #:__init__()

    def radius_at(self, φ: float) -> float:
        i = 0
        φ_deg = np.mod(np.rad2deg(φ), 360)
        while i < len(self.φs_deg) and self.φs_deg[i] <= φ_deg:
            i += 1
        #:
        i -= 1
        assert 0 <= i < len(self.φs_deg)
        r = self.Rs[i]
        return r
    #:radius_at()

    def v(self, φ: float, φdot_rad_per_sec: float) -> float:
        r = self.radius_at(φ)
        v = r * φdot_rad_per_sec
        return v
    #:v()

    def write_to_file(self, name: str, file=sys.stdout) -> None:
        print(f'{name}:', file=file)
        print(f'    type: noisy', file=file)
        print(f'    radius: {self.R}', file=file)
        print(f'    perturbations:', file=file)
        for i in range(len(self.Rs)):
            φ_deg_next = 360 if i == (len(self.Rs) - 1) else self.φs_deg[i+1]
            scale = self.Rs[i] / self.R
            Δφ_deg = φ_deg_next - self.φs_deg[i]
            print(f'        - angular_position_deg: {self.φs_deg[i]}', file=file)
            print(f'          angular_extent_deg: {Δφ_deg}', file=file)
            print(f'          scale: {scale}', file=file)
        #:for
    #:write_to_file()

#: class NoisyWheel

def wheel_factory(params):
    wheel = None
    if params['type'] == 'simple':
        wheel = SimpleWheel(params['radius'])

    elif params['type'] == 'noisy':
        wheel = NoisyWheel(params['radius'], params['perturbations'])
    #:

    assert wheel is not None
    return wheel
#:

