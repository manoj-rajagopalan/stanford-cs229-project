import numpy as np

from constants import *
from robot import DDMR

class KinematicController:
    def __init__(self,
                 name: str,
                 ideal_robot: DDMR) \
        -> None:

        self.name = name
        self.ideal_robot = ideal_robot # for debug logs
    #:__init__()

    def translate(self,
                  target_robot: DDMR,
                  v_ω_desired: np.ndarray,
                  s: np.ndarray) \
        -> np.ndarray: # φdots_lr
        '''
        u_ideal: ideal control which we would apply to the robot we think we have.
        s: measured state of the actual robot

        This function translates body-frame controls (v_ω_desired) into actual
        wheel-frame controls (φdot_l and φdot_r) so that the actual robot behaves
        like the ideal one.
        '''

        assert False, 'To be implemented by child classes'
    #:translate()

    def ideal_body_dynamics(self,
                            s: np.ndarray,
                            v_ω_desired: np.ndarray) \
        -> np.ndarray:
        return self.ideal_robot.body_dynamics(s, v_ω_desired)
    #:ideal_body_dynamics()

#:KinematicController


class KinematicallyControlledDDMR(DDMR):
    def __init__(self,
                 config_filename: str,
                 controller: KinematicController,
                 verbose:bool = False) \
        -> None:

        super().__init__(config_filename)
        self.controller = controller
        self.name += f'-kinCtl-{controller.name}'
        self.verbose = verbose
    #:

    # Override
    def wheel_dynamics(self, s, φdots_ideal):
        assert False, 'Should not be used'
    #:

    # Override
    def body_dynamics(self, s, vω_desired):
        φdots_actual = self.controller.translate(self, vω_desired, s)
        sdot_actual = super().wheel_dynamics(s, φdots_actual)
        sdot_ideal = self.controller.ideal_body_dynamics(s, vω_desired)
        if self.verbose:
            x, y, θ, φ_l, φ_r = s
            print(f'{self.name} dynamics @ ({x:0.3},{y:0.3};{θ:0.3}) | ({φ_l:0.3},{φ_r:0.3}):')
            print(f'    φdots = {φdots_actual[0]:0.5}, {φdots_actual[1]:0.5} vs {vω_desired[0]:0.5}, {vω_desired[1]:0.5}')
            print(f'    v = {np.hypot(sdot_actual[0], sdot_actual[1]):0.5} vs {np.hypot(sdot_ideal[0], sdot_ideal[1]):0.5}')
            print(f'    ω = {sdot_actual[2]:0.5} vs {sdot_ideal[2]:0.5}')
        #:if verbose
        return sdot_actual
    #:
#:KinematicallyControlledDDMR

'''
def dynamics(t: float,
             s: np.ndarray,
             time_points: np.ndarray,
             simΔt: float,
             φdots: np.ndarray,
             kin_ctlr: KinematicController,
             verbose: bool = False) -> np.ndarray:

    time_point_index = max(0, int((t-time_points[0]) / simΔt) - 1)
    while time_point_index < len(φdots) and t < time_points[time_point_index]:
        time_point_index += 1
    #:
    if time_point_index >= len(φdots):
        φdots_actual = np.array([0,0])
    else:
        φdots_ideal = φdots[time_point_index]
        φdots_actual = kin_ctlr.translate(φdots_ideal, s)
    #:
    s_dot_actual = kin_ctlr.actual_robot.dynamics(s, φdots_actual)
    s_dot_ideal = kin_ctlr.ideal_robot.dynamics(s, φdots_ideal)
    if verbose:
        print(f'Dynamics @ t={t:0.3},')
        print(f'    φdots = {φdots_actual[0]:0.5}, {φdots_actual[1]:0.5} vs {φdots_ideal[0]:0.5}, {φdots_ideal[1]:0.5}')
        print(f'    v = {np.hypot(s_dot_actual[0], s_dot_actual[1]):0.5} vs {np.hypot(s_dot_ideal[0], s_dot_ideal[1]):0.5}')
        print(f'    θ_dot = {s_dot_actual[2]:0.5} vs {s_dot_ideal[2]:0.5}')
    #:if verbose
    return s_dot_actual
#:dynamics()
'''
