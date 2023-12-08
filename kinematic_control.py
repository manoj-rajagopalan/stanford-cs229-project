import numpy as np

from constants import *
from robot import DDMR

class KinematicController:
    def __init__(self, ideal_robot: DDMR, actual_robot: DDMR) -> None:
        self.ideal_robot = ideal_robot
        self.actual_robot = actual_robot
    #:__init__()

    def translate(self,
                  u_ideal: np.ndarray,
                  s:np.ndarray) \
         -> np.ndarray:
        '''
        u_ideal: ideal control which we would apply to the robot we think we have.
        s: measured state of the actual robot

        This function translates u_ideal into a u_actual so that the actual robot
        we have behaves like the idea one.
        '''

        v_desired, ω_desired = u_ideal

        # Now compute what actual φdot_l and φdot_r will provide these.
        _, _, _, φ_l, φ_r = s # x, y, θ not used
        L_actual = self.actual_robot.L
        v_r_actual = v_desired + (L_actual * ω_desired)
        v_l_actual = v_desired - (L_actual * ω_desired)
        R_l_actual = self.actual_robot.left_wheel.radius_at(φ_l)
        R_r_actual = self.actual_robot.right_wheel.radius_at(φ_r)
        φdot_l_actual = v_l_actual / R_l_actual
        φdot_r_actual = v_r_actual / R_r_actual
        u_actual = np.array([φdot_l_actual, φdot_r_actual])
        return u_actual

    #:translate()

    '''
    def execute_control_policy(self, t, Δt, u, s0 = np.array([0, 0, 0, 0, 0]), name=''):
        sol = scipy.integrate.solve_ivp(dynamics,
                                        (t[0], t[-1]), s0,
                                        t_eval=t, max_step=kSimΔt,
                                        args=(t, Δt, u, self))
        assert sol.success
        assert len(sol.t) == len(u) + 1
        s = sol.y.T
        assert len(s) == len(t)
        return Trajectory(sol.t, s, u, name)
    #:execute_control_policy

    def execute_trajectory(self,
                           orig_trajectory: Trajectory,
                           new_trajectory_name) -> Trajectory:
        # Execute the control policy in the original trajectory to
        # generate a new one.
        Δt = orig_trajectory.t[1] - orig_trajectory.t[0]
        return self.execute_control_policy(orig_trajectory.t, Δt,
                                           orig_trajectory.u,
                                           orig_trajectory.s[0],
                                           new_trajectory_name)
    #:execute_trajectory()
    '''

#:KinematicController

class KinematicallyControlledDDMR(DDMR):
    def __init__(self, config_filename, ideal_robot, verbose=False) -> None:
        super().__init__(config_filename)
        self.name += '-kinCtl'
        self.controller = KinematicController(ideal_robot, self)
        self.verbose = False
    #:

    def wheel_dynamics(self, s, φdots_ideal):
        φdots_actual = self.controller.translate(φdots_ideal, s)
        sdot_ideal = self.controller.ideal_robot.wheel_dynamics(s, φdots_ideal)
        sdot_actual = super().wheel_dynamics(s, φdots_actual)
        if self.verbose:
            x, y, θ, φ_l, φ_r = s
            print(f'Dynamics @ ({x:0.3},{y:0.3};{θ:0.3}) | ({φ_l:0.3},{φ_r:0.3}):')
            print(f'    φdots = {φdots_actual[0]:0.5}, {φdots_actual[1]:0.5} vs {φdots_ideal[0]:0.5}, {φdots_ideal[1]:0.5}')
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
