import yaml
import numpy as np
import scipy.integrate
from wheel_model import wheel_factory

from trajectory import Trajectory
from constants import *

class DDMR:
    def __init__(self, config_filename = None) -> None:
        if config_filename is None:
            self.name = None
            self.L = None
            self.left_wheel = None
            self.right_wheel = None

        else:
            with open(config_filename, 'r') as config_file:
                cfg = yaml.safe_load(config_file)
                self.name = cfg['name']
                self.L = 0.5 * cfg['baseline']
                self.left_wheel = wheel_factory(cfg['left_wheel'])
                self.right_wheel = wheel_factory(cfg['right_wheel'])
            #:with
        #:if
    #:__init__()

    def write_to_file(self, filename: str) -> None:
        with open(filename, 'w') as file:
            print(f'name: {self.name}\n', file=file)
            print(f'baseline: {2*self.L}\n', file=file)
            self.left_wheel.write_to_file('left_wheel', file=file)
            print('', file=file)
            self.right_wheel.write_to_file('right_wheel', file=file)
            print('', file=file)
        #:with
    #:write_to_file()

    def dynamics(self, s, φ_dots):
        x, y, θ, φ_l, φ_r = s # x and y not used
        φdot_l, φdot_r = φ_dots
        cosθ = np.cos(θ)
        sinθ = np.sin(θ)
        R = np.array([[cosθ, -sinθ], [sinθ, cosθ]])
        v_l = self.left_wheel.v(φ_l, φdot_l)
        v_r = self.right_wheel.v(φ_r, φdot_r)
        xdot_ydot_robot_frame = np.array([0.5*(v_r+v_l), 0.0])
        x_dot, y_dot = R @ xdot_ydot_robot_frame
        θ_dot = (v_r - v_l) / (2 * self.L)
        return np.array([x_dot, y_dot, θ_dot, φdot_l, φdot_r])
    #: dynamics()

    def execute_control_policy(self, t, Δt, u, s0 = np.array([0, 0, 0, 0, 0]), name=''):
        sol = scipy.integrate.solve_ivp(dynamics,
                                        (t[0], t[-1]), s0,
                                        t_eval=t, max_step=kSimΔt,
                                        args=(t, Δt, u, self))
        assert len(sol.t) == len(u) + 1
        s = sol.y.T
        assert len(s) == len(t)
        return Trajectory(sol.t, s, u, name)
    #:execute_control_policy

    def execute_trajectory(self,
                           orig_trajectory: Trajectory,
                           new_trajectory_name) -> Trajectory:
        '''
        Execute the control policy in the original trajectory to
        generate a new one.
        '''
        Δt = orig_trajectory.t[1] - orig_trajectory.t[0]
        return self.execute_control_policy(orig_trajectory.t, Δt,
                                           orig_trajectory.u,
                                           orig_trajectory.s[0],
                                           new_trajectory_name)
    #:execute_trajectory()

# class DDMR

def dynamics(t, s, time_points, simΔt, φdots, robot, verbose = False):
    time_point_index = max(0, int((t-time_points[0]) / simΔt) - 1)
    while time_point_index < len(φdots) and t < time_points[time_point_index]:
        time_point_index += 1
    #:
    if time_point_index >= len(φdots):
        u = np.array([0,0])
    else:
        u = φdots[time_point_index]
    #:
    s_dot = robot.dynamics(s, u)
    if verbose:
        print(f'Dynamics @ t={t:0.3}, v={np.hypot(s_dot[0], s_dot[1])}, θ_dot={s_dot[2]:0.3}')
    return s_dot
#: dynamics()


# Tests
if __name__ == "__main__":
    noisier_robot = DDMR(config_filename='Robots/noisier.yaml')
    noisier_robot.write_to_file(filename='Results/test-noisier.yaml')
