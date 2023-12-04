import yaml
import numpy as np
import scipy.integrate
from wheel_model import wheel_factory

from trajectory import Trajectory
from constants import *

class DDMR:
    def __init__(self, name, config_filename) -> None:
        self.name = name
        with open(config_filename, 'r') as config_file:
            cfg = yaml.safe_load(config_file)
            self.L = 0.5 * cfg['baseline']
            self.left_wheel = wheel_factory(cfg['left_wheel'])
            self.right_wheel = wheel_factory(cfg['right_wheel'])
        #:
    #:

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

    def execute_control_policy(self, t, u, s0 = np.array([0, 0, 0, 0, 0]), name=''):
        sol = scipy.integrate.solve_ivp(dynamics,
                                        (t[0], t[-1]), s0,
                                        t_eval=t, max_step=kSimΔt,
                                        args=(t, kSimΔt, u, self))
        assert len(sol.t) == len(u) + 1
        s = sol.y.T
        assert len(s) == len(t)
        return Trajectory(sol.t, s, u, name)
    #:execute_control_policy

# class DDMR

class Dataset:
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
    #:__init__()
#:Dataset

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

def simulate(ddmr: DDMR):
    t = np.arange(0, 100, kSimΔt)
    φdots = np.zeros((len(t)-1, 2))
    φdots[:,0] = 2.0
    φdots[:,1] = 2.0
    s0 = np.array([0, 0, 0, 0, 0])
    sol = scipy.integrate.solve_ivp(dynamics, (0, len(t)*kSimΔt), s0, t_eval=t, max_step=kSimΔt, args=(t, kSimΔt, φdots, ddmr))
    print(f'IVP solution: {sol.success}')
    if sol.status != 0:
        print(f'solve_ivp failed because {sol.message}')
    #:
    trajectory = Trajectory(sol.t, sol.y.T, φdots, 'robot')
    trajectory.plot()
#:simulate()
