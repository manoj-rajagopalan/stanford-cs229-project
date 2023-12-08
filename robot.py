import yaml
import numpy as np
import scipy.integrate
from wheel_model import wheel_factory

from trajectory import Trajectory
from constants import *


class DDMR:
    def __init__(self, config_filename:str = None) -> None:
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

    def wheel_dynamics(self,
                       s: np.ndarray,
                       φ_dots: np.ndarray) \
        -> np.ndarray:

        _, _, θ, φ_l, φ_r = s # x and y not used
        φdot_l, φdot_r = φ_dots
        v_l = self.left_wheel.v(φ_l, φdot_l)
        v_r = self.right_wheel.v(φ_r, φdot_r)
        v = (v_r + v_l) / 2
        x_dot, y_dot = v * np.cos(θ), v * np.sin(θ)
        ω = (v_r - v_l) / (2 * self.L)
        sdot = np.array([x_dot, y_dot, ω, φdot_l, φdot_r])
        return sdot
    #: wheel_dynamics()

    def execute_wheel_control_policy(self,
                                     t: np.ndarray,
                                     Δt: float,
                                     φdot_lr: np.ndarray,
                                     s0: np.ndarray = np.array([0, 0, 0, 0, 0]),
                                     name: str ='') \
                -> Trajectory:

        sol = scipy.integrate.solve_ivp(wheel_dynamics,
                                        (t[0], t[-1]), s0,
                                        t_eval=t, max_step=kSimΔt,
                                        args=(t, Δt, φdot_lr, self))
        assert len(sol.t) == len(φdot_lr) + 1
        s = sol.y.T
        assert len(s) == len(t)
        return Trajectory(sol.t, s, u=φdot_lr,
                          u_type=Trajectory.Type.WHEEL_DYNAMICS,
                          name=name)
    #:execute_wheel_control_policy()

    def body_dynamics(self,
                      s: np.ndarray,
                      v_ω: np.ndarray) \
        -> np.ndarray:

        _, _, θ, φ_l, φ_r = s # x and y not used
        v, ω = v_ω
        v_l = v - self.L * ω
        v_r = v + self.L * ω
        x_dot, y_dot = v * np.cos(θ), v * np.sin(θ)
        φdot_l = v_l / self.left_wheel.radius_at(φ_l)
        φdot_r = v_r / self.right_wheel.radius_at(φ_r)
        sdot = np.array([x_dot, y_dot, ω, φdot_l, φdot_r])
        return sdot
    #:body_dynamics()

    def execute_body_control_policy(self,
                                    t: np.ndarray,
                                    Δt: float,
                                    v_ω: np.ndarray,
                                    s0: np.ndarray = np.array([0, 0, 0, 0, 0]),
                                    name: str ='') \
        -> Trajectory:

        sol = scipy.integrate.solve_ivp(body_dynamics,
                                        (t[0], t[-1]), s0,
                                        t_eval=t, max_step=kSimΔt,
                                        args=(t, Δt, v_ω, self))
        assert len(sol.t) == len(v_ω) + 1
        s = sol.y.T
        assert len(s) == len(t)
        return Trajectory(sol.t, s, u=v_ω,
                          u_type=Trajectory.Type.BODY_DYNAMICS,
                          name=name)
    #:execute_body_control_policy()

    def execute_trajectory(self,
                           orig_trajectory: Trajectory,
                           new_trajectory_name) \
        -> Trajectory:
        '''
        Execute the control policy in the original trajectory to
        generate a new one.
        '''
        Δt = orig_trajectory.t[1] - orig_trajectory.t[0]
        if orig_trajectory.u_type == Trajectory.Type.WHEEL_DYNAMICS:
            return self.execute_wheel_control_policy(orig_trajectory.t, Δt,
                                                     orig_trajectory.u,
                                                     orig_trajectory.s[0],
                                                     new_trajectory_name)
        else:
            assert orig_trajectory.u_type == Trajectory.Type.BODY_DYNAMICS
            return self.execute_body_control_policy(orig_trajectory.t, Δt,
                                                    orig_trajectory.u,
                                                    orig_trajectory.s[0],
                                                    new_trajectory_name)
        #:if
    #:execute_trajectory()

# class DDMR

def wheel_dynamics(t: float,
                   s: float,
                   time_points: np.ndarray,
                   simΔt: float,
                   φdot_lr: np.ndarray,
                   robot: DDMR,
                   verbose: bool = False):

    time_point_index = max(0, int((t-time_points[0]) / simΔt) - 1)
    while time_point_index < len(φdot_lr) and t < time_points[time_point_index]:
        time_point_index += 1
    #:
    if time_point_index >= len(φdot_lr):
        u = np.array([0,0])
    else:
        u = φdot_lr[time_point_index]
    #:
    s_dot = robot.wheel_dynamics(s, u)
    if verbose:
        print(f'Wheel dynamics @ t={t:0.3}, v={np.hypot(s_dot[0], s_dot[1])}, ω={s_dot[2]:0.3}, φdot_l={u[0]:0.3}, φdot_r={u[1]:0.3}')
    return s_dot
#: wheel_dynamics()

def body_dynamics(t: float,
                  s: float,
                  time_points: np.ndarray,
                  simΔt: float,
                  v_ω: np.ndarray,
                  robot: DDMR,
                  verbose: bool = False):

    time_point_index = max(0, int((t-time_points[0]) / simΔt) - 1)
    while time_point_index < len(v_ω) and t < time_points[time_point_index]:
        time_point_index += 1
    #:
    if time_point_index >= len(v_ω):
        u = np.array([0,0])
    else:
        u = v_ω[time_point_index]
    #:
    s_dot = robot.body_dynamics(s, u)
    if verbose:
        print(f'Body dynamics @ t={t:0.3}, φdot_l={s_dot[3]:0.3}, φdot_r={s_dot[4]:0.3}, v={s_dot[0]:0.3}, ω={s_dot[1]:0.3}')
    return s_dot
#: body_dynamics()


def translate_control_body_to_wheel(traj_body_dyn: Trajectory,
                                    robot: DDMR) \
    -> np.ndarray:

    assert traj_body_dyn.u_type == Trajectory.Type.BODY_DYNAMICS
    φdot_lr = np.zeros_like(traj_body_dyn.u)
    for i in range(len(traj_body_dyn.u)):
        s = traj_body_dyn.s[i]
        v_ω = traj_body_dyn.u[i]
        sdot = robot.body_dynamics(s, v_ω)
        φdot_lr[i] = sdot[-2:]
    #:for i
    return φdot_lr
#:translate_control_body_to_wheel()


def translate_control_wheel_to_body(traj_wheel_dyn: Trajectory,
                                    robot: DDMR) \
    -> np.ndarray:

    assert traj_wheel_dyn.u_type == Trajectory.Type.WHEEL_DYNAMICS
    v_ω = np.zeros_like(traj_wheel_dyn.u)
    for i in range(len(traj_wheel_dyn.u)):
        s = traj_wheel_dyn.s[i]
        φdot_lr = traj_wheel_dyn.u[i]
        sdot = robot.wheel_dynamics(s, φdot_lr)
        v_ω[i,0] = np.hypot(sdot[0], sdot[1])
        if v_ω[i,0] != 0:
            # Check if moving backwards.
            θ_v_deg = np.mod(np.rad2deg(np.arctan2(sdot[1], sdot[0])), 360)
            θ_deg = np.mod(np.rad2deg(s[2]), 360)

            if np.abs(θ_deg - θ_v_deg) > 0.1:
                # Moving backwards.
                assert 179.9 < np.abs(θ_deg - θ_v_deg) < 180.1
                v_ω[i,0] = -v_ω[i,0]
            #:if
        #:if
        v_ω[i,1] = sdot[2]
    #:for i
    return v_ω
#:translate_control_wheel_to_body()
