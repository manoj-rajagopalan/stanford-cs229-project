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
        cosθ = np.cos(θ)
        sinθ = np.sin(θ)
        R = np.array([[cosθ, -sinθ], [sinθ, cosθ]])
        v_l = self.left_wheel.v(φ_l, φdot_l)
        v_r = self.right_wheel.v(φ_r, φdot_r)
        xdot_ydot_robot_frame = np.array([0.5*(v_r+v_l), 0.0])
        x_dot, y_dot = R @ xdot_ydot_robot_frame
        θ_dot = (v_r - v_l) / (2 * self.L)
        return np.array([x_dot, y_dot, θ_dot, φdot_l, φdot_r])
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
        -> Trajectory:

        _, _, θ, φ_l, φ_r = s # x and y not used
        v, ω = v_ω
        cosθ = np.cos(θ)
        sinθ = np.sin(θ)
        R = np.array([[cosθ, -sinθ], [sinθ, cosθ]])
        v_l = v - self.L * ω
        v_r = v + self.L * ω
        φdot_l = v_l / self.left_wheel.radius_at(φ_l)
        φdot_r = v_r / self.right_wheel.radius_at(φ_l)
        xdot_ydot_robot_frame = np.array([0.5*(v_r+v_l), 0.0])
        x_dot, y_dot = R @ xdot_ydot_robot_frame
        θ_dot = (v_r - v_l) / (2 * self.L)
        return np.array([x_dot, y_dot, θ_dot, φdot_l, φdot_r])
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


# Tests
if __name__ == "__main__":
    noisier_robot = DDMR(config_filename='Robots/noisier.yaml')
    noisier_robot.write_to_file(filename='Results/test-noisier.yaml')
