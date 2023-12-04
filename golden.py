import numpy as np
import os

import robot
from trajectory import Trajectory
from constants import *

'''
def generate_golden_trajectory_straight_line(golden_robot: robot.DDMR, simΔt):
    v = 0.02 # m/s, speed
    φdot = v / golden_robot.left_wheel.R # rad/s
    d = 5.0 # m, distance to be travelled
    t_final = d / v
    s0 = [0, 0, 0, 0, 0] # initial state
    num_time_steps = t_final / simΔt
    t = np.arange(num_time_steps+1) * simΔt
    φdots = np.full((len(t)-1, 2), φdot)
    sol = scipy.integrate.solve_ivp(robot.dynamics, [0,t_final], s0, t_eval=t, max_step=simΔt,
                                    args=(t, simΔt, φdots, golden_robot))
    assert sol.success == True
    return Trajectory(sol.t, sol.y.T, φdots, 'golden_straight')
#:generate_golden_trajectory_straight_line()
'''

def generate_golden_trajectory_straight_line(golden_robot: robot.DDMR) -> Trajectory:
    '''
    Accelerate both wheels simultaneously to max φdot and decelerate to zero: move forward in straight line.
    Then reverse the wheel-angular-velocities; retrace straight line back to origin.
    '''
    t_half = 8 # seconds
    t_up = np.linspace(0, t_half, int(t_half)*kSimHz + 1) # 0.01 sec intervals
    φdot_rate = φdot_max_mag_rps * 2*np.pi / t_half
    u_up = np.vstack((t_up[1:], t_up[1:])).T * φdot_rate
    t_down = t_up[-1] + t_up[1:]
    u_down = np.vstack((u_up[-2:-len(u_up)-1:-1],
                        np.zeros((1,2))))
    t = np.hstack((t_up, t_down))
    u = np.vstack((u_up, u_down))
    assert len(t) == len(u) + 1

    # So far, acceleration and deceleration in the forward direction.
    # Now reverse to start.
    t = np.hstack((t, t[-1] + t[1:]))
    u = np.vstack((u, -u))
    assert len(t) == len(u) + 1

    # Simulate trajectory
    s0 = np.array([0,0,0,0,0])
    trajectory = golden_robot.execute_control_policy(t, u, s0, name=f'straight-{golden_robot.name}')
    return trajectory
#:generate_golden_trajectory_straight_line()


def generate_golden_trajectory_spin(golden_robot: robot.DDMR) -> Trajectory:
    '''
    Spin both wheels with equal and opposite angular velocity to rotate in-place.
    Accelerate to max angular velocity and then decelerate to zero to achieve net CCW rotation.
    Then reverse the controls to spin CW with same acceleration an deceleration.
    '''
    t_half = 8 # seconds
    t_up = np.linspace(0, t_half, int(t_half)*kSimHz + 1) # 0.01 sec intervals
    φdot_rate = φdot_max_mag_rps/2 * 2*np.pi / t_half
    u_up = np.vstack((t_up[1:], t_up[1:])).T * np.array([[-φdot_rate, φdot_rate]])
    t_down = t_up[-1] + t_up[1:]
    u_down = np.vstack((u_up[-2:-len(u_up)-1:-1],
                        np.zeros((1,2))))
    t = np.hstack((t_up, t_down))
    u = np.vstack((u_up, u_down))
    assert len(t) == len(u) + 1

    # So far, acceleration and deceleration in the forward direction.
    # Now reverse to start.
    t = np.hstack((t, t[-1] + t[1:]))
    u = np.vstack((u, -u))
    assert len(t) == len(u) + 1

    # Simulate trajectory
    s0 = np.array([0,0,0,0,0])
    trajectory = golden_robot.execute_control_policy(t, u, s0, name=f'spin-{golden_robot.name}')
    return trajectory
#:generate_golden_trajectory_rotate_in_place()

def generate_golden_trajectory_circle_common(t_final: float,
                                             v: float,
                                             θdot: float,
                                             golden_robot: robot.DDMR) -> Trajectory:
    v_r = v + golden_robot.L * θdot
    v_l = v - golden_robot.L * θdot
    φ_r = v_r / golden_robot.right_wheel.R
    φ_l = v_l / golden_robot.left_wheel.R
    num_time_steps = t_final / kSimΔt
    t = np.arange(num_time_steps + 1) * kSimΔt
    t[-1] = min(t_final, t[-1]) # placate IVP solver
    φdots = np.repeat(np.array([[φ_l, φ_r]]), len(t)-1, axis=0)
    s0 = np.array([0,0,0,0,0])
    trajectory = golden_robot.execute_control_policy(t, φdots, s0, name='')
    return trajectory
    # sol = scipy.integrate.solve_ivp(robot.dynamics, [0,t_final], s0, t_eval=t, max_step=kSimΔt,
    #                                 args=(t, kSimΔt, φdots, golden_robot))
    # return Trajectory(sol.t, sol.y.T, φdots, '')
#:generate_golden_trajectory_circle_common()


def generate_golden_trajectory_circle_ccw(golden_robot: robot.DDMR,
                                          R: float,
                                          v: float) -> Trajectory:
    d = 2 * np.pi * R # m, distance to travel
    t_final = d / v # s
    θdot = 2 * np.pi / t_final
    trajectory = generate_golden_trajectory_circle_common(t_final, v, θdot, golden_robot)
    trajectory.name = f'circle_ccw-{golden_robot.name}'
    return trajectory
#:generate_golden_trajectory_circle_ccw()

def generate_golden_trajectory_circle_cw(golden_robot: robot.DDMR,
                                         R: float,
                                         v: float) -> Trajectory:
    d = 2 * np.pi * R # m, distance to travel
    t_final = d / v # s
    θdot = -2 * np.pi / t_final
    trajectory = generate_golden_trajectory_circle_common(t_final, v, θdot, golden_robot)
    trajectory.name = f'circle_cw-{golden_robot.name}'
    return trajectory
#:generate_golden_trajectory_circle_ccw()

def generate_golden_trajectory_figureOf8(golden_robot: robot.DDMR,
                                         R: float,
                                         v: float) -> Trajectory:
    traj_ccw = generate_golden_trajectory_circle_ccw(golden_robot, R, v)
    traj_cw = generate_golden_trajectory_circle_cw(golden_robot, R, v)
    t = np.hstack((traj_ccw.t, traj_cw.t[1:] + traj_ccw.t[-1]))
    assert len(t) == len(traj_ccw.t) + len(traj_cw.t) - 1
    s = np.vstack((traj_ccw.s, traj_cw.s[1:,:] + traj_ccw.s[-1,:][np.newaxis,:]))
    assert s.shape[0] == len(t)
    u = np.vstack((traj_ccw.u, traj_cw.u))
    assert u.shape[0] == s.shape[0]-1
    trajectory = Trajectory(t, s, u, name=f'figureOf8-{golden_robot.name}')
    return trajectory
#:generate_golden_trajectory_figureOf8()

def generate_golden_trajectory_tri_wave_φ(golden_robot: robot.DDMR) -> Trajectory:
    t_cycle = int(32) # seconds
    φdot_r_rate = φdot_max_mag_rps * 2*np.pi / (t_cycle // 4)
    φdot_l_rate = 2 * φdot_r_rate
    t = np.linspace(0, t_cycle, kMeasurementHz * t_cycle + 1)
    u = np.zeros((len(t)-1, 2))

    num_qtr_cycle_samples = len(u) // 4
    u[:num_qtr_cycle_samples, 1] = t[1:num_qtr_cycle_samples+1] * φdot_r_rate
    u[num_qtr_cycle_samples : 2*num_qtr_cycle_samples-1, 1] = \
        u[range(num_qtr_cycle_samples-2,-1,-1), 1]
    u[2*num_qtr_cycle_samples-1, 1] = 0.0
    u[2*num_qtr_cycle_samples:, 1] = -u[:2*num_qtr_cycle_samples, 1]

    num_8th_cycle_samples = len(u) // 8
    u[:num_8th_cycle_samples, 0] = t[1:num_8th_cycle_samples+1] * φdot_l_rate
    u[num_8th_cycle_samples : 2*num_8th_cycle_samples-1, 0] = \
        u[range(num_8th_cycle_samples-2,-1,-1), 0]
    u[2*num_8th_cycle_samples-1, 0] = 0.0
    u[2*num_8th_cycle_samples : 4*num_8th_cycle_samples, 0] = -u[:2*num_8th_cycle_samples, 0]
    u[4*num_8th_cycle_samples:, 0] = u[:4*num_8th_cycle_samples, 0]

    s0 = np.array([0,0,0,0,0])
    trajectory = golden_robot.execute_control_policy(t, u, s0, name=f'tri_wave_phi-{golden_robot.name}')
    return trajectory
#:generate_golden_trajectory_tri_wave_φ()

def generate_trajectories(golden_robot: robot.DDMR) -> dict:
    print('Generating golden straight')
    golden_traj_straight = \
        generate_golden_trajectory_straight_line(golden_robot)
    golden_traj_straight.plot(must_print_markers=True)

    print('Generating golden spin')
    golden_traj_spin = \
        generate_golden_trajectory_spin(golden_robot)
    golden_traj_spin.plot(must_print_markers=True)

    golden_circle_radius = 0.5 # m
    golden_circle_speed = 0.05 # m/s

    print('Generating golden circle_ccw')
    golden_traj_circle_ccw = \
        generate_golden_trajectory_circle_ccw(golden_robot,
                                              golden_circle_radius,
                                              golden_circle_speed)
    golden_traj_circle_ccw.plot()

    print('Generating golden circle_cw')
    golden_traj_circle_cw = \
        generate_golden_trajectory_circle_cw(golden_robot,
                                             golden_circle_radius,
                                             golden_circle_speed)
    golden_traj_circle_cw.plot()

    print('Generating golden figure-of-8')
    golden_traj_figureOf8 = \
        generate_golden_trajectory_figureOf8(golden_robot,
                                             golden_circle_radius,
                                             golden_circle_speed)
    golden_traj_figureOf8.plot()

    print('Generating golden tri_wave_phi')
    golden_traj_tri_wave_φ = generate_golden_trajectory_tri_wave_φ(golden_robot)
    golden_traj_tri_wave_φ.plot(must_print_markers=True)

    result = {}
    result['straight'] = golden_traj_straight
    result['spin'] = golden_traj_spin
    result['circle_ccw'] = golden_traj_circle_ccw
    result['circle_cw'] = golden_traj_circle_cw
    result['figureOf8'] = golden_traj_figureOf8
    result['tri_wave_phi'] = golden_traj_tri_wave_φ
    return result
#: generate_trajectories()

if __name__ == "__main__":
    os.makedirs('Results', exist_ok=True)

    golden_robot = robot.DDMR(name='Golden', config_filename='Robots/golden.yaml')
    golden_trajectories = generate_trajectories(golden_robot)
    # Save to .npz file
    # https://stackoverflow.com/a/33878297
    npz_items = {}
    for traj in golden_trajectories.values():
        npz_items['t_' + traj.name] = traj.t
        npz_items['s_' + traj.name] = traj.s
        npz_items['u_' + traj.name] = traj.u
    #:
    np.savez('Results/golden_trajectories.npz', **npz_items)
