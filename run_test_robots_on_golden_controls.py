import numpy as np

import robot
from trajectory import Trajectory

def run_test_robot_with_golden_controls(test_robot: robot.DDMR,
                                        golden_trajectories: list) -> ModuleNotFoundError:
    npz_items = {}
    for traj in golden_trajectories:
        traj_name = traj.name.replace('Golden', test_robot.name)
        print(f'Generating {traj_name}')
        traj = test_robot.execute_control_policy(traj.t, traj.u, s0=traj.s[0], name=traj_name)
        # must_print_markers = 'straight' not in traj.name \
        #                   and 'spin' not in traj.name \
        #                   and 'tri_wave_phi' not in traj.name
        traj.plot(must_print_markers=True)
        # https://stackoverflow.com/a/33878297
        npz_items['t_' + traj_name] = traj.t
        npz_items['s_' + traj_name] = traj.s
        npz_items['u_' + traj_name] = traj.u
    #:for
    np.savez('Results/test_robot_trajectories.npz', **npz_items)
#:run_test_robot_with_golden_controls()

if __name__ == "__main__":
    golden_robot = robot.DDMR(name='Golden', config_filename='Robots/golden.yaml')
    smaller_left_wheel_robot = robot.DDMR(name='SmallerLeftWheel', config_filename='Robots/smaller_left_wheel.yaml')
    larger_left_wheel_robot = robot.DDMR(name='LargerLeftWheel', config_filename='Robots/larger_left_wheel.yaml')
    noisy_robot = robot.DDMR(name='Noisy', config_filename='Robots/noisy.yaml')
    noisier_robot = robot.DDMR(name='Noisier', config_filename='Robots/noisier.yaml')
    test_robots = [smaller_left_wheel_robot, larger_left_wheel_robot, noisy_robot, noisier_robot]

    npz = np.load('Results/golden_trajectories.npz')
    traj_keys = ['straight', 'spin', 'circle_ccw', 'circle_cw', 'figureOf8', 'tri_wave_phi']
    golden_trajectories = []
    for traj_key in traj_keys:
        traj_name = traj_key + '-Golden'
        golden_trajectory = Trajectory(npz['t_'+traj_name],
                                         npz['s_'+traj_name],
                                         npz['u_'+traj_name],
                                         name=traj_name)
        golden_trajectories.append(golden_trajectory)
    #:
    for test_robot in test_robots:
        run_test_robot_with_golden_controls(test_robot, golden_trajectories)
    #:
