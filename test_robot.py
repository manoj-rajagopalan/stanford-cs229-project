# testing
from matplotlib import pyplot as plt

import eval
import ideal
from robot import *
from trajectory import Trajectory

# Tests
if __name__ == "__main__":
    noisier_robot = DDMR(config_filename='Robots/noisier.yaml')
    noisier_robot.write_to_file(filename='Results/test-noisier.yaml')

    ideal_robot = DDMR(f'Robots/Ideal.yaml')
    ideal_trajectories = ideal.load_trajectories()
    test_robot_names = ['SmallerLeftWheel']
    test_robots = [DDMR(f'Robots/{name}.yaml') for name in test_robot_names]
    for real_robot in test_robots:
        for ideal_traj_wheel_dyn in ideal_trajectories:
            assert ideal_traj_wheel_dyn.u_type == Trajectory.Type.WHEEL_DYNAMICS
            print(f'Wheel-to-body check for robot {real_robot.name} with trajectory {ideal_traj_wheel_dyn.name}')
            wheel_dyn_traj = real_robot.execute_trajectory(ideal_traj_wheel_dyn,
                                                           'wheel_dynamics')
            body_dyn_ctl_for_wheel_dyn_traj = \
                translate_control_wheel_to_body(wheel_dyn_traj,
                                                real_robot)
            wheel_dyn_traj_body_equiv = Trajectory(wheel_dyn_traj.t,
                                                   wheel_dyn_traj.s,
                                                   body_dyn_ctl_for_wheel_dyn_traj,
                                                   Trajectory.Type.BODY_DYNAMICS,
                                                   wheel_dyn_traj.name + '-wheel_to_body')
            body_dyn_traj = real_robot.execute_trajectory(wheel_dyn_traj_body_equiv,
                                                          'body_dynamics')
            fig, ax = plt.subplots()
            ax.plot(wheel_dyn_traj.s[:,0], wheel_dyn_traj.s[:,1], 'r-')
            ax.plot(body_dyn_traj.s[:,0], body_dyn_traj.s[:,1], 'b-')
            plt.show()
            plt.close()

            Δs, Δs_per_unit_length, length = eval.rate_trajectory(body_dyn_traj, wheel_dyn_traj)
            print(f'Evaluation summary for trajectory {body_dyn_traj.name}:')
            print(f'- Traj length  = {length[-1]} m')
            print(f'    max Δs     = {np.max(Δs)} at {length[np.argmax(Δs)]} m')
            print(f'    max Δs/len = {np.max(Δs_per_unit_length)} at {length[np.argmax(Δs_per_unit_length)]} m')
        #:for ideal_traj_wheel_dyn
    #:for robot
#:__main__
