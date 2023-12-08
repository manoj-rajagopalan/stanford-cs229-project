from ideal import *
from robot import *
from eval import rate_trajectory

from matplotlib import pyplot as plt

kResultsDir = 'Results/0-Ideal'

def test():
    ideal_robot = DDMR(config_filename='Robots/Ideal.yaml')
    ideal_robot.write_to_file(filename=f'{kResultsDir}/test-Ideal.yaml')

    ideal_trajectories = load_trajectories()
    for traj_wheel_dyn in ideal_trajectories:
        assert traj_wheel_dyn.u_type == Trajectory.Type.WHEEL_DYNAMICS
        print(f'Wheel-to-body check for robot {ideal_robot.name} with trajectory {traj_wheel_dyn.name}')
        v_ω = translate_control_wheel_to_body(traj_wheel_dyn,
                                              ideal_robot)
        traj_body_dyn = Trajectory(traj_wheel_dyn.t,
                                   traj_wheel_dyn.s,
                                   v_ω,
                                   Trajectory.Type.BODY_DYNAMICS,
                                   traj_wheel_dyn.name + '-wheel_to_body')
        wheel_dyn_traj = ideal_robot.execute_trajectory(traj_wheel_dyn,
                                                        'wheel_dynamics')
        body_dyn_traj = ideal_robot.execute_trajectory(traj_body_dyn,
                                                       'body_dynamics')
        _, ax = plt.subplots()
        ax.plot(wheel_dyn_traj.s[:,0], wheel_dyn_traj.s[:,1], 'r-')
        ax.plot(body_dyn_traj.s[:,0], body_dyn_traj.s[:,1], 'b-')
        plt.show()
        plt.close()

        Δs, Δs_per_unit_length, length = rate_trajectory(body_dyn_traj, wheel_dyn_traj)
        print(f'Evaluation summary for trajectory {body_dyn_traj.name}:')
        print(f'- Traj length  = {length[-1]} m')
        print(f'    max Δs     = {np.max(Δs)} at {length[np.argmax(Δs)]} m')
        print(f'    max Δs/len = {np.max(Δs_per_unit_length)} at {length[np.argmax(Δs_per_unit_length)]} m')
    #:for ideal_traj_wheel_dyn
#:test()

if __name__ == "__main__":
    test()
#:


