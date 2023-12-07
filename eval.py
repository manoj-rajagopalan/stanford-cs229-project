import numpy as np

import robot
from trajectory import Trajectory

def run_robots_with_controls(robots: list[robot.DDMR],
                            trajectories: list[Trajectory],
                            output_dirname: str) -> None:
    '''
    Run the given robot on the control policy within the given trajectory.
    Purpose is to later compare generated state with that in the trajectory.
    '''
    npz_items = {}
    for robot in robots:
        for orig_traj in trajectories:
            new_traj_name = orig_traj.name + '-by-' + robot.name
            print(f'Generating {new_traj_name}')
            new_traj = robot.execute_trajectory(orig_traj, new_traj_name)
            new_traj.plot(output_dirname)
            npz_items['t_'+orig_traj.name] = orig_traj.t
            npz_items['s_'+orig_traj.name] = orig_traj.s
            npz_items['s_'+new_traj.name] = new_traj.s
            npz_items['u_'+orig_traj.name] = orig_traj.u
        #:for orig_traj
    #:for robot
    np.savez(f'{output_dirname}/run_robots_with_controls.npz',
             **npz_items)
#:run_robot_with_controls()
