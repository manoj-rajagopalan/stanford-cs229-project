import numpy as np

import robot
from trajectory import Trajectory

def run_robots_with_controls(robots: list[robot.DDMR],
                            trajectories: list[Trajectory],
                            output_dirname: str) \
    -> list[list[Trajectory]]:

    '''
    Run the given robot on the control policy within the given trajectory.
    Purpose is to later compare generated state with that in the trajectory.
    '''
    new_trajectories = []
    npz_items = {}
    for robot in robots:
        new_trajectories.append([])
        for orig_traj in trajectories:
            new_traj_name = orig_traj.name + '-by-' + robot.name
            print(f'Generating {new_traj_name}')
            new_traj = robot.execute_trajectory(orig_traj, new_traj_name)
            new_traj.plot(output_dirname)
            new_trajectories[-1].append(new_traj)
            npz_items['t_'+orig_traj.name] = orig_traj.t
            npz_items['s_'+orig_traj.name] = orig_traj.s
            npz_items['s_'+new_traj.name] = new_traj.s
            npz_items['u_'+orig_traj.name] = orig_traj.u
        #:for orig_traj
    #:for robot
    np.savez(f'{output_dirname}/run_robots_with_controls.npz',
             **npz_items)
    return new_trajectories
#:run_robot_with_controls()


def rate_trajectory(trajectory: Trajectory,
                    ref_trajectory: Trajectory) -> (float, np.ndarray):
    '''
    Calculate the physical divergence of the trajectory from its reference at
    each time-step. Also provide this value per unit (cumulative) length to
    study the error as a function of trajectory-length.

    This involves only the x, y components of the states within the trajectory
    and excludes θ, φ_l, and φ_r but the latter would have contributed to the
    x, y over time. If the measured and reference (x,y) are close, this must
    have included indirect contributions from θ, φ_l, and φ_r.
    '''
    assert len(trajectory.t) == len(ref_trajectory.t)
    assert len(trajectory.s) == len(ref_trajectory.s)
    assert len(trajectory.u) == len(ref_trajectory.u)
    Δs = np.linalg.norm(trajectory.s[:,:2] - ref_trajectory.s[:,:2],
                        axis=1)
    length = np.zeros_like(Δs)
    Δs_per_unit_length = np.zeros_like(Δs)
    prev_xy = trajectory.s[0,:2]
    for i in np.arange(1,len(Δs), dtype=int):
        length[i] = length[i-1] + np.linalg.norm(trajectory.s[i,:2] - prev_xy)
        Δs_per_unit_length[i] = Δs[i] / length[i]
        prev_xy = trajectory.s[i,:2]
    #:for
    return Δs, Δs_per_unit_length, length

#:rate_trajectory()
