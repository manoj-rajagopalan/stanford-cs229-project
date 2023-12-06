import numpy as np
import os

import robot
from constants import *
from trajectory import Trajectory

kResultsDir = 'Results/1-Uncontrolled'

def load_ideal_trajectories():
    ideal_trajectories = []
    for traj_key in kIdealTrajectoryKeys:
        traj_name = traj_key + '-Ideal'
        ideal_trajectory = Trajectory(npz['t_'+traj_name],
                                      npz['s_'+traj_name],
                                      npz['u_'+traj_name],
                                      name=traj_name)
        ideal_trajectories.append(ideal_trajectory)
    #:
    return ideal_trajectories
#:load_ideal_trajectories()

def run_robot_with_controls(robot: robot.DDMR,
                            trajectories: list[Trajectory],
                            output_npz_filename: str) -> None:
    '''
    Run the given robot on the control policy within the given trajectory.
    Purpose is to later compare generated state with that in the trajectory.
    '''
    npz_items = {}
    for traj in trajectories:
        traj_name = traj.name.replace('Ideal', robot.name)
        print(f'Generating {traj_name}')
        Δt = traj.t[1] - traj.t[0] # pray this is good enough
        traj = robot.execute_control_policy(traj.t, Δt, traj.u, s0=traj.s[0],
                                            name=traj_name)
        traj.plot(kResultsDir)
        # https://stackoverflow.com/a/33878297
        npz_items['t_' + traj_name] = traj.t
        npz_items['s_' + traj_name] = traj.s
        npz_items['u_' + traj_name] = traj.u
    #:for
    np.savez(output_npz_filename, **npz_items)
#:run_robot_with_controls()

if __name__ == "__main__":
    os.makedirs(kResultsDir, exist_ok=True)
    ideal_robot = robot.DDMR(config_filename='Robots/Ideal.yaml')
    real_robots = [robot.DDMR(f'Robots/{name}.yaml') for name in kRobotNames]
    npz = np.load('Results/0-Ideal/ideal_trajectories.npz')
    ideal_trajectories = load_ideal_trajectories()
    for robot in real_robots:
        run_robot_with_controls(robot,
                                ideal_trajectories,
                                output_npz_filename=f'{kResultsDir}/real_robot_trajectories.npz')
    #:
#:__main__
