import numpy as np
import matplotlib
matplotlib.use('Agg') # suppress popup windows
from matplotlib import pyplot as plt

import ideal
import robot
from trajectory import Trajectory
from kinematic_control import KinematicallyControlledDDMR

# using namespace
DDMR = robot.DDMR

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
            print(f'Generating {new_traj_name}', flush=True)
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
                    ref_trajectory: Trajectory) \
    -> (float, np.ndarray):
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

def evaluate_trajectories(new_trajectories: list[list[Trajectory]],
                          ref_trajectories: list[list[Trajectory]],
                          real_robots: list[robot.DDMR],
                          results_dir: str) \
    -> None:

    assert len(new_trajectories) == len(ref_trajectories) == len(real_robots)
    for i in range(len(real_robots)):
        new_trajs_per_robot = new_trajectories[i]
        ref_trajs_per_robot = ref_trajectories[i]
        real_robot = real_robots[i]
        print('--------------------------------------')
        print(f'Evaluating robot {real_robot.name}')
        print('--------------------------------------', flush=True)

        for new_traj, ref_traj in zip(new_trajs_per_robot, ref_trajs_per_robot):
            Δs, Δs_per_unit_length, length = \
                rate_trajectory(new_traj, ref_traj)

            _, ax = plt.subplots()
            ax.plot(ref_traj.s[:,0], ref_traj.s[:,1], 'r-')
            ax.plot(new_traj.s[:,0], new_traj.s[:,1], 'k-')
            ax.set_xlabel('$x$ (m)')
            ax.set_ylabel('$y$ (m)')
            ax.set_title(f'{ref_traj.name} with sys-id control')
            plt.savefig(f'{results_dir}/{new_traj.name}-overlap.png')
            plt.close()

            _, ax = plt.subplots()
            ax.plot(length, Δs)
            ax.set_xlabel('Trajectory length (m)')
            ax.set_ylabel('Separation from ref traj (m)')
            plt.savefig(f'{results_dir}/{new_traj.name}-eval-sep.png')
            plt.close()

            print(f'Evaluation summary for trajectory {new_traj.name} w.r.t. {ref_traj.name}:')
            print(f'- Traj length  = {length[-1]} m')
            print(f'- max Δs     = {np.max(Δs)} at {length[np.argmax(Δs)]} m')
            print(f'- max Δs/len = {np.max(Δs_per_unit_length)} at {length[np.argmax(Δs_per_unit_length)]} m')
        #:for
    #:for i
#:evaluate_trajectories()

def evaluate_controlled_trajectories(real_robots: list[DDMR],
                                     controlled_robots: list[KinematicallyControlledDDMR],
                                     results_dir: str) \
    -> None:
    '''
    Apply the kinematic controller to all learnt robots
    and hope that we match the ideal robot.

    Here, we provide body-dynamical controls (v,ω) because this
    is what the real-world use-case will provide.
    '''

    assert len(real_robots) == len(controlled_robots)

    ideal_robot = DDMR('Robots/Ideal.yaml')
    # ideal_trajectories = ideal.load_trajectories(['figureOf8'])
    ideal_trajectories = ideal.load_trajectories()

    # Convert from wheel-dynamical trajectory to body-dynamical trajectory
    ideal_trajectories_wheel_to_body = []
    for ideal_traj_wheel_dyn in ideal_trajectories:
        assert ideal_traj_wheel_dyn.u_type == Trajectory.Type.WHEEL_DYNAMICS
        v_ω = robot.translate_control_wheel_to_body(ideal_traj_wheel_dyn,
                                                    ideal_robot)
        ideal_traj_body_dyn = Trajectory(ideal_traj_wheel_dyn.t,
                                         ideal_traj_wheel_dyn.s,
                                         v_ω,
                                         Trajectory.Type.BODY_DYNAMICS,
                                         name=ideal_traj_wheel_dyn.name + '-wheel_to_body')
        ideal_trajectories_wheel_to_body.append(ideal_traj_body_dyn)
    #:for

    # Evaluate if learning-based controller can match ideal robot given its
    # desired body-dynamical trajectory.
    new_trajectories = \
        run_robots_with_controls(controlled_robots,
                                 ideal_trajectories_wheel_to_body,
                                 results_dir)
    evaluate_trajectories(new_trajectories,
                          [ideal_trajectories] * len(real_robots),
                          real_robots,
                          results_dir)
#:evaluate_controlled_trajectories()
