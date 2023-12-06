import numpy as np

import robot
import trajectory
from constants import *

Trajectory = trajectory.Trajectory # using namespace

kResultsDir = 'Results/2-Dataset'

def generate_dataset_trajectory(real_robot: robot.DDMR) -> (Trajectory, np.ndarray):
    '''
    Get the robot to perform `num_loop` figures-of-8 with various radii.
    '''
    print(f'\nGenerating dataset for {real_robot.name} robot')
    rng = np.random.default_rng()
    
    # Pre-generate samples of φdot_l and φdot_R to create figure-of-8 loops with
    φdot_samples = np.array([[1.0, 7.0],
                             [1.5, 6.5],
                             [2.0 ,6.0],
                             [2.5, 5.5],
                             [3.0, 5.0]]) # rotations/sec
    φdot_samples *= 2 * np.pi # convert to radians/sec

    s0 = np.array([0,0,0,0,0])
    t0 = 0.0
    traj_t = [t0]
    traj_s = [s0]
    traj_u = []
    aux_v_θdot = []

    for iloop in range(len(φdot_samples)):
        φdots = φdot_samples[iloop]
        for ibranch in range(2):
            # 1st branch goes CCW
            # 2nd branch goes CW due to the φdot swap at the end
            counter = 0
            while 0 <= s0[2] <= (2*np.pi):
                counter += 1
                tf = t0 + kSimΔt
                t = np.array([t0, tf])
                incremental_traj = \
                    real_robot.execute_control_policy(t, kSimΔt, φdots[np.newaxis,:], s0)
                t0 = incremental_traj.t[-1]
                s0 = incremental_traj.s[-1]

                # Aux info at state: v and θdot
                _, _, _, φ_l, φ_r = s0
                φdot_l, φdot_r = φdots
                v_l = real_robot.left_wheel.v(φ_l, φdot_l)
                v_r = real_robot.right_wheel.v(φ_r, φdot_r)
                v = 0.5 * (v_r + v_l)
                θdot = 0.5 / real_robot.L * (v_r - v_l)

                traj_u.append(φdots.copy())
                traj_t.append(t0.copy())
                traj_s.append(s0.copy())
                aux_v_θdot.append([v, θdot])

            #:while s0[2] in [0, 2π)

            print(f'- loop #{iloop}: φdots = {φdots}, {counter} iterations')

            # swap
            temp = φdots[0]
            φdots[0] = φdots[1]
            φdots[1] = temp

            # Discard last trajectory point which violates limits
            del traj_u[-1]
            del traj_t[-1]
            del traj_s[-1]
            del aux_v_θdot[-1]
            t0 = traj_t[-1]
            s0 = traj_s[-1]

        #:for ibranch
    #:for iloop

    traj_t = np.array(traj_t)
    traj_s = np.array(traj_s)
    traj_u = np.array(traj_u)
    aux_v_θdot = np.array(aux_v_θdot)

    assert traj_s.shape == (len(traj_t), 5)
    traj_s[:,2:] = np.mod(traj_s[:,2:], 2*np.pi) # make θ, φ_l, φ_r canonical
    assert traj_u.shape == (len(traj_t)-1, 2)
    assert aux_v_θdot.shape == (len(traj_u), 2)

    dataset_trajectory = Trajectory(traj_t, traj_s, traj_u, name=f'dataset-{real_robot.name}')
    dataset_trajectory.plot(kResultsDir)

    # Decimate and add noise
    dataset_trajectory_decimated, _ = trajectory.decimate(dataset_trajectory)
    dataset_trajectory_decimated.plot(kResultsDir)
    
    dataset_trajectory_measured = trajectory.add_noise(dataset_trajectory_decimated,
                                                       xy_std_dev=0.01,
                                                       θ_deg_std_dev=1,
                                                       φ_lr_deg_std_dev=1,
                                                       φdot_lr_deg_per_sec_std_dev=1)
    dataset_trajectory_measured.plot(kResultsDir)
    aux_v_θdot_decimated = aux_v_θdot[range(0, len(aux_v_θdot), 10)]
    aux_v_θdot_measured = aux_v_θdot_decimated \
                        + np.vstack((np.random.normal(loc=0, scale=0.01, size=len(aux_v_θdot_decimated)),
                                     np.random.normal(loc=0, scale=np.deg2rad(1), size=len(aux_v_θdot_decimated)))).T
    
    np.savez(f'{kResultsDir}/dataset-{real_robot.name}.npz',
             t=traj_t,
             s=traj_s,
             u=traj_u,
             v=aux_v_θdot[:,0],
             θdot=aux_v_θdot[:,1],
             t_decimated=dataset_trajectory_decimated.t,
             s_decimated=dataset_trajectory_decimated.s,
             u_decimated=dataset_trajectory_decimated.u,
             v_decimated=aux_v_θdot_decimated[:,0],
             θdot_decimated=aux_v_θdot_decimated[:,1],
             t_measured=dataset_trajectory_measured.t,
             s_measured=dataset_trajectory_measured.s,
             u_measured=dataset_trajectory_measured.u,
             v_measured=aux_v_θdot_measured[:,0],
             θdot_measured=aux_v_θdot_measured[:,1],
             φdot_samples=φdot_samples)
    print(f'- Summary of dataset generation for {real_robot.name} robot:')
    print(f'  * Length of decimated trajectory = {len(traj_u)}')
    print(f'  * φdot_l in [{np.min(traj_u[:,0])}, {np.max(traj_u[:,0])}]')
    print(f'  * φdot_r in [{np.min(traj_u[:,1])}, {np.max(traj_u[:,1])}]')
    return dataset_trajectory, aux_v_θdot
#:generate_dataset_trajectory()


if __name__ == "__main__":
    real_robots = [robot.DDMR(f'Robots/{name}.yaml') for name in kRobotNames]
    for real_robot in real_robots:
        generate_dataset_trajectory(real_robot)
    #:
#:__main__
