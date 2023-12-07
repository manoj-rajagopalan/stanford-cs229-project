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
    traj_aux_v_ω = []

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
                    real_robot.execute_wheel_control_policy(t, kSimΔt, φdots[np.newaxis,:], s0)
                t0 = incremental_traj.t[-1]
                s0 = incremental_traj.s[-1]

                # Aux info at state: v and θdot
                _, _, _, φ_l, φ_r = s0
                φdot_l, φdot_r = φdots
                v_l = real_robot.left_wheel.v(φ_l, φdot_l)
                v_r = real_robot.right_wheel.v(φ_r, φdot_r)
                v = 0.5 * (v_r + v_l)
                ω = 0.5 / real_robot.L * (v_r - v_l)

                traj_u.append(φdots.copy())
                traj_t.append(t0.copy())
                traj_s.append(s0.copy())
                traj_aux_v_ω.append([v, ω])

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
            del traj_aux_v_ω[-1]
            t0 = traj_t[-1]
            s0 = traj_s[-1]

        #:for ibranch
    #:for iloop

    traj_t = np.array(traj_t)
    traj_s = np.array(traj_s)
    traj_u = np.array(traj_u)
    traj_aux_v_ω = np.array(traj_aux_v_ω)

    assert traj_s.shape == (len(traj_t), 5)
    traj_s[:,2:] = np.mod(traj_s[:,2:], 2*np.pi) # make θ, φ_l, φ_r canonical
    assert traj_u.shape == (len(traj_t)-1, 2)
    assert traj_aux_v_ω.shape == (len(traj_u), 2)

    dataset_trajectory = Trajectory(traj_t, traj_s, traj_u,
                                    Trajectory.Type.WHEEL_DYNAMICS,
                                    name=f'dataset-{real_robot.name}')
    dataset_trajectory.plot(kResultsDir)

    # Decimate and add noise
    dataset_trajectory_measurement_ground_truth, _ = \
        trajectory.decimate(dataset_trajectory)
    dataset_trajectory_measurement_ground_truth.plot(kResultsDir)
    
    dataset_trajectory_measured = trajectory.add_noise(dataset_trajectory_measurement_ground_truth,
                                                       xy_std_dev=0.01,
                                                       θ_deg_std_dev=1,
                                                       φ_lr_deg_std_dev=1,
                                                       φdot_lr_deg_per_sec_std_dev=1)
    dataset_trajectory_measured.plot(kResultsDir)

    # Massage trajectory names to be more semantic.
    dataset_trajectory_measurement_ground_truth.name = \
    dataset_trajectory_measurement_ground_truth.name.replace('decimated',
                                                             'measurement_ground_truth')
    dataset_trajectory_measured.name = \
    dataset_trajectory_measured.name.replace('decimated-with-noise',
                                             'measured')

    aux_v_ω_measurement_ground_truth = traj_aux_v_ω[range(0, len(traj_aux_v_ω), 10)]
    aux_v_ω_measured = aux_v_ω_measurement_ground_truth \
                        + np.vstack((np.random.normal(loc=0, scale=0.01, size=len(aux_v_ω_measurement_ground_truth)),
                                     np.random.normal(loc=0, scale=np.deg2rad(1), size=len(aux_v_ω_measurement_ground_truth)))).T
    
    np.savez(f'{kResultsDir}/dataset-{real_robot.name}.npz',
             t=dataset_trajectory.t,
             s=dataset_trajectory.s,
             u_type=int(dataset_trajectory.u_type),
             φdots_lr=dataset_trajectory.u,
             v_ω=traj_aux_v_ω,

             t_measurement_ground_truth=dataset_trajectory_measurement_ground_truth.t,
             s_measurement_ground_truth=dataset_trajectory_measurement_ground_truth.s,
             u_type_measurement_ground_truth=int(dataset_trajectory_measurement_ground_truth.u_type),
             φdots_lr_measurement_ground_truth=dataset_trajectory_measurement_ground_truth.u,
             v_ω_measurement_ground_truth=aux_v_ω_measurement_ground_truth,

             t_measured=dataset_trajectory_measured.t,
             s_measured=dataset_trajectory_measured.s,
             u_type_measured=int(dataset_trajectory_measured.u_type),
             φdots_lr_measured=dataset_trajectory_measured.u,
             v_ω_measured=aux_v_ω_measured,

             φdot_samples=φdot_samples)

    print(f'- Summary of dataset generation for {real_robot.name} robot:')
    print(f'  * Length of decimated trajectory = {len(traj_u)}')
    print(f'  * φdot_l in [{np.min(traj_u[:,0])}, {np.max(traj_u[:,0])}]')
    print(f'  * φdot_r in [{np.min(traj_u[:,1])}, {np.max(traj_u[:,1])}]')
    return dataset_trajectory, traj_aux_v_ω
#:generate_dataset_trajectory()


if __name__ == "__main__":
    real_robots = [robot.DDMR(f'Robots/{name}.yaml') for name in kRobotNames]
    for real_robot in real_robots:
        generate_dataset_trajectory(real_robot)
    #:
#:__main__
