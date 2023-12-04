import numpy as np

import robot
from constants import *
from trajectory import Trajectory
from learn import learn_system_params_via_SGD


def generate_dataset_trajectory(test_robot: robot.DDMR,
                                num_loops: int) -> (Trajectory, np.ndarray):
    '''
    Get the robot to perform `num_loop` figures-of-8 with various radii.
    '''
    print(f'Generating dataset for {test_robot.name} robot')
    rng = np.random.default_rng()
    
    # Pre-generate samples of φdot_l and φdot_R to create figure-of-8 loops with
    δφdot = (0.5 * φdot_max_mag_rps) / num_loops
    φdot0 = 0
    φdot_samples = np.zeros((num_loops,2))
    for iloop in range(num_loops):
        φdot_samples[iloop,0] = φdot0 + rng.random() * δφdot
        φdot_samples[iloop,1] = φdot_max_mag_rps - φdot0 - δφdot * rng.random()
        assert φdot_samples[0,0] < φdot_samples[0,1]
        φdot0 += δφdot
    #:

    s0 = np.array([0,0,0,0,0])
    t0 = 0.0
    measurement_t0 = 0.0
    traj_t = [t0]
    traj_s = [s0]
    traj_u = []
    aux_v_θdot = []

    for iloop in range(num_loops):
        φdots = φdot_samples[iloop]
        for ibranch in range(2):
            # 1st branch goes CCW
            # 2nd branch goes CW due to the φdot swap at the end
            counter = 0
            while 0 <= s0[2] <= (2*np.pi):
                counter += 1
                tf = t0 + kSimΔt
                t = np.array([t0, tf])
                incremental_traj = test_robot.execute_control_policy(t, φdots[np.newaxis,:], s0)
                prev_t0 = t0
                prev_s0 = s0
                t0 = incremental_traj.t[-1]
                s0 = incremental_traj.s[-1]

                # Aux info at state: v and θdot
                _, _, _, φ_l, φ_r = s0
                φdot_l, φdot_r = φdots
                v_l = test_robot.left_wheel.v(φ_l, φdot_l)
                v_r = test_robot.right_wheel.v(φ_r, φdot_r)
                v = 0.5 * (v_r + v_l)
                θdot = 0.5 / test_robot.L * (v_r - v_l)

                # Measurement at lower sampling rate
                if (t0 - measurement_t0) >= kMeasurementΔt:
                    traj_u.append(φdots.copy())
                    traj_t.append(t0.copy())
                    traj_s.append(s0.copy())
                    aux_v_θdot.append([v, θdot])
                    measurement_t0 = t0
                #:if

            #:while s0[2] in [0, 2π)

            print(f'- loop #{iloop}: φdots = {φdots}, {counter} iterations')

            # swap
            temp = φdots[0]
            φdots[0] = φdots[1]
            φdots[1] = temp

            # Discard last trajectory point which violates limits
            t0 = prev_t0
            s0 = prev_s0

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

    dataset_trajectory = Trajectory(traj_t, traj_s, traj_u, name=f'dataset-{test_robot.name}')
    dataset_trajectory.plot()
    np.savez(f'Results/dataset-{test_robot.name}.npz',
             t=traj_t, s=traj_s, u=traj_u,
             v=aux_v_θdot[:,0], θdot=aux_v_θdot[:,1],
             φdot_samples=φdot_samples)
    print(f'- Summary for {test_robot.name} robot:')
    print(f'  * Length of decimated trajectory = {len(traj_u)}')
    print(f'  * φdot_l in [{np.min(traj_u[:,0])}, {np.max(traj_u[:,0])}]')
    print(f'  * φdot_r in [{np.min(traj_u[:,1])}, {np.max(traj_u[:,1])}]')
    return dataset_trajectory, aux_v_θdot
#:generate_dataset_trajectory()


def add_noise(traj: Trajectory) -> Trajectory:
    t = traj.t
    x, y, θ, φ_l, φ_r = traj.s.copy().T
    φdot_l, φdot_r = traj.u.copy().T
    x += np.random.normal(loc=0, scale=0.001, size=x.shape)
    y += np.random.normal(loc=0, scale=0.001, size=y.shape)
    θ = np.mod(θ + np.random.normal(loc=0, scale=np.deg2rad(1), size=θ.shape), 2*np.pi)
    φ_l = np.mod(φ_l + np.random.normal(loc=0, scale=np.deg2rad(1), size=φ_l.shape), 2*np.pi)
    φ_r = np.mod(φ_r + np.random.normal(loc=0, scale=np.deg2rad(1), size=φ_r.shape), 2*np.pi)

    φdot_l += np.random.normal(loc=0, scale=np.deg2rad(1), size=φdot_l.shape)
    φdot_r += np.random.normal(loc=0, scale=np.deg2rad(1), size=φdot_r.shape)

    s = np.transpose(np.vstack((x, y, θ, φ_l, φ_r)))
    u = np.vstack((φdot_l, φdot_r)).T
    print(f'add_noise:')
    print(f'  |Δs|={np.linalg.norm(s-traj.s)}')
    print(f'  |Δu|={np.linalg.norm(u-traj.u)}')
    return Trajectory(t,s,u, traj.name+'-with-noise')
#:add_noise()

if __name__ == "__main__":
    smaller_left_wheel_robot = robot.DDMR(name='SmallerLeftWheel', config_filename='Robots/smaller_left_wheel.yaml')
    larger_left_wheel_robot = robot.DDMR(name='LargerLeftWheel', config_filename='Robots/larger_left_wheel.yaml')
    noisy_robot = robot.DDMR(name='Noisy', config_filename='Robots/noisy.yaml')
    noisier_robot = robot.DDMR(name='Noisier', config_filename='Robots/noisier.yaml')
    test_robots = [smaller_left_wheel_robot, larger_left_wheel_robot, noisy_robot, noisier_robot]

    for test_robot in test_robots:
        dataset_trajectory = generate_dataset_trajectory(test_robot, num_loops=5)
    #:
    # dataset_trajectory, dataset_aux = generate_dataset2_trajectory(noisier_robot, 5, φdot_max_mag_rps)
    # dataset_trajectory.plot()
'''
    N_φ = 30
    κ = 2*np.pi / φdot_max_mag_rps
    κ_sqr = κ * κ # Mahalanobis distance metric-tensor parameter
    α = 0.001 # learning rate
    saved = np.load('dataset2-trajectory.npz')
    dataset_trajectory = Trajectory(saved['t'], saved['s'], saved['u'], name='dataset2')
    dataset_trajectory = add_noise(dataset_trajectory)
    dataset_trajectory.plot()
    dataset_aux = np.vstack((saved['v'], saved['θdot'])).T
    # R_ls, R_rs, L = learn_system_params_via_SGD(dataset_trajectory, dataset_aux, golden_robot,
    #                                             N_φ, κ_sqr, α)
    # np.savez(f'{noisier_robot.name}-SGD-results.npz',
    #          R_ls=R_ls, R_rs=R_rs, L=L)
'''
#: __main__
