import numpy as np
from tqdm import tqdm


import trajectory
from robot import DDMR
from constants import *

Trajectory = trajectory.Trajectory # using namespace


def learn_system_params_via_SGD(dataset_trajectory: Trajectory,
                                dataset_v_θdot: np.ndarray, # labels
                                ref_robot: DDMR, # the one we think this one is, for initial estimates
                                N_φ: int, # number of angular bins per wheel
                                κ_sqr: float,
                                α: float):
    '''
    κ_sqr: parameter for Mahalanobis distance matrix [[κ_sqr, 0], [0, 1]]
    α: learning rate
    '''
    N_data = len(dataset_trajectory.u)
    φ_bins = np.linspace(0, 2*np.pi, N_φ+1)
    params = np.zeros(2*N_φ + 1) # 2 wheels, and one for baseline
    M = np.array([[κ_sqr, 0], [0,1]]) # Mahalanobis distance metric tensor

    # Initialize with factory estimates
    # Layout: N_φ values for R_l, then N_φ values for R_r, then 1 for 1/L
    params[:N_φ] = ref_robot.left_wheel.R
    params[N_φ:-1] = ref_robot.right_wheel.R
    params[-1] = 1.0 / ref_robot.L

    batch_size = 20
    num_epochs = 1000

    grad_params = np.zeros_like(params)
    v, θdot = dataset_v_θdot[:,0], dataset_v_θdot[:,1] # separate label categories
    
    shuffle = \
        np.random.default_rng().permutation(np.arange(0, len(dataset_trajectory.u)))

    for i_epoch in range(num_epochs):
        print(f'Epoch {i_epoch+1}/{num_epochs}')
        for batch_start in range(0, N_data, batch_size):
            batch_end = min(batch_start + batch_size, N_data)
            grad_params.fill(0.0)
            loss = 0
            for j in range(batch_start, batch_end):
                i = shuffle[j]
                _, _, _, φ_l, φ_r = dataset_trajectory.s[i+1]
                # argwhere returns 2D array of shape (1,1)
                n_l = np.argwhere(np.logical_and(φ_bins[:-1] <= φ_l, φ_l < φ_bins[1:]))
                n_r = np.argwhere(np.logical_and(φ_bins[:-1] <= φ_r, φ_r < φ_bins[1:]))
                assert n_l.shape == n_r.shape == (1,1)
                assert 0 <= n_l < N_φ
                assert 0 <= n_r < N_φ
                n_l, n_r = n_l[0,0], n_r[0,0]
                R_l, R_r = params[n_l], params[N_φ + n_r]
                one_by_L = params[-1]
                φdot_l, φdot_r = dataset_trajectory.u[i]
                v_pred_i = 0.5 * (R_r * φdot_r + R_l * φdot_l)
                θdot_pred_i = 0.5 * (R_r * φdot_r - R_l * φdot_l) * one_by_L

                # Loss
                y_pred = np.array([v_pred_i, θdot_pred_i])
                y = np.array([v[i], θdot[i]])
                y_diff = y_pred - y
                v_diff, θdot_diff = y_diff
                loss += np.inner(y_diff, M @ y_diff)

                # Gradient
                grad_params[n_l] += κ_sqr * v_diff * φdot_l/2 \
                                  - θdot_diff * φdot_l/2 * one_by_L
                grad_params[N_φ + n_r] += κ_sqr * v_diff * φdot_r/2 \
                                        + θdot_diff * φdot_r/2 * one_by_L
                grad_params[-1] += θdot_diff * (R_r * φdot_r - R_l * φdot_l)/2
            #:for i

            grad_params /= (batch_end - batch_start)
            params -= α * grad_params

            loss /= (batch_end - batch_start)
            batch_num = batch_start // batch_size
            print(f'    Batch {batch_num+1} loss = {loss:0.5f}')
        #:for batch_start
    #:for i_epoch

    return params[:N_φ], params[N_φ:-1], (1.0 / params[-1])
#:learn_system_params_via_SGD()

if __name__ == "__main__":
    N_φ = 30
    κ = 2*np.pi / φdot_max_mag_rps
    κ_sqrs = [κ*κ, 1] # Mahalanobis distance metric-tensor parameter
    
    # Learning rate
    # 0.001 yields nan very quickly
    α = 0.0001

    golden_robot = DDMR(config_filename='Robots/golden.yaml')
    robot_yaml_filenames = ['smaller_left_wheel', 'larger_left_wheel',
                            'smaller_baseline', 'larger_baseline',
                            'noisy', 'noisier']
    for f in robot_yaml_filenames:
        robot = DDMR(config_filename=f'Robots/{f}.yaml')
        saved = np.load(f'Results/dataset-{robot.name}.npz')
        for i_loss_type in range(2):
            loss_type = 'mahalanobis' if i_loss_type == 0 else 'mse'
            print('----------------')
            print(f'Learning {robot.name} with {loss_type} loss')
            dataset_trajectory = Trajectory(saved['t_measured'],
                                            saved['s_measured'],
                                            saved['u_measured'],
                                            name=f'dataset-{f}-measured')
            dataset_aux = np.vstack((saved['v_measured'], saved['θdot_measured'])).T
            R_ls, R_rs, L = learn_system_params_via_SGD(dataset_trajectory, dataset_aux,
                                                        golden_robot,
                                                        N_φ, κ_sqrs[i_loss_type], α)
            np.savez(f'Results/sgd-{robot.name}-{loss_type}.npz',
                     R_ls=R_ls, R_rs=R_rs, L=L)
            
        #:for i_loss_type
        # exit()
    #:for f
#:__main__