from typing import Any
import numpy as np
from tqdm import tqdm


import trajectory
from robot import DDMR
from constants import *

Trajectory = trajectory.Trajectory # using namespace

class SgdModel:
    def __init__(self, N_φ: int) -> None:
        self.N_φ = N_φ
        φ_bins = np.linspace(0, 2*np.pi, N_φ+1)
        self.φ_bins = np.vstack((φ_bins, φ_bins)).T
        self.params = np.zeros(2*N_φ + 1) # 2 wheels, and one for baseline
        self.grad_params = np.zeros_like(self.params)
    #:__init__()

    def initialize(self, init_object: any) -> None:
        if type(init_object) == DDMR:
            ref_robot: DDMR = init_object
            self.params[:N_φ] = ref_robot.left_wheel.R
            self.params[N_φ:-1] = ref_robot.right_wheel.R
            self.params[-1] = 1.0 / ref_robot.L
        elif type(init_object) == np.ndarray:
            params_value: np.ndarray = init_object
            assert params_value.shape == self.params.shape
            self.params[:] = params_value[:]
        else:
            assert False, 'Invalid initialization object for SgdModel'
        #:
    #:

    def predict(self, X:np.ndarray) -> np.ndarray:
        φ_lr = X[:, 3:5]
        φ_lr_3D = φ_lr[:,:,np.newaxis]
        φ_bins_3D = self.φ_bins.T[np.newaxis, :, :]
        n_lr = np.argmax(np.logical_and(φ_bins_3D[:,:,:-1] <= φ_lr_3D,
                                        φ_lr_3D < φ_bins_3D[:,:,1:]),
                         axis=2)
        assert n_lr.shape == φ_lr.shape
        self.n_l, self.n_r = np.hsplit(n_lr, 2) # save in self for use in step() below
        R_l, R_r = self.params[self.n_l], self.params[self.n_r + N_φ]
        one_by_L = self.params[-1]
        φdot_l, φdot_r = np.hsplit(X[:, 5:], 2)
        v_pred = 0.5 * (R_r * φdot_r + R_l * φdot_l)
        θdot_pred = 0.5 * (R_r * φdot_r - R_l * φdot_l) * one_by_L
        Y_pred = np.hstack((v_pred, θdot_pred))
        self.n_l, self.n_r = np.squeeze(self.n_l), np.squeeze(self.n_r) # convert to vector from 1D matrix
        return Y_pred
    #:predict()

    def step(self,
             X: np.ndarray,
             Y_diff: np.ndarray,
             α: float,
             κ_sqr: float) -> None:
        '''
        Y_diff = Y_pred - Y
        '''
        v_diff, θdot_diff = Y_diff[:,0], Y_diff[:,1]
        φdot_l, φdot_r = X[:,-2], X[:,-1]
        one_by_L = self.params[-1]

        self.grad_params.fill(0)
        for i in range(len(X)):
            n_l, n_r = self.n_l[i], self.n_r[i]
            R_l, R_r = self.params[n_l], self.params[n_r + self.N_φ]
            self.grad_params[n_l] += κ_sqr * v_diff[i] * φdot_l[i]/2 \
                                   - θdot_diff[i] * φdot_l[i]/2 * one_by_L
            self.grad_params[n_r + N_φ] += κ_sqr * v_diff[i] * φdot_r[i]/2 \
                                         + θdot_diff[i] * φdot_r[i]/2 * one_by_L
            self.grad_params[-1] += θdot_diff[i] * (R_r * φdot_r[i] - R_l * φdot_l[i])/2
        #:for i
        self.grad_params /= len(X)

        self.params -= α * self.grad_params
    #:step()

    def results(self) -> (np.ndarray, np.ndarray, float):
        return self.params[:N_φ], self.params[N_φ:-1], 1.0 / self.params[-1]
    #:

#:Model

class SgdMahalanobisLoss:
    def __init__(self, metric_tensor: np.ndarray) -> None:
        assert metric_tensor.shape == (2,2)
        self.M = metric_tensor
    #:

    def __call__(self, Y_pred, Y) -> float:
        Y_diff = Y_pred - Y
        M_Y_diff = self.M @ Y_diff.T
        # Treat matrix as list of row-vectors and perform element-wise inner-prods
        loss_value = (1/2) * np.sum((1/len(Y)) * Y_diff * M_Y_diff.T)
        # loss_value = np.einsum('ij,ij->i', (1/len(Y)) * Y_diff, M_Y_diff.T)
        return loss_value
    #:__call__()

#:Mahalanobis

class SgdMseLoss(SgdMahalanobisLoss):
    def __init__(self) -> None:
        super().__init__(np.eye(2))
    #:
#:MseLoss

def learn_system_params_via_SGD(dataset_trajectory: Trajectory,
                                dataset_v_θdot: np.ndarray, # labels
                                ref_robot: DDMR, # the one we think this one is, for initial estimates
                                N_φ: int, # number of angular bins per wheel
                                κ_sqr: float,
                                α: float,
                                is_shuffling_enabled: bool):
    '''
    κ_sqr: parameter for Mahalanobis distance matrix [[κ_sqr, 0], [0, 1]]
    α: learning rate
    '''
    N_data = len(dataset_trajectory.u)

    model = SgdModel(N_φ)
    model.initialize(ref_robot)
    loss_fn = SgdMahalanobisLoss(np.array([[κ_sqr, 0],
                                           [   0 , 1]])) # is MSE loss when κ_sqr==1
    batch_size = 20
    num_epochs = 1000
    epoch_losses = np.zeros(num_epochs)

    # grad_params = np.zeros_like(params) # pre-allocate gradient vector

    X = np.hstack((dataset_trajectory.s[1:], dataset_trajectory.u))
    Y = dataset_v_θdot
    # v, θdot = dataset_v_θdot[:,0], dataset_v_θdot[:,1] # separate label categories
    
    if is_shuffling_enabled:
        shuffle = \
            np.random.default_rng().permutation(np.arange(0, len(dataset_trajectory.u)))
    #:if

    for i_epoch in range(num_epochs):
        print(f'Epoch {i_epoch+1}/{num_epochs}')

        for batch_start in range(0, N_data, batch_size):
            batch_end = min(batch_start + batch_size, N_data)

            i = range(batch_start, batch_end)
            if is_shuffling_enabled:
                i = shuffle[i]
            #:

            Y_pred_i = model.predict(X[i])
            loss = loss_fn(Y_pred_i, Y[i])
            model.step(X[i], Y_pred_i - Y[i], α, κ_sqr)

            '''
            for j in range(batch_start, batch_end):
                i = shuffle[j] if is_shuffling_enabled else j
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
            '''
            # grad_params /= (batch_end - batch_start)
            # params -= α * grad_params

            # loss /= (batch_end - batch_start)
            batch_num = batch_start // batch_size
            print(f'    Batch {batch_num+1} loss = {loss:0.5f}')
        #:for batch_start

        # End of epoch loss
        Y_pred = model.predict(X)
        loss = loss_fn(Y_pred, Y)
        epoch_losses[i_epoch] = loss
        print(f'  Epoch {i_epoch} loss = {loss:0.5f}')
    #:for i_epoch

    return *model.results(), epoch_losses
#:learn_system_params_via_SGD()

def compute_κ_sqr(R: float, L: float):
    '''
    Compute a constant to bring v and θdot on the same scale for norm-computations.

    R: radius of golden robot's wheel
    '''
    max_wheel_v = φdot_max_mag_rps * 2*np.pi * R
    max_v = 1/2 * (max_wheel_v + max_wheel_v)
    max_θdot = (1/(2*L)) * (max_wheel_v - 0)
    κ = max_θdot / max_v
    return κ*κ
#:compute_κ_sqr()


if __name__ == "__main__":
    N_φ = 30
    
    # Learning rate
    # 0.001 yields nan very quickly
    α = 0.0001

    golden_robot = DDMR(config_filename='Robots/golden.yaml')

    robot_yaml_filenames = ['smaller_left_wheel', 'larger_left_wheel',
                            'smaller_baseline', 'larger_baseline',
                            'noisy', 'noisier']
    for f in robot_yaml_filenames:
        robot = DDMR(config_filename=f'Robots/{f}.yaml')
        dataset = np.load(f'Results/dataset-{robot.name}.npz')
        results_dict = {}

        # κ = 2*np.pi / (φdot_max_mag_rps * 2*np.pi * golden_robot.left_wheel.R)
        κ_sqr = compute_κ_sqr(golden_robot.left_wheel.R, golden_robot.L)
        print(f'Robot {robot.name} κ_sqr = {κ_sqr}')
        κ_sqrs = [κ_sqr, 1] # Mahalanobis distance metric-tensor parameter

        # Experiment with Mahalanobis and MSE loss functions
        for i_loss_type in range(2):
            loss_type = 'mahalanobis' if i_loss_type == 0 else 'mse'
            print('----------------')
            print(f'Learning {robot.name} with {loss_type} loss')
            dataset_trajectory = Trajectory(dataset['t_measured'],
                                            dataset['s_measured'],
                                            dataset['u_measured'],
                                            name=f'dataset-{f}-measured')
            dataset_aux = np.vstack((dataset['v_measured'],
                                     dataset['θdot_measured'])).T

            # Experiment with dataset-shuffling to study effects
            for is_shuffling_enabled in [False, True]:
                R_ls, R_rs, L, epoch_losses = \
                    learn_system_params_via_SGD(dataset_trajectory, # X
                                                dataset_aux,  # Y
                                                golden_robot,
                                                N_φ, κ_sqrs[i_loss_type], α,
                                                is_shuffling_enabled)
                shuffle_suffix = '-shuffled' if is_shuffling_enabled else ''
                results_dict[f'R_ls-{loss_type}{shuffle_suffix}'] = R_ls
                results_dict[f'R_rs-{loss_type}{shuffle_suffix}'] = R_rs
                results_dict[f'L-{loss_type}{shuffle_suffix}'] = L
                results_dict[f'epoch_losses-{loss_type}{shuffle_suffix}'] = epoch_losses
                # break
            #:for is_shuffling_enabled
            # break
        #:for i_loss_type
        np.savez(f'Results/sgd-{robot.name}.npz', **results_dict)

        # break
    #:for f
#:__main__