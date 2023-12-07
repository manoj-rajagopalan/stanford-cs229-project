import numpy as np
import os
import time
from matplotlib import pyplot as plt

import eval
import ideal
import trajectory
from robot import DDMR
from constants import *
from wheel_model import NoisyWheel
from kinematic_control import KinematicallyControlledDDMR

Trajectory = trajectory.Trajectory # using namespace

kResultsDir = 'Results/3-SysId_via_SGD'

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
            self.params[:self.N_φ] = ref_robot.left_wheel.R
            self.params[self.N_φ:-1] = ref_robot.right_wheel.R
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
        R_l, R_r = self.params[self.n_l], self.params[self.n_r + self.N_φ]
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
            self.grad_params[n_r + self.N_φ] += κ_sqr * v_diff[i] * φdot_r[i]/2 \
                                              + θdot_diff[i] * φdot_r[i]/2 * one_by_L
            self.grad_params[-1] += θdot_diff[i] * (R_r * φdot_r[i] - R_l * φdot_l[i])/2
        #:for i
        self.grad_params /= len(X)

        self.params -= α * self.grad_params
    #:step()

    def results(self) -> (np.ndarray, np.ndarray, float):
        return self.params[:self.N_φ], self.params[self.N_φ:-1], 1.0 / self.params[-1]
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
        loss_value = (1/2) * np.sum((1/len(Y)) * Y_diff * M_Y_diff.T)
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
    # param_init = ref_robot.left_wheel.R \
    #            + .5 * (np.random.rand(len(model.params)) - 0.5)
    # param_init[-1] = ref_robot.L + 0.2
    # model.initialize(param_init)
    loss_fn = SgdMahalanobisLoss(np.array([[κ_sqr, 0],
                                           [   0 , 1]])) # is MSE loss when κ_sqr==1
    batch_size = 20
    num_epochs = 1000
    epoch_losses = np.zeros(num_epochs)

    X = np.hstack((dataset_trajectory.s[1:], dataset_trajectory.u))
    Y = dataset_v_θdot
    
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

            batch_num = batch_start // batch_size
            print(f'    Batch {batch_num+1} loss = {loss:0.5f}')
        #:for batch_start

        # End-of-epoch loss
        Y_pred = model.predict(X)
        loss = loss_fn(Y_pred, Y)
        epoch_losses[i_epoch] = loss
        print(f'  Epoch {i_epoch+1} loss = {loss:0.5f}')
    #:for i_epoch

    R_ls, R_rs, L = model.results()
    return R_ls, R_rs, L, epoch_losses
#:learn_system_params_via_SGD()

def make_wheel(ref_wheel_radius: float,
               Rs: np.ndarray) -> NoisyWheel:
    perturbations = []
    φ = 0
    Δφ_deg = 360 / len(Rs)
    for i in range(len(Rs)):
        scale = Rs[i] / ref_wheel_radius
        perturbations.append({'angular_position_deg': φ,
                              'angular_extent_deg': Δφ_deg,
                              'scale': scale})
        φ += Δφ_deg
    #:for i
    wheel = NoisyWheel(ref_wheel_radius, perturbations)
    return wheel
#:make_wheel()

def make_robot(name: str,
               L: float,
               R_ls: np.ndarray,
               R_rs: np.ndarray,
               ref_wheel_radius: float) -> DDMR:

    robot = DDMR()
    robot.name = name
    robot.L = L
    robot.left_wheel = make_wheel(ref_wheel_radius, R_ls)
    robot.right_wheel = make_wheel(ref_wheel_radius, R_rs)
    return robot
#:make_robot()


# def compute_κ_sqr(R: float, L: float):
def compute_κ_sqr(dataset_aux: np.ndarray):

    return 10

    # v, θdot = dataset_aux[:,0], dataset_aux[:,1]
    # var_v, var_θdot = np.var(v), np.var(θdot)
    # κ_sqr = var_θdot / var_v
    # return κ_sqr


    '''
    Compute a constant to bring v and θdot on the same scale for norm-computations.

    R: radius of golden robot's wheel
    '''

    '''
    max_wheel_v = φdot_max_mag_rps * 2*np.pi * R
    max_v = 1/2 * (max_wheel_v + max_wheel_v)
    max_θdot = (1/(2*L)) * (max_wheel_v - 0)
    κ = max_θdot / max_v
    return κ*κ
    '''
#:compute_κ_sqr()

def train(real_robots: list[DDMR]) -> None:
    print('-------------------')
    print('Performing training')
    print('-------------------')

    # Number of angular bins to learn radii per
    N_φ = 30
    
    # Learning rate
    α = 0.0001 # values of 0.001 and above yield nan very quickly

    ideal_robot = DDMR(config_filename='Robots/Ideal.yaml')

    for robot in real_robots:
        dataset = np.load(f'Results/2-Dataset/dataset-{robot.name}.npz')
        dataset_trajectory = Trajectory(dataset['t_measured'],
                                        dataset['s_measured'],
                                        dataset['u_measured'],
                                        name=f'dataset-{robot.name}-measured')
        dataset_aux = np.vstack((dataset['v_measured'],
                                 dataset['θdot_measured'])).T

        κ_sqr = compute_κ_sqr(dataset_aux) # scaling parameter for Mahalanobis distance
        print(f'Robot {robot.name} κ_sqr = {κ_sqr}')
        κ_sqrs = [κ_sqr, 1] # Mahalanobis distance metric-tensor parameter

        results_dict = {}

        # Experiment with Mahalanobis and MSE loss functions
        for i_loss_type in range(2):
            loss_type = 'mahalanobis' if i_loss_type == 0 else 'mse'

            # Experiment with dataset-shuffling to study effects
            for is_shuffling_enabled in [False, True]:
                print('.....................................')
                print(f'Learning {robot.name} with {loss_type} loss and {"" if is_shuffling_enabled else "no"} shuffling')
                print('.....................................')

                start_time_ns = time.time_ns()
                R_ls, R_rs, L, epoch_losses = \
                    learn_system_params_via_SGD(dataset_trajectory, # X
                                                dataset_aux,  # Y
                                                ideal_robot,
                                                N_φ, κ_sqrs[i_loss_type], α,
                                                is_shuffling_enabled)
                end_time_ns = time.time_ns()
                print(f'Learnt {robot.name} with {loss_type} loss and{"" if is_shuffling_enabled else " no"} shuffling in {(end_time_ns-start_time_ns)/1.0e9} seconds.')
                shuffle_suffix = '-shuffled' if is_shuffling_enabled else ''
                learnt_robot = make_robot(f'robot-sysId-{robot.name}-{loss_type}{shuffle_suffix}',
                                          L, R_ls, R_rs, ideal_robot.left_wheel.R)
                learnt_robot.write_to_file(f'{kResultsDir}/{learnt_robot.name}.yaml')
                results_dict[f'R_ls-{loss_type}{shuffle_suffix}'] = R_ls
                results_dict[f'R_rs-{loss_type}{shuffle_suffix}'] = R_rs
                results_dict[f'L-{loss_type}{shuffle_suffix}'] = L
                results_dict[f'epoch_losses-{loss_type}{shuffle_suffix}'] = epoch_losses
            #:for is_shuffling_enabled

        #:for i_loss_type

        np.savez(f'{kResultsDir}/sysId-{robot.name}.npz', **results_dict)
    #:for f

#:train_robots()


def plot_convergence_profiles(real_robots: list[DDMR]) -> None:
    for robot in real_robots:
        results = np.load(f'{kResultsDir}/sysId-{robot.name}.npz')
        _, ax = plt.subplots()
        epoch_losses_mah = results['epoch_losses-mahalanobis']
        epoch_losses_mah_shuf = results['epoch_losses-mahalanobis-shuffled']
        epoch_losses_mse = results['epoch_losses-mahalanobis']
        epoch_losses_mse_shuf = results['epoch_losses-mahalanobis-shuffled']
        assert len(epoch_losses_mah) == len(epoch_losses_mah_shuf)
        assert len(epoch_losses_mse) == len(epoch_losses_mse_shuf)
        assert len(epoch_losses_mse) == len(epoch_losses_mah)
        epoch_nums = np.arange(len(epoch_losses_mah)) + 1
        assert len(epoch_losses_mse) == len(epoch_losses_mse_shuf)
        ax.plot(epoch_nums, epoch_losses_mah, 'm-')
        ax.plot(epoch_nums, epoch_losses_mah_shuf, 'r-')
        ax.plot(epoch_nums, epoch_losses_mse, 'c-')
        ax.plot(epoch_nums, epoch_losses_mse_shuf, 'g-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.set_title(f'SGD convergence: robot {robot.name}')
        plt.savefig(f'{kResultsDir}/epoch_losses-{robot.name}.png')
        plt.close()
    #:for robot
#:plot_convergence_profiles()


def evaluate(new_trajectories: list[list[Trajectory]],
             ideal_trajectories) -> None:
    for new_traj_set_per_robot in new_trajectories:
        assert len(new_traj_set_per_robot) == len(ideal_trajectories)
        for (new_traj, ideal_traj) in zip(new_traj_set_per_robot, ideal_trajectories):
            Δs, Δs_per_unit_length, length = \
                eval.rate_trajectory(new_traj, ideal_traj)

            _, ax = plt.subplots()
            ax.plot(length, Δs)
            ax.set_xlabel('Trajectory length (m)')
            ax.set_ylabel('Separation from ref traj (m)')
            plt.savefig(f'{kResultsDir}/{new_traj.name}-eval-sep.png')
            plt.close()

            print(f'Evaluation summary for trajectory {new_traj.name}:')
            print(f'- Traj length  = {length[-1]} m')
            print(f'    max Δs     = {np.max(Δs)} at {length[np.argmax(Δs)]} m')
            print(f'    max Δs/len = {np.max(Δs_per_unit_length)} at {length[np.argmax(Δs_per_unit_length)]} m')
        #: for (traj s)
    #:for new_traj_set_per_robot
#:evaluate()


def evaluate_uncontrolled(real_robots: list[DDMR]) -> None:
    '''
    If system-identification was performed correctly, the learnt
    robots must produce the same trajectory as their target,
    provided the same controls.
    Here, the controls are those provided to the ideal robot.
    '''
    ideal_trajectories = ideal.load_trajectories()
    learnt_robots = [DDMR(f'{kResultsDir}/robot-sysId-{robot.name}-mahalanobis-shuffled.yaml')
                     for robot in real_robots]
    new_trajectories = \
        eval.run_robots_with_controls(learnt_robots,
                                      ideal_trajectories,
                                      kResultsDir)
    evaluate(new_trajectories, ideal_trajectories)
#:evaluate_uncontrolled()


def evaluate_controlled(real_robots: list[DDMR]) -> None:
    '''
    Apply the kinematic controller to all learnt robots
    and hope that we match the ideal robot.
    '''
    ideal_robot = DDMR('Robots/Ideal.yaml')
    controlled_robots = [KinematicallyControlledDDMR(
                            config_filename=f'{kResultsDir}/robot-sysId-{robot.name}-mahalanobis-shuffled.yaml',
                            ideal_robot=ideal_robot,
                            verbose=False)
                         for robot in real_robots]
    ideal_trajectories = ideal.load_trajectories()
    new_trajectories = \
        eval.run_robots_with_controls(controlled_robots,
                                      ideal_trajectories,
                                      kResultsDir)
    evaluate(new_trajectories, ideal_trajectories)
#:evaluate_controlled()

if __name__ == "__main__":
    os.makedirs(kResultsDir, exist_ok=True)
    real_robots = [DDMR(f'Robots/{name}.yaml') for name in kRobotNames]
    train(real_robots)
    plot_convergence_profiles(real_robots)
    evaluate_uncontrolled(real_robots)
    evaluate_controlled(real_robots)
#:__main__
