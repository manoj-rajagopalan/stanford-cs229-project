import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as nn_optim

import ideal
import eval
import robot
import trajectory
from constants import *
from kinematic_control import KinematicController, KinematicallyControlledDDMR

# using namespace
Trajectory = trajectory.Trajectory
DDMR = robot.DDMR

kResultsDir = 'Results/4-Model_Free_FCNN'

class KinematicController_ModelFree(KinematicController):
    def __init__(self, name: str,
                 ideal_robot: DDMR,
                 nn_model: torch.nn.Module) \
        -> None:

        super().__init__(name, ideal_robot)
        self.nn_model = nn_model
    #:

    # Override
    def translate(self,
                  target_robot: DDMR,
                  v_ω_desired: np.ndarray,
                  s: np.ndarray) \
        -> np.ndarray: # φdots_lr

        x = np.hstack((s, v_ω_desired))
        self.nn_model.eval()
        y = self.nn_model(torch.Tensor(x))
        return y.detach().numpy()
    #:translate()
#:KinematicController_ModelFree

class KinematicController_ModelFree_Small(KinematicController):
    def __init__(self, name: str,
                 ideal_robot: DDMR,
                 nn_model: torch.nn.Module) \
        -> None:

        super().__init__(name, ideal_robot)
        self.nn_model = nn_model
    #:

    # Override
    def translate(self,
                  target_robot: DDMR,
                  v_ω_desired: np.ndarray,
                  s: np.ndarray) \
        -> np.ndarray: # φdots_lr

        x = np.hstack((s[3:], v_ω_desired)) # strip out x, y, θ
        self.nn_model.eval()
        y = self.nn_model(torch.Tensor(x))
        return y.detach().numpy()
    #:translate()
#:KinematicController_ModelFree



def train_nn(X: torch.Tensor,
             Y: torch.Tensor) \
    -> torch.nn.Module:

    nn_model = nn.Sequential(nn.Linear(7, 20),
                             nn.ReLU(),
                             nn.Linear(20,40),
                             nn.Sigmoid(),
                             nn.Linear(40,20),
                             nn.ReLU(),
                             nn.Linear(20,2))
    for tensor in nn_model.parameters():
        if len(tensor.shape) == 1:
            nn.init.normal_(tensor)
        else:
            nn.init.xavier_uniform_(tensor)
        #:
    #:
    loss_fn = nn.MSELoss()
    optimizer = nn_optim.Adam(nn_model.parameters(),
                              lr=0.001,
                              weight_decay=1.0e-5)

    num_epochs = 2000
    train_X = torch.Tensor(X)
    train_Y = torch.Tensor(Y)
    batch_size = 50
    num_batches = (train_X.shape[0] + batch_size - 1) // batch_size
    print('Running NN training')
    for epoch in range(num_epochs):
        nn_model.train()
        batch_start = 0

        for batch in range(num_batches):
            batch_end = min(batch_start + batch_size, train_X.shape[0])
            X = train_X[batch_start:batch_end]
            Y = train_Y[batch_start:batch_end]
            Y_pred = nn_model(X)
            loss = loss_fn(Y_pred, Y)
            print(f'@{epoch+1}:{batch+1}/{num_batches}, loss = {loss:0.4}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_start += batch_size
        #:for batch
    #:for epoch

    nn_model.eval()
    # test_Y_pred = model(test_X)
    # test_loss = loss_fn(test_Y_pred, test_Y)
    # print(f'Test loss = {test_loss}')
    # print(f'Max abs diff = {np.max(np.abs(test_Y_pred.detach().numpy() - test_Y.detach().numpy()))}')
    return nn_model
#:train_nn()

def train_nn_small(X: torch.Tensor,
                   Y: torch.Tensor) \
    -> torch.nn.Module:

    nn_model = nn.Sequential(nn.Linear(4, 100),
                             nn.ReLU(),
                             nn.Linear(100,40),
                             nn.ReLU(),
                             nn.Linear(40,20),
                             nn.ReLU(),
                             nn.Linear(20,2))
    for tensor in nn_model.parameters():
        if len(tensor.shape) == 1:
            nn.init.normal_(tensor)
        else:
            nn.init.xavier_uniform_(tensor)
        #:
    #:
    loss_fn = nn.MSELoss()
    optimizer = nn_optim.Adam(nn_model.parameters(),
                              lr=0.01,
                              weight_decay=1.0e-3)

    num_epochs = 10000
    train_X = torch.Tensor(X)
    train_Y = torch.Tensor(Y)
    batch_size = 50
    num_batches = (train_X.shape[0] + batch_size - 1) // batch_size
    print('Running NN training')
    for epoch in range(num_epochs):
        nn_model.train()
        batch_start = 0

        for batch in range(num_batches):
            batch_end = min(batch_start + batch_size, train_X.shape[0])
            X = train_X[batch_start:batch_end]
            Y = train_Y[batch_start:batch_end]
            Y_pred = nn_model(X)
            loss = loss_fn(Y_pred, Y)
            print(f'@{epoch+1}:{batch+1}/{num_batches}, loss = {loss:0.4}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_start += batch_size
        #:for batch
    #:for epoch

    nn_model.eval()
    return nn_model
#:train_nn()


def train(real_robots: list[DDMR],
          is_shuffling_enabled: bool) \
    -> list[torch.nn.Module]:

    nn_models = []
    for robot in real_robots:
        # Construct dataset
        dataset = np.load(f'Results/2-Dataset/dataset-{robot.name}.npz')
        X = np.hstack((dataset['s_measured'][1:],
                       dataset['v_ω_measured']))
        Y = dataset['φdots_lr_measured']
        assert X.shape[0] == Y.shape[0]

        indices = np.arange(0, len(X), dtype=int)
        if is_shuffling_enabled:
            indices = np.random.default_rng().permutation(indices) # shuffle
            X = X[indices]
            Y = Y[indices]
        #:

        nn_model = train_nn(X, Y)
        torch.save(nn_model, f'{kResultsDir}/nn_model-{robot.name}.pth')
        # -- alternately --
        # nn_model = train_nn_small(X[:,3:], Y)
        # torch.save(nn_model, f'{kResultsDir}/nn_model-small-{robot.name}.pth')

        nn_models.append(nn_model)
    #:for robot
    return nn_models
#:train()

def evaluate(real_robots: list[DDMR]) -> None:

    ideal_robot = DDMR('Robots/Ideal.yaml')
    ideal_trajectories = ideal.load_trajectories()

    for real_robot in real_robots:
        nn_model = torch.load(f'{kResultsDir}/nn_model-{real_robot.name}.pth')

        # -- alternately --
        # nn_model = torch.load(f'{kResultsDir}/nn_model-small-{real_robot.name}.pth')

        kinematic_controller = KinematicController_ModelFree(name='nnCtl',
                                                             ideal_robot=ideal_robot,
                                                             nn_model=nn_model)
        # -- alternately --
        # kinematic_controller = KinematicController_ModelFree_Small(name='nnSmallCtl',
        #                                                            ideal_robot=ideal_robot,
        #                                                            nn_model=nn_model)

        kin_ctl_robot = KinematicallyControlledDDMR(
                            config_filename=f'Robots/{real_robot.name}.yaml',
                            controller=kinematic_controller,
                            verbose=False)
        eval.evaluate_controlled_trajectories([real_robot],
                                              [kin_ctl_robot],
                                              kResultsDir)
        loss = 0
        loss_fn = torch.nn.MSELoss(reduction='sum')
        for ideal_traj in ideal_trajectories:
            ideal_traj_u_wheel_to_body = \
                robot.translate_control_wheel_to_body(ideal_traj,
                                                      ideal_robot)
            X = np.hstack((ideal_traj.s[:-1], ideal_traj_u_wheel_to_body))

            Y_pred = nn_model(torch.Tensor(X)).detach()
            # -- alternately --
            # Y_pred = nn_model(torch.Tensor(X[:,3:])).detach()

            traj_loss = loss_fn(Y_pred, torch.Tensor(ideal_traj.u))
            loss += traj_loss
            traj_loss /= len(ideal_traj.u)
            print(f'For robot {real_robot.name}, test loss for trajectory {ideal_traj.name} = {traj_loss}',
                  flush=True)
        #:for ideal_traj
    #:for real_robot
#:evaluate()

if __name__ == "__main__":
    os.makedirs(kResultsDir, exist_ok=True)

    # real_robot_names = ['Noisier']
    # real_robots = [DDMR(f'Robots/{name}.yaml') for name in real_robot_names]
    # -- alternately --
    real_robots = [DDMR(f'Robots/{name}.yaml') for name in kRobotNames]

    train(real_robots, is_shuffling_enabled=True)
    evaluate(real_robots)
#:__main__