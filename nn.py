import numpy as np
import torch
import torch.nn as nn
import torch.optim as nn_optim

from trajectory import Trajectory

def analyze_trajectory_distribution(φdot_max_mag_rps):
    traj_file = np.load('dataset2-trajectory.npz')
    traj = Trajectory(traj_file['t'], traj_file['s'], traj_file['u'], 'dataset')
    num_bins = 10
    φ_bins = np.linspace(0, 2*np.pi, num_bins+1)
    φ_l, φ_r = traj.s[1:,3], traj.s[1:,4]
    print(f'min φ = {min(np.min(φ_l), np.min(φ_r))}')
    print(f'max φ = {min(np.max(φ_l), np.max(φ_r))}')

    φdot_bins = np.linspace(0, φdot_max_mag_rps * 2*np.pi, num_bins+1)
    φdot_l = traj.u[:,0]
    φdot_r = traj.u[:,1]
    print(f'min φdot = {min(np.min(φdot_l), np.min(φdot_r))}')
    print(f'max φdot = {min(np.max(φdot_l), np.max(φdot_r))}')

    for i in range(num_bins):
        print(f'#φ_l in bin {i+1} = {np.count_nonzero(np.logical_and(φ_bins[i] <= φ_l, φ_l < φ_bins[i+1]))}')
    #:for i
    for j in range(num_bins):
        print(f'#φ_r in bin {j+1} = {np.count_nonzero(np.logical_and(φ_bins[j] <= φ_r, φ_r < φ_bins[j+1]))}')
    #:for i
    for k in range(num_bins):
        print(f'#φdot_l in bin {k+1} = {np.count_nonzero(np.logical_and(φdot_bins[k] <= φdot_l, φdot_l < φdot_bins[k+1]))}')
    #:for i
    for l in range(num_bins):
        print(f'#φdot_r in bin {l+1} = {np.count_nonzero(np.logical_and(φdot_bins[l] <= φdot_r, φdot_r < φdot_bins[l+1]))}')
    #:for i


    counts = np.zeros((num_bins, num_bins, num_bins, num_bins), dtype=int)
    bins = []
    for i in range(num_bins):
        φ_l_subset = np.logical_and(φ_bins[i] <= φ_l, φ_l < φ_bins[i+1])
        bins.append([])
        for j in range(num_bins):
            φ_l_φ_r_subset = np.logical_and(φ_bins[j] <= φ_r, φ_r < φ_bins[j+1], φ_l_subset)
            bins[-1].append([])
            for k in range(num_bins):
                φ_l_φ_r_φdot_l_subset = np.logical_and(φdot_bins[k] <= φdot_l,
                                                       φdot_l < φdot_bins[k+1],
                                                       φ_l_φ_r_subset)
                bins[-1][-1].append([])
                for l in range(num_bins):
                    φ_l_φ_r_φdot_l_φdot_r_subset = np.logical_and(φdot_bins[l] <= φdot_r,
                                                                  φdot_r < φdot_bins[l+1],
                                                                  φ_l_φ_r_φdot_l_subset)
                    count = np.count_nonzero(φ_l_φ_r_φdot_l_φdot_r_subset)
                    counts[i,j,k,l] =  count
                    bin = np.argwhere(φ_l_φ_r_φdot_l_φdot_r_subset).flatten()
                    assert type(bin) == np.ndarray
                    assert len(bin.shape) == 1
                    bins[-1][-1][-1].append(bin)
                #:for l
            #:for k
        #:for j
    #:for i
    print('Dataset distribution:')
    print(f'    min count = {np.min(counts)}')
    print(f'    max count = {np.max(counts)}')
    return traj, bins, counts
#:analyze_trajectory_distribution()


def split_train_test_datasets(dataset_trajectory: Trajectory,
                              u_bins: list,
                              test_set_fraction: float):
    train_indices = []
    test_indices = []
    random_gen = np.random.default_rng()
    for φ_l_indices in u_bins:
        for φ_l_φ_r_indices in φ_l_indices:
            for φ_l_φ_r_φdot_l_indices in φ_l_φ_r_indices:
                for φ_l_φ_r_φdot_l_φdot_r_indices in φ_l_φ_r_φdot_l_indices:
                    assert type(φ_l_φ_r_φdot_l_φdot_r_indices) == np.ndarray
                    assert len(φ_l_φ_r_φdot_l_φdot_r_indices.shape) == 1
                    dataset_indices = random_gen.permutation(φ_l_φ_r_φdot_l_φdot_r_indices)
                    num_samples = len(dataset_indices)
                    num_test_samples = \
                        int(np.round(test_set_fraction * num_samples))
                    test_indices += list(dataset_indices[:num_test_samples])
                    train_indices += list(dataset_indices[num_test_samples:])
                #:for φ_l_φ_r_φdot_l_φdot_r_indices
            #:for φ_l_φ_r_φdot_l_indices
        #:for φ_l_φ_r_indices
    #:for φ_l_indices

    print(f'train set size = {len(train_indices)}')
    print(f'test set size = {len(test_indices)}')

    train_X = dataset_trajectory.s[train_indices, :]
    train_Y = dataset_trajectory.u[train_indices,:]
    test_X = dataset_trajectory.s[test_indices, :]
    test_Y = dataset_trajectory.u[test_indices, :]
    return Dataset(train_X, train_Y), Dataset(test_X, test_Y)
#:split_train_test_datasets()

def run_nn(train_dataset, test_dataset):
    model = nn.Sequential(
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20,20),
        nn.ReLU(),
        nn.Linear(20,20),
        nn.ReLU(),
        nn.Linear(20,2)
    )
    for tensor in model.parameters():
        if len(tensor.shape) == 1:
            nn.init.normal_(tensor)
        else:
            nn.init.xavier_uniform_(tensor)
        #:
    #:
    loss_fn = nn.MSELoss()
    optimizer = nn_optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 10
    train_X = torch.Tensor(train_dataset.X)
    train_Y = torch.Tensor(train_dataset.Y)
    test_X = torch.Tensor(test_dataset.X)
    test_Y = torch.Tensor(test_dataset.Y)
    batch_size = 10000
    num_batches = (train_X.shape[0] + batch_size - 1) // batch_size
    print('Running NN training')
    for epoch in range(num_epochs):
        model.train()
        batch_start = 0

        for batch in range(num_batches):
            batch_end = min(batch_start + batch_size, train_X.shape[0])
            X = train_X[batch_start:batch_end]
            Y = train_Y[batch_start:batch_end]
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            print(f'@{epoch}:{batch}/{num_batches}, loss = {loss:0.4}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_start += batch_size
        #:for batch
    #:for epoch

    model.eval()
    test_Y_pred = model(test_X)
    test_loss = loss_fn(test_Y_pred, test_Y)
    print(f'Test loss = {test_loss}')
    print(f'Max abs diff = {np.max(np.abs(test_Y_pred.detach().numpy() - test_Y.detach().numpy()))}')

#:run_nn()

def learn_controller_via_NN(φdot_max_mag_rps: float):
    dataset_trajectory, u_bins, u_bin_counts = analyze_trajectory_distribution(φdot_max_mag_rps)
    train_data, test_data = split_train_test_datasets(dataset_trajectory, u_bins, 0.2)
    run_nn(train_data, test_data)
#:learn_controller_via_NN()
