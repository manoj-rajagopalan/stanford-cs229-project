import csv
import sys

import numpy as np

from Dataset import Dataset
from Model import Model
from Loss import GiniLoss, MisclassificationLoss
import Visualize

g_rng = np.random.default_rng()

def print_usage_and_exit():
    print(f'{sys.argv[0]} <CSV file>')
#:

def load_dataset(file_path):
    X_data = []
    y_data = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        variable_names = next(reader) # skip header
        class_names = ['setosa', 'versicolor', 'virginica']
        for n, line in enumerate(reader):
            X_data.append([float(f) for f in line[:-1]])
            species = line[-1].replace('Iris-', '')
            assert species in class_names
            y = 1 if species == class_names[0] else (2 if species == class_names[1] else 3)
            y_data.append(y)
        #:form
    #:with
    return Dataset(np.array(X_data), np.array(y_data),
                   variable_names[:-1], class_names)
#:load_IRIS()

def confirm_dataset(dataset: Dataset):
    print(f'Found classes: ' + ', '.join(dataset.class_names()))
    print(f'X.shape = {dataset.X.shape}')
    print(f'y.shape = {dataset.y.shape}')
    var_lower_lims = np.floor(np.min(dataset.X, axis=0)).astype(int)
    var_upper_lims = np.ceil(np.max(dataset.X, axis=0)).astype(int)
    print(f'Found variable names: {", ".join(dataset.attribute_names())}')
    for idx, name in enumerate(dataset.attribute_names()):
        print(f'\t- {name} range: [{var_lower_lims[idx]}, {var_upper_lims[idx]}]')
    #:
#:confirm_dataset()

def split_training_test(dataset: Dataset, test_set_fraction: float):
    assert 0.0 < test_set_fraction < 1.0
    K = dataset.num_classes()
    N, d = dataset.X.shape

    # Analyze distribution over classes
    test_set_size = 0
    test_set_indices = []
    training_set_size = 0
    training_set_indices = []
    for k in np.arange(1, K+1):
        where_y_is_k = np.argwhere(dataset.y == k)
        num_examples_of_k = len(where_y_is_k)
        num_test_k = int(num_examples_of_k * test_set_fraction)
        where_y_is_k = g_rng.permutation(where_y_is_k).flatten() # shuffle
        test_set_indices.append(where_y_is_k[:num_test_k])
        test_set_size += num_test_k
        num_training_k = num_examples_of_k - num_test_k
        training_set_indices.append(where_y_is_k[num_test_k:])
        training_set_size += num_training_k
    #:for k
    assert (training_set_size + test_set_size) == N

    # Make split to be fair to all classes
    X_test = np.zeros((test_set_size, d))
    y_test = np.zeros(test_set_size, dtype=int)
    test_set_size = 0 # now use as counter

    X_training = np.zeros((training_set_size, d))
    y_training = np.zeros(training_set_size, dtype=int)
    training_set_size = 0 # now use as counter

    for k in np.arange(1, K+1):
        test_indices_k = test_set_indices[k-1]
        num_test_indices_k = len(test_indices_k)
        X_test[test_set_size : test_set_size + num_test_indices_k, :] = \
            dataset.X[test_indices_k, :]
        y_test[test_set_size : test_set_size + num_test_indices_k] = \
            dataset.y[test_indices_k]
        test_set_size += num_test_indices_k

        training_indices_k = training_set_indices[k-1]
        num_training_indices_k = len(training_indices_k)
        X_training[training_set_size : training_set_size + num_training_indices_k, :] = \
            dataset.X[training_indices_k]
        y_training[training_set_size : training_set_size + num_training_indices_k] = \
            dataset.y[training_indices_k]
        training_set_size += num_training_indices_k
    #:for k

    assert test_set_size == X_test.shape[0] == y_test.shape[0]
    assert training_set_size == X_training.shape[0] == y_training.shape[0]
    
    training_dataset = Dataset(X_training, y_training,
                               dataset.attribute_names(),
                               dataset.class_names())
    test_dataset = Dataset(X_test, y_test,
                           dataset.attribute_names(),
                           dataset.class_names())

    return training_dataset, test_dataset
#:split_training_test()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage_and_exit()
    #:
    dataset: Dataset = load_dataset(sys.argv[1])
    confirm_dataset(dataset)
    training_dataset, test_dataset = split_training_test(dataset, 0.20)
    model = Model('IRIS decision tree')
    cost_fn = GiniLoss(K=dataset.num_classes())
    model.fit(training_dataset, cost_fn)
    Visualize.render_as_dot(model.root_node, 'debug.dot')
    y_test_predicted = model.predict(test_dataset.X)
    num_mistakes = np.count_nonzero(y_test_predicted != test_dataset.y)
    num_examples = test_dataset.num_examples()
    accuracy = num_mistakes / num_examples
    print(f'Accuracy = {accuracy * 100} %  [{num_mistakes} / {num_examples}]')
#:__main__