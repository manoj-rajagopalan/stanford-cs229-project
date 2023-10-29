import copy
import typing

# https://realpython.com/python-type-self/
# import typing_extensions

import numpy as np

from Dataset import Dataset
from Predicate import Predicate_LessThan, Predicate_GreaterEqual

class Region:
    '''
    A tuple of intervals, one for each field in the dataset.
    Each interval is stored as a pair of indices [lower_limit, upper_limit)
    where the indices point into a 'splits' array.
    The 'splits' array is an ordered sequence of points at which a field can be split.
    These points are simply mid-points between the sorted values of this field found in the training dataset.
    If the training dataset contains N examples, then len(splits) == (N-1).
    Initially, lower_limit is set to -1 so that it contains the first sorted-field-value.
    The value of upper_limit for the "last" region (along an axis) is always (N-1) so that
    the last sorted-field-value is captured.
    '''

    # class variables
    dataset = None
    split_points = None # (N-1) x d table of values to split each dimension along

    @classmethod
    def bind_dataset(cls, dataset: Dataset) -> None:
        Region.dataset = dataset
        N, d = dataset.X.shape
        Region.split_points = np.zeros((N-1,d))
        for j in range(d):
            x_j_sorted = np.sort(Region.dataset.X[:,j])
            Region.split_points[:,j] = 0.5 * (x_j_sorted[:-1] + x_j_sorted[1:])
        #:for j
    #:compute_splits()

    def __init__(self) -> None:
        '''
        Create the root region.
        All other regions must be created by calling split().
        '''
        N, d = Region.dataset.X.shape
        self.dataset_indices = np.arange(N) # all indices
        self.lower_limit_indices = np.full(d, -1)
        self.upper_limit_indices = np.full(d, N-1)
        self.predicate = None
    #:__init__()

    def copy(self) -> typing.Self:
        r = Region()
        r.dataset_indices = self.dataset_indices.copy()
        r.lower_limit_indices = self.lower_limit_indices.copy()
        r.upper_limit_indices = self.upper_limit_indices.copy()
        r.predicate = copy.deepcopy(self.predicate)
        return r
    #:

    def set_cost(self, cost: float) -> None:
        self.cost = cost
    #:

    def count_examples(self) -> int:
        return len(self.dataset_indices)
    #:

    def count_examples_of_class(self, k: int) -> int:
        '''
        k: desired class to count, in 1..K
        '''
        assert 1 <= k <= self.dataset.num_classes()
        return np.count_nonzero(self.dataset.y[self.dataset_indices] == k)
    #:

    def majority_class(self) -> int:
        K = self.dataset.num_classes()
        class_counts = [self.count_examples_of_class(k+1)
                        for k in range(K)]
        assert sum(class_counts) == self.count_examples()
        max_idx = np.argmax(class_counts)
        return max_idx + 1 # classes are numbered starting with 1
    #:

    def limits_indices(self, j):
        return np.array([self.lower_limit_indices[j], self.upper_limit_indices[j]])
    #:

    def limits(self, j):
        lower_limit = -np.inf if self.lower_limit_indices[j] < 0 else Region.split_points[self.lower_limit_indices[j]]
        N = len(Region.split_points) + 1
        upper_limit = np.inf if self.upper_limit_indices[j] >= (N-1) else Region.split_points[self.upper_limit_indices[j]]
        return np.array([lower_limit, upper_limit])
    #:

    def split(self, j, n) -> typing.Tuple[typing.Self, typing.Self]:
        assert self.lower_limit_indices[j] <= n < self.upper_limit_indices[j]
        theta = Region.split_points[j][n]

        # Left child region
        L: Region = self.copy()
        L.predicate = Predicate_LessThan(theta)
        L.upper_limit_indices[j] = n
        predicate_local_indices = np.argwhere(self.dataset.X[self.dataset_indices][:,j] < theta).flatten()
        L.dataset_indices = self.dataset_indices[predicate_local_indices] # global indices
        assert len(L.dataset_indices) > 0
        
        # Right child region
        R: Region = self.copy()
        R.predicate = Predicate_GreaterEqual(theta)
        R.lower_limit_indices[j] = n
        predicate_local_indices = np.argwhere(self.dataset.X[self.dataset_indices][:,j] >= theta).flatten()
        R.dataset_indices = self.dataset_indices[predicate_local_indices] # global indices
        assert len(R.dataset_indices) > 0

        assert len(L.dataset_indices) + len(R.dataset_indices) == len(self.dataset_indices)
        return L, R
    #:split()

    def __str__(self) -> str:
        d = len(self.field_names)
        intervals = []
        for j in range(d):
            lower_limit_j, upper_limit_j = self.limits(j)
            intervals.append(f'{lower_limit_j} <= {self.field_names[j]} < {upper_limit_j}')
        #:
        s = '(' + ', '.join(intervals) + ')'
        return s
    #:__str__()
#:class Region
