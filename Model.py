import enum
import numpy as np

from Dataset import Dataset
from Region import Region
from DecisionTreeNode import TerminalNode, SplitNode

class Model:

    def __init__(self, name):
        self.name = name
    #:

    def fit(self, dataset, cost_fn):
        Region.bind_dataset(dataset)
        region = Region()
        region.set_cost(cost_fn(region))
        self.root_node = self.make_decision_tree(dataset, cost_fn, region)
    #:fit()

    def make_decision_tree(self, dataset, cost_fn, region):
        # Greedily identify optimal split point.
        split_j = -1 # which var to split on.
        split_n = -1 # index of value to split on for that var.
        split_regions = None
        min_cost = np.inf
        d = dataset.X.shape[1] # number of decision vars
        for j in range(d):
            dim_j_lower_lim_idx, dim_j_upper_lim_idx = region.limits_indices(j)
            for n in range(dim_j_lower_lim_idx+1, dim_j_upper_lim_idx):
                L, R = region.split(j, n)
                if L is None or R is None:
                    continue # no split
                #:
                L.set_cost(cost_fn(L))
                R.set_cost(cost_fn(R))
                assert region.count_examples() == L.count_examples() + R.count_examples()
                weighted_cost = (L.count_examples() * L.cost + R.count_examples() * R.cost) / region.count_examples()
                if weighted_cost < min_cost: # new min
                    split_j = j
                    split_n = n
                    split_regions = (L,R)
                #:if
            #: for n
        #: for j

        # Make the split
        assert (split_j >= 0) == (split_n >= 0) == (split_regions is not None)
        node = None # return value
        if split_regions is None:
            assert split_j < 0 and split_n < 0
            majority_k = region.majority_class()
            node = TerminalNode(region, majority_k)
        else:
            L, R = split_regions
            node = SplitNode(region, split_j, split_n)
            theta = Region.split_points[split_j][split_n]
            child_node_L = self.make_decision_tree(dataset, cost_fn, L)
            child_node_R = self.make_decision_tree(dataset, cost_fn, R)
            node.add_child(child_node_L)
            node.add_child(child_node_R)
        #:if split_regions

        return node
    #:make_decision_tree()

    def predict(self, X):
        N = X.shape[0]  
        y = np.zeros(N, dtype=int)
        for n in range(N):
            y[n] = self.predict_sample(X[n])
        #:
        return y
    #:predict()

    def predict_sample(self, x) -> int:
        y = np.nan
        node = self.root_node
        while node:
            if node.is_terminal():
                y = node(x)
                node = None # break out of loop
            else:
                assert node.is_split()
                if node(x):
                    node = node.children[0]
                else:
                    node = node.children[1]
                #:if
            #:if
        #:while node
        return y
    #:predict_sample()

#:class
