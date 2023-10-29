
class GiniLoss:

    def __init__(self, K) -> None:
        self.K = K # number of classes
    #:

    def __call__(self, region):
        sum = 0.0
        N = region.count_examples()
        for k_minus_1 in range(self.K):
            N_k = region.count_examples_of_class(k_minus_1+1)
            p_k = N_k / N
            sum += p_k * (1.0 - p_k)
        #:
        return sum
    #:__call__()
#:class

class MisclassificationLoss:
    def __init__(self, K, k_target) -> None:
        self.K = K # number of classes
        self.k_target = k_target # in 1..K
    #:

    def __call__(self, region):
        return 1.0 \
               - max([region.count(k) / region.count() 
                      for k in range(self.K)])
    #:
#:class

