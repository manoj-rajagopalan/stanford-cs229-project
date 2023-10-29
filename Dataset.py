import typing 
import numpy as np
import numpy.typing as nptype

class Dataset:
    def __init__(self,
                 X, y,
                 x_names: typing.List[str],
                 y_names: typing.List[str]) -> None:
        self.X = X
        self.y = y
        self.x_names = x_names
        self.y_names = y_names

        classes = np.unique(y)
        self.K = len(classes)
        assert (classes == np.arange(1, self.K+1)).all()
        self.class_counts = np.zeros(self.K)
        for k in range(self.K):
            self.class_counts[k] = np.count_nonzero(y == (k+1))
        #:
    #:

    def num_examples(self) -> int:
        return self.X.shape[0]
    #:

    def num_attributes(self) -> int:
        return self.X.shape[1]
    #:

    def attribute_names(self) -> typing.List[str]:
        return self.x_names
    #:

    def attribute_name(self, j: int) -> str:
        return self.x_names[j]
    #:

    def num_classes(self) -> int:
        return self.K
    #:

    def class_count(self, k:int) -> int:
        assert 1 <= k <= self.K
        return self.class_counts[k-1]
    #:

    def class_names(self) -> typing.List[str]:
        return self.y_names
    #:

    def class_name(self, k: int) -> str:
        return self.y_names[k-1]
    #:

#:class
