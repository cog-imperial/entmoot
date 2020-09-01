"""Different benchmark problems for testing"""

import numpy as np
from entmoot.utils import is_2Dlistlike
from abc import ABC, abstractmethod
class BenchmarkFunction(ABC):

    def __init__(self, func_config={}):
        self.name = 'benchmark_function'
        self.func_config = func_config
        self.y_opt = 0.0

    def get_bounds(self, n_dim=2):
        pass

    def get_X_opt(self, n_dim=2):
        pass

    def __call__(self, X):
        # check if multiple points are given
        if not is_2Dlistlike(X):
            X = [X]

        res = []

        for x in X:
            res.append(self._eval_point(x))
        
        if len(res) == 1:
            res = res[0]

        return res
    
    @abstractmethod
    def _eval_point(self,x):
        pass

class Rosenbrock(BenchmarkFunction):

    def __init__(self, func_config={}):
        self.name = 'benchmark_function'
        self.func_config = func_config
        self.y_opt = 0.0

    def get_bounds(self, n_dim=2):
        return [(-2.048,2.048) for _ in range(n_dim)]

    def get_X_opt(self, n_dim=2):
        return [[ 1.0 for _ in range(n_dim) ]]

    def _eval_point(self, X):
        X = np.asarray_chkfinite(X)
        X0 = X[:-1]
        X1 = X[1:]

        add1 = sum( (1.0 - X0)**2.0 )
        add2 = 100.0 * sum( (X1 - X0**2.0)**2.0 )
        return add1 + add2