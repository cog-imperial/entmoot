class SamplingOptimizer:
    def __init__(self, space, params):
        raise NotImplementedError()

    def solve(self, model):
        raise NotImplementedError()

    def sample_feas(self, num_points):
        raise NotImplementedError()
