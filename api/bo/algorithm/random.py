import opti

from bo.algorithm.algorithm import Algorithm


class RandomSearch(Algorithm):
    def __init__(self, problem: opti.Problem):
        """Random selection of new data.

        Args:
            problem: Optimization problem
            data: Initial Data to use instead of the problem.data
        """
        super().__init__(problem)

    def propose(self, n_proposals=1):
        df = opti.sampling.constrained_sampling(
            n_proposals,
            parameters=self.inputs,
            constraints=self._problem.constraints,
        )
        df.index += 0 if self.data is None else len(self.data)
        return df

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {"method": "RandomSearch", "problem": self._problem.to_config()}
