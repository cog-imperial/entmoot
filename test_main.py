from entmoot.problem_config import ProblemConfig
import numpy as np

def test_func(X):
    X = np.atleast_2d(X)
    return np.sin(X[:,0])

def test_func_multi_obj(X):
    X = np.atleast_2d(X)
    y0 = np.sin(X[:,0])
    y1 = np.cos(X[:, 0])
    return np.squeeze(np.column_stack([y0,y1]))

problem_config = ProblemConfig()

problem_config.add_feature('real', (5.0, 6.0))
problem_config.add_feature('real', (4.6, 6.0))
problem_config.add_feature('real', (5.0, 6.0))
problem_config.add_feature('categorical', ("blue", "orange", "gray"))
problem_config.add_feature('integer', (5, 6))
problem_config.add_feature('binary')
problem_config.add_min_objective()
problem_config.add_min_objective()

rnd_sample = problem_config.get_rnd_sample_list(num_samples=2)

rnd_sample = problem_config.get_rnd_sample_numpy(num_samples=2)
pred = test_func_multi_obj(rnd_sample)
print(pred)

from entmoot.models.mean_models.tree_ensemble import TreeEnsemble
tree = TreeEnsemble(problem_config)
tree.fit(rnd_sample, pred)

y_vals = tree.predict(rnd_sample)

print(y_vals)

# print(rnd_sample[0])
#
# rnd_sample = space.get_rnd_sample_list(num_samples=20)
# print(rnd_sample[0])
#
# print(space)