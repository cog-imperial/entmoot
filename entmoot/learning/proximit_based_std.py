from .distance_based_std import DistanceMetric
import numpy as np

class ProximityMetric(DistanceMetric):

    def __init__(self, space):
        self.space = space
        self.std_type = 'proximity'
        self.active_leaves_per_data = []
        self.obj_label = 0

    @staticmethod
    def get_distance(x_left, x_right):
        pass

    def add_to_gurobi_model(self,model):

        model._alpha = \
            model.addVar(
                lb=0, 
                ub=1, 
                name="alpha", 
                vtype='C'
            )

        # iterate over all leaf activations for data points
        for data_id, data_enc in enumerate(self.active_leaves_per_data):
            temp_lhs = 0
            for tree_id, leaf_enc in enumerate(data_enc):
                temp_lhs += model._z_l[self.obj_label, tree_id, leaf_enc]

            temp_constr = 1-1/len(data_enc)*temp_lhs >= model._alpha
            model.addConstr(
                temp_constr,
                name=f"prox_{self.obj_label}_{data_id}"
            )
        model.update()

    def update(self, Xi, yi, gbm_model, cat_column=[]):
        # collect active leaves for all datapoints

        self.gbm_model = gbm_model
        self.active_leaves_per_data = np.asarray(
            [
                self.gbm_model.get_active_leaves(x) for x in Xi
            ]
        )

    def set_params(self,**kwargs):
        """Sets parameters related to distance-based standard estimator.

        Parameters
        ----------
        kwargs : dict
            Additional arguments to be passed to the standard estimator

        Returns
        -------
        -
        """
        zeta = kwargs.get("zeta", 0.5)
        self.zeta = zeta

    def predict(self, X, scaled=True):
        """Predict standard estimate at location `X`.

        Parameters
        ----------
        X : numpy array, shape (n_rows, n_dims)
            Points at which the standard estimator is evaluated.

        Returns
        -------
        dist : numpy array, shape (n_rows,)
            Returns distances to closest `ref_point` for every point per row
            in `n_rows`.
        """
        dist = np.empty([X.shape[0],])

        for row_res,Xi in enumerate(X):
            ref_dist = self.get_distance(Xi)
            dist[row_res] = ref_dist
        return dist

    def get_distance(self, X):
        active_leaves = np.asarray(self.gbm_model.get_active_leaves(X))

        #compute overlap of active leaves for X and the entire dataset 
        overlap_mat = self.active_leaves_per_data == active_leaves

        # scaling with number of trees and substracting from 1 makes distance
        prox_mat = (-np.sum(overlap_mat,1)/len(active_leaves)) + 1
        return np.min(prox_mat)

    def get_gurobi_obj(self, model, scaled=False):
        """Get contribution of standard estimator to gurobi model objective
        function.

        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.

        Returns
        -------
        alpha : gurobipy.Var,
            Model variable that takes the value of the uncertainty measure.
        """

        return -model._alpha