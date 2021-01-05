from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

import numpy as np


class DistanceMetric(ABC):
    """Computes distances and defines the optimization model for both 
    exploration and penalty.

    Parameters
    ----------
    -

    Attributes
    ----------
    -
    """

    @abstractmethod
    def get_distance(self, x_left, x_right):
        """Compute the distance between `x_left` and `x_right` per row of x_left.

        Parameters
        ----------
        x_left : np.array, shape (n_rows, n_dims)
            Each row of `n_rows` is a reference point. Each dimension of `n_dim` 
            is the numerical value of a continuous variable.

        x_right : np.array, shape (n_dims,)
            Each dimension is the numerical value of a continuous variable.

        Returns
        -------
        dist : np.array, shape(n_rows,)
            Distance between `x_left` and `x_right`. If multiple rows are given 
            for `x_left`, a 1-dimensional array, i.e. `n_rows` > 1, is returned.
        """

        pass

    def get_max_space_scaled_dist(self, ref_points, x_means, x_stddev, model):
        # computes maximum distance in search space
        lb = np.asarray(
            [model._cont_var_dict[i].lb for i in model._cont_var_dict.keys()]
        )
        ub = np.asarray(
            [model._cont_var_dict[i].ub for i in model._cont_var_dict.keys()]
        )

        lb_std = np.divide(lb - x_means, x_stddev)
        ub_std = np.divide(ub - x_means, x_stddev)

        max_dist = self.get_distance(
            lb_std,
            ub_std
        )
        return max_dist

class SquaredEuclidean(DistanceMetric):
    """Computes distances and defines the optimization model for both 
    exploration and penalty. The distance metric used is the squared euclidean
    distance.

    Parameters
    ----------
    -

    Attributes
    ----------
    -
    """

    @staticmethod
    def get_distance(x_left, x_right):
        """Compute the distance between `x_left` and `x_right` per row of x_left.
        Here the squared euclidean distance is used.

        Parameters
        ----------
        x_left : np.array, shape (n_rows, n_dims)
            Each row of `n_rows` is a reference point. Each dimension of `n_dim` 
            is the numerical value of a continuous variable.

        x_right : np.array, shape (n_dims,)
            Each dimension is the numerical value of a continuous variable.

        Returns
        -------
        dist : np.array, shape(n_rows,)
            Distance between `x_left` and `x_right`. If multiple rows are given 
            for `x_left`, a 1-dimensional array, i.e. `n_rows` > 1, is returned.
        """

        if x_left.ndim == 1:
            dist = np.sum((x_left - x_right)**2)
        else:
            dist = np.sum((x_left - x_right)**2, axis=1)
        return dist

    def add_exploration_to_gurobi_model(self,
        ref_points, x_means, x_stddev, distance_bound, model, add_rhs=None):
        """Adds exploration constraints to a gurobi optimization model.
        Incentivizes solutions far away from reference points.

        Parameters
        ----------
        ref_points : np.array, shape (n_rows, n_dims)
            Each row of `n_rows` is a reference point. Each dimension of `n_dim` 
            is the numerical value of a continuous variable.

        x_means : np.array, shape (n_dims,)
            Each dimension is the mean value of a continuous variable used to 
            scale the data set.

        x_stddev : np.array, shape (n_dims,)
            Each dimension is the std value of a continuous variable used to 
            scale the data set.

        distance_bound : float
            Defines the maximum value that the exploration term can take.

        Returns
        -------
        -
        """

        from gurobipy import GRB, quicksum

        n_ref_points = len(ref_points)

        # variable alpha captures distance measure
        alpha_bound = distance_bound
        model._alpha = \
            model.addVar(
                lb=0, 
                ub=alpha_bound, 
                name="alpha", 
                vtype='C'
            )

        def distance_ref_point_i(model, xi_ref, x_mean, x_stddev, add_rhs=None):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._cont_var_dict
            alpha = model._alpha
            n_features = len(xi_ref)

            diff_to_ref_point_i = quicksum(
                ( (xi_ref[j] - (c_x[key]-x_mean[j]) / x_stddev[j]) * \
                    (xi_ref[j] - (c_x[key]-x_mean[j]) / x_stddev[j]) )
                for j,key in enumerate(model._cont_var_dict.keys())
            )
            
            if add_rhs is None:
                return alpha <= diff_to_ref_point_i
            else:
                return alpha <= diff_to_ref_point_i + add_rhs

        # add exploration distances as quadratic constraints to the model
        for i in range(n_ref_points):
            if add_rhs:
                temp_rhs = add_rhs[i]
            else:
                temp_rhs = None

            model.addQConstr(
                distance_ref_point_i(
                    model, ref_points[i], x_means, x_stddev, temp_rhs
                ),
                name=f"std_const_{i}"
            )
        model.update()

    def add_penalty_to_gurobi_model(self,
        ref_points, x_means, x_stddev, model, add_rhs=None):
        """Adds penalty constraints to a gurobi optimization model.
        Incentivizes solutions close to reference points.

        Parameters
        ----------
        ref_points : np.array, shape (n_rows, n_dims)
            Each row of n_rows is a reference point. Each dimension of n_dim is 
            the numerical value of a continuous variable.

        x_means : np.array, shape (n_dims,)
            Each dimension is the mean value of a continuous variable used to 
            scale the data set.

        x_stddev : np.array, shape (n_dims,)
            Each dimension is the std value of a continuous variable used to 
            scale the data set.

        distance_bound : float
            Defines the maximum value that the exploration term can take.

        Returns
        -------
        -
        """

        from gurobipy import GRB, quicksum

        n_ref_points = len(ref_points)

        # big m is required to formulate the constraints
        model._big_m = \
            self.get_max_space_scaled_dist(ref_points, x_means, x_stddev, model)
        
        # binary variables b_ref correspond to active cluster centers
        model._b_ref = \
            model.addVars(
                range(n_ref_points),
                name="b_ref", 
                vtype=GRB.BINARY
            )

        # variable alpha captures distance measure
        model._alpha = \
            model.addVar(
                ub=GRB.INFINITY,
                lb=0.0,
                name="alpha", 
                vtype='C'
            )

        def distance_ref_point_k(model, xk_ref, k_ref, x_mean, x_stddev, add_rhs=None):
            # function returns constraints capturing the standardized
            # penalty distance
            c_x = model._cont_var_dict
            b_ref = model._b_ref
            alpha = model._alpha
            n_features = len(xk_ref)

            diff_to_ref_point_k = quicksum(
                ( (xk_ref[j]  - (c_x[key]-x_mean[j]) / x_stddev[j]) * \
                    (xk_ref[j]  - (c_x[key]-x_mean[j]) / x_stddev[j]) )
                for j,key in enumerate(model._cont_var_dict.keys())
            )
            big_m_term = model._big_m*(1-b_ref[k_ref])
            if add_rhs is None:
                return diff_to_ref_point_k <= alpha + big_m_term
            else:
                return diff_to_ref_point_k + add_rhs[k_ref] <= alpha + big_m_term

        # add penalty distances as quadratic constraints to the model
        for k in range(n_ref_points):
            model.addQConstr(
                distance_ref_point_k(
                    model, 
                    ref_points[k], 
                    k, 
                    x_means, 
                    x_stddev
                ),
                name=f"std_const_{k}"
            )

        def sum_ref_point_vars(n_ref_points, model):
            return quicksum(model._b_ref[k] \
                for k in range(n_ref_points))== 1

        # add additional sum constraints forcing only one ref_point to
        # be active
        model.addConstr(
            sum_ref_point_vars(n_ref_points, model),
            name="std_ref_sum"
        )


class Manhattan(DistanceMetric):
    """Computes distances and defines the optimization model for both 
    exploration and penalty. The distance metric used is the manhattan
    distance.

    Parameters
    ----------
    -

    Attributes
    ----------
    -
    """

    @staticmethod
    def get_distance(x_left, x_right):
        """Compute the distance between `x_left` and `x_right` per row of 
        `x_left`. Here the manhattan distance is used.

        Parameters
        ----------
        x_left : np.array, shape (n_rows, n_dims)
            Each row of `n_rows` is a reference point. Each dimension of n_dim is 
            the numerical value of a continuous variable.

        x_right : np.array, shape (n_dims,)
            Each dimension is the numerical value of a continuous variable.

        Returns
        -------
        dist : np.array, shape(n_rows,)
            Distance between `x_left` and `x_right`. If multiple rows are given 
            for `x_left`, a 1-dimensional array, i.e. `n_rows` > 1, is returned.
        """

        if x_left.ndim == 1:
            dist = np.sum( np.abs(x_left - x_right) )
        else:
            dist = np.sum( np.abs(x_left - x_right), axis=1)
        return dist

    def add_exploration_to_gurobi_model(self,
        ref_points, x_means, x_stddev, distance_bound, model, add_rhs=None):
        """Adds exploration constraints to a gurobi optimization model.
        Incentivizes solutions far away from reference points.

        Parameters
        ----------
        ref_points : np.array, shape (n_rows, n_dims)
            Each row of `n_rows` is a reference point. Each dimension of `n_dim` 
            is the numerical value of a continuous variable.

        x_means : np.array, shape (n_dims,)
            Each dimension is the mean value of a continuous variable used to 
            scale the data set.

        x_stddev : np.array, shape (n_dims,)
            Each dimension is the std value of a continuous variable used to 
            scale the data set.

        distance_bound : float
            Defines the maximum value that the exploration term can take.

        Returns
        -------
        -
        """

        from gurobipy import GRB, quicksum

        n_ref_points = len(ref_points)
        cont_feat_ids = model._cont_var_dict.keys()

        # two sets of variables are used to capture positive and negative 
        # parts of manhattan distance
        model._c_x_aux_pos = \
            model.addVars(range(n_ref_points), cont_feat_ids, 
                        name="c_x_aux_pos", vtype='C')

        model._c_x_aux_neg = \
            model.addVars(range(n_ref_points), cont_feat_ids, 
                        name="c_x_aux_neg", vtype='C')
        
        # variable alpha captures distance measure
        alpha_bound = distance_bound
        model._alpha = \
            model.addVar(lb=0, 
                        ub=alpha_bound, 
                        name="alpha", vtype='C')

        def distance_ref_point_i_for_feat_j(
            model, xi_ref, i_ref, feat_j, i_var, x_mean, x_stddev, add_rhs=None):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._cont_var_dict

            diff_to_ref_point_i = \
                ( xi_ref[feat_j] - (c_x[i_var]-x_mean[feat_j]) / \
                    x_stddev[feat_j] ) 
            return diff_to_ref_point_i == model._c_x_aux_pos[i_ref, i_var] - \
                model._c_x_aux_neg[i_ref, i_var]

        for i_ref in range(n_ref_points):
            for feat_j,key in enumerate(cont_feat_ids):
                # add constraints to capture distances in variables
                # _c_x_aux_pos and _c_x_aux_neg
                model.addConstr(
                    distance_ref_point_i_for_feat_j(model, 
                        ref_points[i_ref], i_ref, feat_j, key, x_means, x_stddev),
                    name=f"std_const_feat_{key}_{i_ref}"
                )
                # add sos constraints that allow only one of the +/- vars, 
                # i.e. _c_x_aux_pos / _c_x_aux_neg to be active
                model.addSOS(GRB.SOS_TYPE1, 
                    [
                        model._c_x_aux_pos[i_ref, key], 
                        model._c_x_aux_neg[i_ref, key]
                    ]
                )

            # add exploration distances as linear constraints to the model
            if add_rhs is None:
                model.addConstr(
                    model._alpha <= quicksum(
                        (model._c_x_aux_pos[i_ref, j] + model._c_x_aux_neg[i_ref, j])
                        for j in cont_feat_ids
                    ),
                    name=f"alpha_sum"
                )
            else:
                model.addConstr(
                    model._alpha <= quicksum(
                        (model._c_x_aux_pos[i_ref, j] + model._c_x_aux_neg[i_ref, j])
                        for j in cont_feat_ids
                    ) + add_rhs[i_ref],
                    name=f"alpha_sum"
                )            
        model.update()

    def add_penalty_to_gurobi_model(self,
        ref_points, x_means, x_stddev, model, add_rhs=None):
        """Adds penalty constraints to a gurobi optimization model.
        Incentivizes solutions close to reference points.

        Parameters
        ----------
        ref_points : np.array, shape (n_rows, n_dims)
            Each row of `n_rows` is a reference point. Each dimension of `n_dim` 
            is the numerical value of a continuous variable.

        x_means : np.array, shape (n_dims,)
            Each dimension is the mean value of a continuous variable used to 
            scale the data set.

        x_stddev : np.array, shape (n_dims,)
            Each dimension is the std value of a continuous variable used to 
            scale the data set.

        distance_bound : float
            Defines the maximum value that the exploration term can take.

        Returns
        -------
        -
        """

        from gurobipy import GRB, quicksum

        n_ref_points = len(ref_points)
        cont_feat_ids = model._cont_var_dict.keys()

        # big m is required to formulate the constraints
        model._big_m = \
            self.get_max_space_scaled_dist(ref_points, x_means, x_stddev, model)
        
        # two sets of variables are used to capture positive and negative 
        # parts of manhattan distance
        model._c_x_aux_pos = \
            model.addVars(range(n_ref_points), cont_feat_ids, 
                        name="c_x_aux_pos", vtype='C')

        model._c_x_aux_neg = \
            model.addVars(range(n_ref_points), cont_feat_ids, 
                        name="c_x_aux_neg", vtype='C')
        
        # binary variables b_ref correspond to active cluster centers
        model._b_ref = \
            model.addVars(n_ref_points,
                        name="b_ref", vtype=GRB.BINARY)

        # variable alpha captures distance measure
        model._alpha = \
            model.addVar(ub=GRB.INFINITY,
                        lb=0.0,
                        name="alpha", vtype='C')

        def distance_ref_point_i_for_feat_j(
            model, xi_ref, i_ref, feat_j, i_var, x_mean, x_stddev, add_rhs=None):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._cont_var_dict

            diff_to_ref_point_i = \
                ( xi_ref[feat_j] - (c_x[i_var]-x_mean[feat_j]) / \
                    x_stddev[feat_j] ) 
            return diff_to_ref_point_i == model._c_x_aux_pos[i_ref, i_var] - \
                model._c_x_aux_neg[i_ref, i_var]

        for i_ref in range(n_ref_points):
            for feat_j,key in enumerate(cont_feat_ids):
                # add constraints to capture distances in variables
                # _c_x_aux_pos and _c_x_aux_neg
                model.addConstr(
                    distance_ref_point_i_for_feat_j(
                        model, ref_points[i_ref], i_ref, feat_j, key, x_means, x_stddev),
                    name=f"std_const_feat_{key}_{i_ref}"
                )
                # add sos constraints that allow only one of the +/- vars, 
                # i.e. _c_x_aux_pos / _c_x_aux_neg to be active
                model.addSOS(GRB.SOS_TYPE1, 
                    [
                        model._c_x_aux_pos[i_ref, key], 
                        model._c_x_aux_neg[i_ref, key]
                    ]
                )

        # add penalty distances as linear constraints to the model
            if add_rhs is None:
                model.addConstr(
                    quicksum(
                        (model._c_x_aux_pos[i_ref, j] + model._c_x_aux_neg[i_ref, j])
                        for j in cont_feat_ids
                    ) <= model._alpha + model._big_m*(1-model._b_ref[i_ref]),
                    name=f"alpha_sum"
                )
            else:
                model.addConstr(
                    quicksum(
                        (model._c_x_aux_pos[i_ref, j] + model._c_x_aux_neg[i_ref, j])
                        for j in cont_feat_ids
                    )  + add_rhs[i_ref] <= \
                        model._alpha + model._big_m*(1-model._b_ref[i_ref]),
                    name=f"alpha_sum"
                )    
        model.update()

        def sum_ref_point_vars(n_ref_points, model):
            return quicksum(model._b_ref[k] for k in range(n_ref_points))== 1

        # add additional sum constraints forcing only one ref_point to
        # be active
        model.addConstr(
            sum_ref_point_vars(n_ref_points, model),
            name="std_ref_sum"
        )

class NonSimilarityMetric:
    def __init__(self):
        pass

    def get_non_similarity(self,x_left,x_right):
        dsim = self.get_similarity(x_left, x_right)
        n_cat = x_left.shape[1]
        
        dsim = (-1) * dsim + n_cat*1
        return dsim

    def update(self, Xi_cat, cat_dims):
        pass

    def get_gurobi_model_rhs(self,cat_idx,ref_points,model):
        if cat_idx:
            constr_list = []
            for X in ref_points:
                temp_constr = 0

                for cat_id in range(len(X)):
                    for cat in range(len(self.cat_mat[cat_id])):
                        temp_cat_mat = self.cat_mat[cat_id][cat,X[cat_id]]
                        temp_constr += \
                            (1 - temp_cat_mat)*model._cat_var_dict[cat_idx[cat_id]][cat]

                constr_list.append(temp_constr)
            return constr_list
        else:
            return None


class Overlap(NonSimilarityMetric):
    def __init__(self):
        pass

    def update(self,Xi_cat,cat_dims):

        Xi_cat = np.asarray(Xi_cat)

        # generate matrix rules
        cat_mat_rule = \
            lambda i,j,cat_id: 1 if i==j else 0

        self.cat_mat = []

        for cat_id,cat in enumerate(cat_dims):
            temp_cat = cat.transform(cat.categories)
            n_cat = len(temp_cat)

            # populate matrix
            temp_mat = np.fromfunction(
                np.vectorize(cat_mat_rule,), (n_cat,n_cat), dtype=int, cat_id=cat_id
            )

            self.cat_mat.append(temp_mat)

    def get_similarity(self,x_left,x_right):
        dist_cat = np.sum(x_left == x_right, axis=1)
        return dist_cat


class Goodall4(NonSimilarityMetric):
    def __init__(self):
        pass

    def update(self,Xi_cat,cat_dims):

        def get_pk2(cat_rows, cat):
            count_cat = np.sum(cat_rows == cat)
            n_rows = len(cat_rows)
            return ( count_cat*(count_cat-1) ) / ( n_rows*(n_rows-1) )

        Xi_cat = np.asarray(Xi_cat)

        # generate matrix rules
        cat_mat_rule = \
            lambda i,j,cat_id: get_pk2( Xi_cat[:,cat_id],i ) \
                if i == j else 0

        self.cat_mat = []

        for cat_id,cat in enumerate(cat_dims):
            temp_cat = cat.transform(cat.categories)
            n_cat = len(temp_cat)

            # populate matrix
            temp_mat = np.fromfunction(
                np.vectorize(cat_mat_rule,), (n_cat,n_cat), dtype=int, cat_id=cat_id
            )

            self.cat_mat.append(temp_mat)

    def get_similarity(self,x_left,x_right):
        # create zero vector that contains similarities for all data points
        sim_vec = np.zeros(x_left.shape[0])

        # iterate through cat features and populate sim_mat entries
        for cat_id in range(x_left.shape[1]):

            # compute individual similarities
            sim_vec += \
                self.cat_mat[cat_id][int(x_right[cat_id]),x_left[:,cat_id]]
        return sim_vec

class OF(NonSimilarityMetric):
    def __init__(self):
        pass

    def update(self,Xi_cat,cat_dims):

        def get_of_frac(cat_rows, cat_left, cat_right):
            from math import log
            count_cat_left = np.sum(cat_rows == cat_left)
            count_cat_right = np.sum(cat_rows == cat_right)
            n_rows = len(cat_rows)
            denom = \
                1 + log( n_rows/count_cat_left )*log( n_rows/count_cat_right )
            return 1 / denom

        Xi_cat = np.asarray(Xi_cat)

        # generate matrix rules
        cat_mat_rule = \
            lambda i,j,cat_id: get_of_frac( Xi_cat[:,cat_id], i, j ) \
                if i is not j else 1

        self.cat_mat = []

        for cat_id,cat in enumerate(cat_dims):
            temp_cat = cat.transform(cat.categories)
            n_cat = len(temp_cat)

            # populate matrix
            temp_mat = np.fromfunction(
                np.vectorize(cat_mat_rule,), (n_cat,n_cat), dtype=int, cat_id=cat_id
            )

            self.cat_mat.append(temp_mat)

    def get_similarity(self,x_left,x_right):
        # create zero vector that contains similarities for all data points
        sim_vec = np.zeros(x_left.shape[0])

        # iterate through cat features and populate sim_mat entries
        for cat_id in range(x_left.shape[1]):

            # compute individual similarities
            sim_vec += \
                self.cat_mat[cat_id][int(x_right[cat_id]),x_left[:,cat_id]]
        return sim_vec


class DistanceBasedStd(ABC):
    """Define a distance-based standard estimator.

    A `DistanceBasedStd` object is used to quantify model uncertainty based 
    on distance to reference points, e.g. data points. The underlying assumption 
    is that base estimator predictions are good close to training data.

    Use this class as a template if you want to develop your own distance-based
    measure.

    Parameters
    ----------
    metric : string
        Metric used to compute distances, e.g. squared euclidean, manhattan

    Attributes
    ----------
    Xi : list
        Points at which objective has been evaluated.
    yi : scalar
        Values of objective at corresponding points in `Xi`.
    cont_dist_metric : DistanceMetric
        Object used to compute distances between continuous variables.
    x_means : list
        Mean of attribute `Xi`.
    x_scaler : list
        Scalers of attribute `Xi`.
    Xi_standard : list
        Standardized `Xi` array.
    ref_points : list
        Points to which the distance is computed to estimate model uncertainty.
        Is different for all child classes.
    """
    def __init__(self,
        metric_cont='sq_euclidean',
        metric_cat='overlap',
        space=None):

        # set distance metric for cont variables
        if metric_cont == 'sq_euclidean':
            from entmoot.learning.distance_based_std import SquaredEuclidean
            self.cont_dist_metric = SquaredEuclidean()

        elif metric_cont == 'manhattan':
            from entmoot.learning.distance_based_std import Manhattan
            self.cont_dist_metric = Manhattan()

        self.space = space

        # set similarity metric for cat variables
        if metric_cat == "overlap":
            from entmoot.learning.distance_based_std import Overlap
            self.cat_sim_metric = Overlap()
            
        elif metric_cat == "goodall4":
            from entmoot.learning.distance_based_std import Goodall4
            assert self.space is not None, "Space parameter is needed for goodall4."
            self.cat_sim_metric = Goodall4()

        elif metric_cat == "of":
            from entmoot.learning.distance_based_std import OF
            assert self.space is not None, "Space parameter is needed for of."
            self.cat_sim_metric = OF()

    def get_x_cont_vals(self, xi):
        return [x for idx,x in enumerate(xi) \
            if idx not in self.cat_idx]

    def get_x_cat_vals(self, xi):
        return [x for idx,x in enumerate(xi) \
            if idx in self.cat_idx]


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
        pass

    def update(self, Xi, yi, cat_column=[]):
        """Update available data points which is usually done after every
        iteration.

        Parameters
        ----------
        Xi : list
            Points at which objective has been evaluated.
        yi : scalar
            Values of objective at corresponding points in `Xi`.

        Returns
        -------
        -
        """

        self.cat_idx = cat_column
        
        # collect cont and cat variable values
        if self.cat_idx:
            self.Xi_cont = []
            self.Xi_cat = []
            
            for xi in Xi:

                # check if all 

                # collect continuous vars
                # check if all vars are categorical
                if len(self.cat_idx) != len(xi):
                    self.Xi_cont.append( [x for idx,x in enumerate(xi) \
                        if idx not in self.cat_idx]
                    )

                # collect categorical vars
                self.Xi_cat.append( [x for idx,x in enumerate(xi) \
                    if idx in self.cat_idx]
                )
        else:
            self.Xi_cont = Xi
            self.Xi_cat = []

        # update data set attributes
        self.Xi = Xi
        self.yi = yi

        self.n_features = Xi.shape[1]

        # compute mean and scaler of data set
        if list(self.Xi_cont):
            standard_scaler = StandardScaler()
            projected_features = standard_scaler.fit_transform(self.Xi_cont)
            self.x_means = standard_scaler.mean_
            self.x_scalers = standard_scaler.scale_

            # standardize dataset
            self.Xi_cont_standard = self.standardize_with_Xi(self.Xi_cont)

        # compute scale coefficient
        y_scaler = np.std(self.yi)
        self.std_scale_coef = abs(y_scaler / self.n_features)

        # update cat_sim_metric with new data
        cat_dims = self.get_x_cat_vals(self.space.dimensions)
        self.cat_sim_metric.update(self.Xi_cat, cat_dims)

    def standardize_with_Xi(self, X):
        """Standardize given input `X` based on attribute `Xi`.

        Parameters
        ----------
        X : numpy array, shape (n_rows, n_dims)
            Each row of n_rows is a point in `X`.
            Points which are standardized based on `Xi`

        Returns
        -------
        x_standard : numpy array, shape (n_rows, n_dims)
            Standardized array of `X`
        """
        x_standard = np.divide(X - self.x_means, self.x_scalers)
        return x_standard

    def get_closest_point_distance(self, X):
        """Get distance to point of attribute `ref_points` which is closest to 
        point given as parameter `X`.

        Parameters
        ----------
        X : numpy array, shape (n_dims,)
            Point to which the distance of closest reference point is 
            computed

        Returns
        -------
        dist : numpy array, shape (1, )
            Returns distance to closest `ref_point`.
        """
        if list(self.Xi_cont):
            ref_points_cont = np.asarray(self.ref_points_cont)

            X_cont = self.get_x_cont_vals(X)

            # standardize cont variables
            x_cont_standard = self.standardize_with_Xi(X_cont)  
            x_cont_standard = np.asarray(x_cont_standard)

            # compute all distances for cont variables
            dist_cont = \
                self.cont_dist_metric.get_distance(ref_points_cont,x_cont_standard)

        # compute all distances for cat variables
        if self.Xi_cat:
            X_cat = X[self.cat_idx]
            dist_cat = \
                self.cat_sim_metric.get_non_similarity(
                    self.ref_points_cat, X_cat)

            if list(self.Xi_cont):
                dist_cont += dist_cat
            else:
                dist_cont = dist_cat

        return np.min(dist_cont)

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
            ref_distance = self.get_closest_point_distance(Xi)
            dist[row_res] = ref_distance

        if scaled:
            dist *= self.std_scale_coef

        return dist

    @abstractmethod
    def add_to_gurobi_model(self,model):
        """Add standard estimator to gurobi model. Model details are 
        implemented in child class.

        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.
        """
        pass

    @abstractmethod
    def get_gurobi_obj(self,model):
        """Get contribution of standard estimator to gurobi model objective
        function.

        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.

        s
        ----------
        alpha : gurobipy.Var,
            Model variable that takes the value of the uncertainty measure.
        """
        pass


class DistanceBasedExploration(DistanceBasedStd):
    """Defines a child class based on `DistanceBasedStd`. Exploration 
    refers to how the distance measure contributes to the acquisition
    function. Exploration refers to incentivizing distance to reference points
    leading to a negative contribution of the distance measure to the objective
    function.
    
    Parameters
    ----------
    metric : string
        Metric used to compute distances, e.g. squared euclidean, manhattan
    zeta : scalar
        Coefficient determining how the distance measure is bounded

    Attributes
    ----------
    Xi : list
        Points at which objective has been evaluated.
    yi : scalar
        Values of objective at corresponding points in `Xi`.
    cont_dist_metric : DistanceMetric
        Object used to compute distances between continuous variables.
    x_means : list
        Mean of attribute `Xi`.
    x_scaler : list
        Scalers of attribute `Xi`.
    Xi_standard : list
        Standardized `Xi` array.
    ref_points : list
        `ref_points` standardized to which the distance measure is computed.
    ref_points_unscaled : list
        Unscaled `ref_points` to which the distance measure is computed.
    distance_bound : scalar
        Bound of exploration measure to prohibit over-exploration
    """
    def __init__(self,
        metric_cont='sq_euclidean',
        metric_cat='overlap',
        space=None,
        zeta=0.5):

        super().__init__(metric_cont,metric_cat,space)
        self.zeta = zeta

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

    def update(self, Xi, yi, cat_column=[]):
        """Update available data points which is usually done after every
        iteration.

        Parameters
        ----------
        Xi : list
            Points at which objective has been evaluated.
        yi : scalar
            Values of objective at corresponding points in `Xi`.

        Returns
        -------
        -
        """
        super().update(Xi, yi, cat_column=cat_column)

        self.ref_points_unscaled = self.Xi_cont

        #self.ref_points = self.Xi_standard

        if list(self.Xi_cont):
            self.ref_points_cont = self.Xi_cont_standard
        self.ref_points_cat = np.asarray(self.Xi_cat, dtype=int)

        # compute upper bound of uncertainty
        y_var = np.var(yi)
        self.distance_bound = abs(self.zeta*y_var)

    def predict(self, X, scaled=True):
        """Predict standard estimate at location `X`. By default `dist` is
        bounded by attribute `distance_bound`.

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
        dist = super().predict(X, scaled=scaled)

        # prediction has max out at `distance_bound`
        dist[dist > self.distance_bound] = self.distance_bound
        return dist

    def get_gurobi_obj(self, model, scaled=True):
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
        # negative contributation of alpha requires non-convex flag in gurobi.
        model.Params.NonConvex=2
        if scaled:
            return -self.std_scale_coef*model._alpha
        else:
            return -model._alpha

    def add_to_gurobi_model(self,model):
        """Add standard estimator to gurobi model. Model details are 
        implemented in child class.

        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.
        """
        cat_rhs = self.cat_sim_metric.get_gurobi_model_rhs(
            self.cat_idx, self.ref_points_cat, model
        )

        self.cont_dist_metric.add_exploration_to_gurobi_model(
            self.ref_points_cont, 
            self.x_means, 
            self.x_scalers, 
            self.distance_bound, 
            model,
            add_rhs=cat_rhs
        )

class DistanceBasedPenalty(DistanceBasedStd):
    """Defines a child class based on `DistanceBasedStd`. Penalty 
    refers to how the distance measure contributes to the acquisition
    function. Penalty refers to penalizing distance to reference points
    leading to a positive contribution of the distance measure to the objective
    function.
    
    Parameters
    ----------
    metric : string
        Metric used to compute distances, e.g. squared euclidean, manhattan

    Attributes
    ----------
    Xi : list
        Points at which objective has been evaluated.
    yi : scalar
        Values of objective at corresponding points in `Xi`.
    cont_dist_metric : DistanceMetric
        Object used to compute distances between continuous variables.
    x_means : list
        Mean of attribute `Xi`.
    x_scaler : list
        Scalers of attribute `Xi`.
    Xi_standard : list
        Standardized `Xi` array.
    ref_points : list
        `ref_points` reference to which the distance measure is computed
    n_ref_points : scalar
        length of `ref_points"""

    def __init__(self,
        metric_cont='sq_euclidean',
        metric_cat='overlap',
        space=None):

        super().__init__(metric_cont,metric_cat,space)

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
        pass

    def update(self, Xi, yi, cat_column=[]):
        """Update available data points which is usually done after every
        iteration.

        Parameters
        ----------
        Xi : list
            Points at which objective has been evaluated.
        yi : scalar
            Values of objective at corresponding points in `Xi`.

        Returns
        -------
        -
        """
        super().update(Xi, yi, cat_column=cat_column)

        self.ref_points_unscaled = self.Xi_cont

        #self.ref_points = self.Xi_standard

        if list(self.Xi_cont):
            self.ref_points_cont = self.Xi_cont_standard
        self.ref_points_cat = np.asarray(self.Xi_cat, dtype=int)


    def predict(self, X, scaled=True):
        """Predict standard estimate at location `X`. Sign of `dist` is negative
        because it contributes as a penalty.

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
        dist = super().predict(X, scaled=scaled)
        return -dist

    def get_gurobi_obj(self, model, scaled=True):
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
        if scaled:
            return self.std_scale_coef*model._alpha
        else:
            return model._alpha

    def add_to_gurobi_model(self,model):
        """Add standard estimator to gurobi model. Model details are 
        implemented in child class.

        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.
        """
        cat_rhs = self.cat_sim_metric.get_gurobi_model_rhs(
            self.cat_idx, self.ref_points_cat, model
        )

        self.cont_dist_metric.add_penalty_to_gurobi_model(
            self.ref_points_cont, 
            self.x_means, 
            self.x_scalers, 
            model,
            add_rhs=cat_rhs
        )