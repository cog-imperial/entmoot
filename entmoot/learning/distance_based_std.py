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
        n_features = len(model._c_x)
        lb = np.asarray(model._c_x_lb)
        ub = np.asarray(model._c_x_ub)

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
        ref_points, x_means, x_stddev, distance_bound, model):
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

        def distance_ref_point_i(model, xi_ref, x_mean, x_stddev):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._c_x
            alpha = model._alpha
            n_features = len(xi_ref)

            diff_to_ref_point_i = quicksum(
                ( (xi_ref[j] - (c_x[j]-x_mean[j]) / x_stddev[j]) * \
                    (xi_ref[j] - (c_x[j]-x_mean[j]) / x_stddev[j]) )
                for j in range(n_features)
            )
            
            return alpha <= diff_to_ref_point_i

        # add exploration distances as quadratic constraints to the model
        for i in range(n_ref_points):
            model.addQConstr(
                distance_ref_point_i(model, ref_points[i], x_means, x_stddev),
                name=f"std_const_{i}"
            )
        model.update()

    def add_penalty_to_gurobi_model(self,
        ref_points, x_means, x_stddev, model):
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
                vtype=GRB.INTEGER
            )

        # variable alpha captures distance measure
        model._alpha = \
            model.addVar(
                ub=GRB.INFINITY,
                lb=0.0,
                name="alpha", 
                vtype='C'
            )

        def distance_ref_point_k(model, xk_ref, k_ref, x_mean, x_stddev):
            # function returns constraints capturing the standardized
            # penalty distance
            c_x = model._c_x
            b_ref = model._b_ref
            alpha = model._alpha
            n_features = len(xk_ref)

            diff_to_ref_point_k = quicksum(
                ( (xk_ref[j]  - (c_x[j]-x_mean[j]) / x_stddev[j]) * \
                    (xk_ref[j]  - (c_x[j]-x_mean[j]) / x_stddev[j]) )
                for j in range(n_features)
            )
            big_m_term = model._big_m*(1-b_ref[k_ref])
            return diff_to_ref_point_k <= alpha + big_m_term

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
        ref_points, x_means, x_stddev, distance_bound, model):
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

        # two sets of variables are used to capture positive and negative 
        # parts of manhattan distance
        model._c_x_aux_pos = \
            model.addVars(range(n_ref_points), range(model._n_feat), 
                        name="c_x_aux_pos", vtype='C')

        model._c_x_aux_neg = \
            model.addVars(range(n_ref_points), range(model._n_feat), 
                        name="c_x_aux_neg", vtype='C')
        
        # variable alpha captures distance measure
        alpha_bound = distance_bound
        model._alpha = \
            model.addVar(lb=0, 
                        ub=alpha_bound, 
                        name="alpha", vtype='C')

        def distance_ref_point_i_for_feat_j(
            model, xi_ref, i_ref, feat_j, x_mean, x_stddev):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._c_x

            diff_to_ref_point_i = \
                ( xi_ref[feat_j] - (c_x[feat_j]-x_mean[feat_j]) / \
                    x_stddev[feat_j] ) 
            return diff_to_ref_point_i == model._c_x_aux_pos[i_ref, feat_j] - \
                model._c_x_aux_neg[i_ref, feat_j]

        for i_ref in range(n_ref_points):
            for feat_j in range(model._n_feat):
                # add constraints to capture distances in variables
                # _c_x_aux_pos and _c_x_aux_neg
                model.addConstr(
                    distance_ref_point_i_for_feat_j(model, 
                        ref_points[i_ref], i_ref, feat_j, x_means, x_stddev),
                    name=f"std_const_feat_{feat_j}_{i_ref}"
                )
                # add sos constraints that allow only one of the +/- vars, 
                # i.e. _c_x_aux_pos / _c_x_aux_neg to be active
                model.addSOS(GRB.SOS_TYPE1, 
                    [
                        model._c_x_aux_pos[i_ref, feat_j], 
                        model._c_x_aux_neg[i_ref, feat_j]
                    ]
                )

            # add exploration distances as linear constraints to the model
            model.addConstr(
                model._alpha <= quicksum(
                    (model._c_x_aux_pos[i_ref, j] + model._c_x_aux_neg[i_ref, j])
                    for j in range(model._n_feat)
                ),
                name=f"alpha_sum"
            )
        model.update()

    def add_penalty_to_gurobi_model(self,
        ref_points, x_means, x_stddev, model):
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

        # big m is required to formulate the constraints
        model._big_m = \
            self.get_max_space_scaled_dist(ref_points, x_means, x_stddev, model)
        
        # two sets of variables are used to capture positive and negative 
        # parts of manhattan distance
        model._c_x_aux_pos = \
            model.addVars(range(n_ref_points), range(model._n_feat), 
                        name="c_x_aux_pos", vtype='C')

        model._c_x_aux_neg = \
            model.addVars(range(n_ref_points), range(model._n_feat), 
                        name="c_x_aux_neg", vtype='C')
        
        # binary variables b_ref correspond to active cluster centers
        model._b_ref = \
            model.addVars(n_ref_points,
                        name="b_ref", vtype=GRB.INTEGER)

        # variable alpha captures distance measure
        model._alpha = \
            model.addVar(ub=GRB.INFINITY,
                        lb=0.0,
                        name="alpha", vtype='C')

        def distance_ref_point_i_for_feat_j(
            model, xi_ref, i_ref, feat_j, x_mean, x_stddev):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._c_x

            diff_to_ref_point_i = \
                ( xi_ref[feat_j] - (c_x[feat_j]-x_mean[feat_j]) / \
                    x_stddev[feat_j] ) 
            return diff_to_ref_point_i == model._c_x_aux_pos[i_ref, feat_j] - \
                model._c_x_aux_neg[i_ref, feat_j]

        for i_ref in range(n_ref_points):
            for feat_j in range(model._n_feat):
                # add constraints to capture distances in variables
                # _c_x_aux_pos and _c_x_aux_neg
                model.addConstr(
                    distance_ref_point_i_for_feat_j(
                        model, ref_points[i_ref], i_ref, feat_j, x_means, x_stddev),
                    name=f"std_const_feat_{feat_j}_{i_ref}"
                )
                # add sos constraints that allow only one of the +/- vars, 
                # i.e. _c_x_aux_pos / _c_x_aux_neg to be active
                model.addSOS(GRB.SOS_TYPE1, 
                    [model._c_x_aux_pos[i_ref, feat_j], model._c_x_aux_neg[i_ref, feat_j]]
                )

        # add penalty distances as linear constraints to the model
            model.addConstr(
                quicksum(
                    (model._c_x_aux_pos[i_ref, j] + model._c_x_aux_neg[i_ref, j])
                    for j in range(model._n_feat)
                ) <= model._alpha + model._big_m*(1-model._b_ref[i_ref]),
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
    def __init__(self, metric='sq_euclidean'):
        # define the distance metric for continuous variables
        if metric == 'sq_euclidean':
            from entmoot.learning.distance_based_std import SquaredEuclidean
            self.cont_dist_metric = SquaredEuclidean()

        elif metric == 'manhattan':
            from entmoot.learning.distance_based_std import Manhattan
            self.cont_dist_metric = Manhattan()

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

    def update(self, Xi, yi):
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
        # update data set attributes
        self.Xi = Xi
        self.yi = yi

        # compute mean and scaler of data set
        standard_scaler = StandardScaler()
        projected_features = standard_scaler.fit_transform(self.Xi)
        self.x_means = standard_scaler.mean_
        self.x_scalers = standard_scaler.scale_

        # standardize dataset
        self.Xi_standard = self.standardize_with_Xi(self.Xi)

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
        ref_points = np.asarray(self.ref_points)

        x_standard = self.standardize_with_Xi(X)
        x_standard = np.asarray(x_standard)

        dist_cont = \
            self.cont_dist_metric.get_distance(ref_points,x_standard)
        
        return np.min(dist_cont)

    def predict(self, X):
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
        metric="sq_euclidean",
        zeta=0.5):

        super().__init__(metric)
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

    def update(self, Xi, yi):
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
        super().update(Xi, yi)

        self.ref_points_unscaled = self.Xi

        self.ref_points = self.Xi_standard

        # compute upper bound of uncertainty
        y_var = np.var(yi)
        self.distance_bound = abs(self.zeta*y_var)

    def predict(self, X):
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
        dist = super().predict(X)

        # prediction has max out at `distance_bound`
        dist[dist > self.distance_bound] = self.distance_bound
        return dist

    def get_gurobi_obj(self, model):
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
        return -model._alpha

    def add_to_gurobi_model(self,model):
        """Add standard estimator to gurobi model. Model details are 
        implemented in child class.

        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.
        """
        self.cont_dist_metric.add_exploration_to_gurobi_model(
            self.ref_points, 
            self.x_means, 
            self.x_scalers, 
            self.distance_bound, 
            model
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
        metric="sq_euclidean"):

        super().__init__(metric)

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

    def update(self, Xi, yi):
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
        super().update(Xi, yi)

        self.ref_points_unscaled = self.Xi

        self.ref_points = self.Xi_standard  


    def predict(self, X):
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
        dist = super().predict(X)
        return -dist

    def get_gurobi_obj(self, model):
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
        return model._alpha

    def add_to_gurobi_model(self,model):
        """Add standard estimator to gurobi model. Model details are 
        implemented in child class.

        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.
        """
        self.cont_dist_metric.add_penalty_to_gurobi_model(
            self.ref_points, 
            self.x_means, 
            self.x_scalers, 
            model
        )