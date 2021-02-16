from abc import ABC
import numpy as np

class MisicProximityStd(ABC):    
    """        
    Proximity-based standard estimator. Proximity between two points in a 
    space is defined as the proportion of trees in an ensemble model where the 
    two points activate the same leaf.
    This class favours exploration by limiting the proximity between the 
    solution point and the reference set; a penalty version could 
    Parameters
    ----------
    threshold : 
        Proximity threshold to 
    Attributes
    ----------
    Xi : list
        Points at which objective has been evaluated.
    yi : scalar
        Values of objective at corresponding points in `Xi`.
    ref_model : entmoot.learning.gbm_model.GbmModel
        Stored model for evaluating leaf proximity.
    ref_leaves : list
        Points in 'Xi', encoded as leaves with 'ref_model'
        
    """
    def __init__(self,threshold=0.9):
        self.threshold = threshold
        # define the distance metric for continuous variables

    def set_params(self,**kwargs):
        """Sets parameters related to distance-based standard estimator.
        Parameters
        ----------
        threshold: 
        kwargs : dict
            Additional arguments to be passed to the standard estimator
        Returns
        -------
        -
        """
        self.threshold = kwargs.get("threshold",0.9)

    def update(self, Xi, yi,cat_column=[]):
        """Update available data points which is usually done after every
        iteration.
        Parameters
        ----------
        Xi : list
            Points at which objective has been evaluated.
        yi : scalar
            Values of objective at corresponding points in `Xi`.
        cat_column : 
            Currently categorical variables are not supported, but putting 
            this in for compatibility.
            
        Returns
        -------
        -
        """
        if cat_column != []:
            raise ValueError(
                "Categorical variables are currently unsupported with the "
                "Misic Proximity standard estimator."
                )
        # update data set attributes
        self.Xi = Xi
        self.yi = yi

        self.n_features = self.Xi.shape[1]
        self.ref_points_unscaled = self.Xi

    def update_model(self,gbm_model):
        """ Add a reference GBM model, and encode leaf assignments for the 
        Xi reference points.
        Parameters
        ----------
        gbm_model : entmoot.learning.gbm_model.GbmModel
            GBM model to add to the estimator.
        """
        
        self.ref_model = gbm_model
        self.ref_leaves = [self.encode_point(gbm_model,x) for x in self.Xi]


    def encode_point(self,gbm_model,X):
        """Identifies the leaves that point 'X' will activate in 
        GBM model 'gbm_model'. 
        Parameters
        ----------
        gbm_model : entmoot.learning.gbm_model.GbmModel
            Entmoot GBM tree model.
        X : numpy array, shape (n_dims,)
            Point to be encoded.
        Returns
        -------
        X_leaves: numpy array, shape (n_trees,)
            Leaves activated in 'gbm_model' by point 'X'. Each entry is a 
            string of 0s and 1s, describing the left and right splits at 
            each branch point of the tree.
            
        """
        X_leaves = np.empty(gbm_model.n_trees,dtype='U32') # Assuming no regressors go deeper than 32 branches
        for tree in range(gbm_model.n_trees):
            # Identify which nodes are leaves.
            leaves = list(gbm_model.get_leaf_encodings(tree))
            encoding = '' # Start at the root, walk until we reach a leaf.
            while encoding not in leaves:
                # Take the split at this node, get the participating variable and
                # split value.
                var,split = gbm_model.get_branch_partition_pair(tree,encoding)
                # If the point is less than the split, go left.
                if np.round(X[var],9) < np.round(split,9):
                    encoding = encoding + '0'
                # If the point is greater than the split value, go right.
                elif np.round(X[var],9) > np.round(split,9):
                    encoding = encoding + '1'
                # If the point is on the split value, see which side has the 
                # smallest leaf under it.
                else:
                    # identify leaves below this branch point
                    possible_leaves = [
                        leaves[i] for i in range(len(leaves)) if leaves[i].startswith(encoding)]
                    # get weights for the leaves below this branch point
                    weights = list(gbm_model.get_leaf_weights(tree))
                    possible_weights = [
                        weights[i] for i in range(len(leaves)) if leaves[i].startswith(encoding)]
                    # Determine which of these is the smallest
                    best_leaf = possible_leaves[np.argmin(possible_weights)]
                    # Move one step down that branch of the tree
                    encoding = encoding + best_leaf[len(encoding)] 
            # Add this leaf to the set and move onto the next tree.
            X_leaves[tree] = encoding
        return X_leaves

    def get_closest_point_proximity(self, X):
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
        ref_leaves = np.asarray(self.ref_leaves)

        X_leaves = self.encode_point(self.ref_model, X)

        prox_list = [
            sum(1 for i in range(self.ref_model.n_trees) \
                if X_leaves[i] == ref_point_leaves[i])/self.ref_model.n_trees \
             for ref_point_leaves in ref_leaves
             ]
        return np.max(prox_list)

    def predict(self, X, scaled=True):
        """Predict standard estimate at location `X`.
        Parameters
        ----------
        X : numpy array, shape (n_rows, n_dims)
            Points at which the standard estimator is evaluated.
        scaled : not relevant here, but included for compatibility.
        Returns
        -------
        prox : numpy array, shape (n_rows,)
            Returns proximities to closest `ref_point` for every point per row
            in `n_rows`.
        """
        prox = np.empty([X.shape[0],])

        for row_res,Xi in enumerate(X):
            ref_proximity = self.get_closest_point_proximity(Xi)
            prox[row_res] = ref_proximity

        return prox

    def add_to_gurobi_model(self,model):
        """Add standard estimator constraints to gurobi model. Uses the GBM
        models already in the Gurobi model to determine proximity.
        
        Parameters
        ----------
        model : gurobipy.Model,
            Model to which the standard estimator is added.
        """
        from gurobipy import LinExpr
        
        n_points = len(self.ref_leaves)

        model._alpha = \
            model.addVar(
                lb=0, 
                ub=self.threshold, 
                name="alpha", 
                vtype='C'
            )
        
        for label in model._gbm_set:
            n_trees = model._num_trees(label)
            self.update_model(model._gbm_models[label])
                
            # For each observation in the training data, construct the proximity
            # expression in terms of the indicator variables.
            for observation in range(n_points):
                prox_name = 'prox_X'+str(observation)+' < '+str(self.threshold)
                alpha = model._alpha
                prox_obs = LinExpr()
                for j in range(n_trees):
                    # Construct the leaf variable name
                    leafvar_name = "z_l[" + label + "," + str(j) + "," + str(self.ref_leaves[observation][j])+"]"
                    leafvar = model.getVarByName(leafvar_name)
                    prox_obs.addTerms([1],[leafvar])
                
                # Add constraint to model.
                model.addLConstr(prox_obs/n_trees,'<=',alpha, name=prox_name)
                model.update()    
        return model

    def get_gurobi_obj(self,model,scaled=True):
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
        return model._alpha
