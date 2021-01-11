from gurobipy import GRB, quicksum
import gurobipy as gp

def get_core_gurobi_model(space, add_model_core=None):
    """Add core to gurobi model, i.e. bounds, variables and parameters.

    Parameters
    ----------
    space : scikit-optimize object
        Captures the feature space
    model : gurobipy.Model,
        Model to which the core is added.

    Returns
    -------
    -
    """
    from entmoot.utils import get_cat_idx
    from entmoot.space.space import Integer

    if add_model_core is None:
        model = gp.Model()
        cat_idx = get_cat_idx(space)

        x_lb = {}
        x_ub = {}

        model._cont_var_dict = {}
        model._cat_var_dict = {}

        for idx,dim in enumerate(space.dimensions):
            if idx not in cat_idx:
                # store bounds of numerical vars
                x_lb[idx] = dim.transformed_bounds[0]
                x_ub[idx] = dim.transformed_bounds[1]

                if isinstance(dim, Integer) and dim.prior in ["uniform"]:
                    model._cont_var_dict[idx] = \
                        model.addVar(lb=x_lb[idx], ub=x_ub[idx], 
                            name=f"x_{idx}", vtype='I')
                else:
                    model._cont_var_dict[idx] = \
                        model.addVar(lb=x_lb[idx], ub=x_ub[idx], 
                            name=f"x_{idx}", vtype='C')
            else:
                # store categories of categorical vars
                model._cat_var_dict[idx] = {}
                for cat in dim.transform(dim.bounds):
                    model._cat_var_dict[idx][cat] = \
                        model.addVar(name=f"x_{idx}_{cat}", vtype=GRB.BINARY)
            
        model._n_feat = len(model._cont_var_dict) + len(model._cat_var_dict)
        model.update()
        return model
    else:
        # validate model core input
        assert type(add_model_core) is gp.Model, \
            "wrong model core type given, please create model core using: " + \
                "entmoot.optimizer.gurobi_utils.add_core_to_gurobi_model(..."

        check_attributes = \
            hasattr(add_model_core,"_cont_var_dict") and \
            hasattr(add_model_core,"_cat_var_dict") and \
            hasattr(add_model_core,"_n_feat")

        assert check_attributes, \
            "model core was not configured correctly, please create model core using: " + \
                "entmoot.optimizer.gurobi_utils.add_core_to_gurobi_model(..."

        return add_model_core

def add_std_to_gurobi_model(est, model):
    """Adds standard estimator formulation to gurobi model.

    Parameters
    ----------
    est : EntingRegressor object
        Has both tree model and uncertainty estimate embedded
    model : gurobipy.Model,
        Model to which the core is added.

    Returns
    -------
    -
    """
    est.std_estimator.add_to_gurobi_model(model)
    model.update()

def set_gurobi_init_to_ref(est, model):
    """Sets intial values of gurobi model variables to the reference points.

    Parameters
    ----------
    est : EntingRegressor object
        Has both tree model and uncertainty estimate embedded
    model : gurobipy.Model,
        Model to which the core is added.

    Returns
    -------
    -
    """
    ref_points_unscaled = \
        est.std_estimator.Xi

    best_val = est.predict(ref_points_unscaled[0].reshape(1, -1))
    best_ref = ref_points_unscaled[0]

    for ref_point in ref_points_unscaled:
        temp_val = est.predict(ref_point.reshape(1, -1))

        if best_val > temp_val:
            best_val = temp_val
            best_ref = ref_point

    for i in model._cont_var_dict.keys():
        model._cont_var_dict[i].start = best_ref[i]

    for i in model._cat_var_dict.keys():
        model._cat_var_dict[i][int(best_ref[i])].start = 1

    model._alpha.start = 0.0
    model.update()

def add_acq_to_gurobi_model(model, est, acq_func="LCB", acq_func_kwargs=None):
    """Sets gurobi model objective function to acquisition function.

    Parameters
    ----------
    model : gurobipy.Model
        Model to which the core is added.
    est : EntingRegressor object
        Has both tree model and uncertainty estimate embedded
    acq_func : str
        Type of acquisition function used for gurobi model objective
    acq_func_kwargs : dict
        Allows additional parameter settings for acquisition function

    Returns
    -------
    -
    """
    # check inputs
    if acq_func_kwargs is None:
        acq_func_kwargs = dict()

    # read kappa parameter
    kappa = acq_func_kwargs.get("kappa", 1.96)

    # collect objective contribution for tree model and std estimator
    mu, std = get_gurobi_obj(
        model, est, return_std=True, acq_func_kwargs=acq_func_kwargs
    )
    ob_expr = quicksum((mu, kappa*std))

    model.setObjective(ob_expr,GRB.MINIMIZE)


def get_gurobi_obj(model, est, return_std=False, acq_func_kwargs=None):
    """Returns gurobi model objective contribution of tree model and std
    estimator.

    Returns tree model objective contribution if `return_std` is set to False.
    If `return_std` is set to True, then return tree model and std estimator 
    contribution to gurobi model objective.

    Parameters
    ----------
    model : gurobipy.Model
        Model to which the core is added.
    est : EntingRegressor object
        Has both tree model and uncertainty estimate embedded
    return_std : bool
        Set `True` to return both tree model and std estimator contribution

    Returns
    -------
    mean or (mean, std): gurobipy.expr, or tuple(gurobipy.expr, gurobipy.expr), 
    depending on value of `return_std`.
    
    """
    scaled = acq_func_kwargs.get("scaled", False)

    mean = get_gbm_obj(model)

    if return_std:
        std = est.std_estimator.get_gurobi_obj(model, scaled=scaled)
        return mean, std

    return mean

def get_gbm_obj(model):
    """Returns objective of `gbm_model` specified in gurobi model.

    Parameters
    ----------
    model : gurobipy.Model
        Model to which the core is added.

    Returns
    -------
    ob_expr: gurobipy.expr
        Defines the gurobipy expression corresponding to the tree model objective
        contribution.
    
    """
    weighted_sum = quicksum(
        model._leaf_weight(label, tree, leaf) * \
            model._z_l[label, tree, leaf]
        for label, tree, leaf in leaf_index(model)
    )
    ob_expr = weighted_sum
    return ob_expr

### GBT HANDLER
## gbt model helper functions
def tree_index(model):
    for label in model._gbm_set:
        for tree in range(model._num_trees(label)):
            yield (label, tree)

tree_index.dimen = 2

def leaf_index(model):
    for label, tree in tree_index(model):
        for leaf in model._leaves(label, tree):
            yield (label, tree, leaf)

leaf_index.dimen = 3

def misic_interval_index(model):
    for var in model._breakpoint_index:
        for j in range(len(model._breakpoints(var))):
            yield (var, j)

misic_interval_index.dimen = 2

def misic_split_index(model):
    gbm_models = model._gbm_models
    for label, tree in tree_index(model):
        for encoding in gbm_models[label].get_branch_encodings(tree):
            yield (label, tree, encoding)

misic_split_index.dimen = 3

def alt_interval_index(model):
    for var in model.breakpoint_index:
        for j in range(1, len(model.breakpoints[var])+1):
            yield (var, j)

alt_interval_index.dimen = 2

def add_gbm_to_gurobi_model(space, gbm_model_dict, model):
    from entmoot.utils import get_cat_idx
    cat_idx = get_cat_idx(space)

    add_gbm_parameters(cat_idx, gbm_model_dict, model)
    add_gbm_variables(model)
    add_gbm_constraints(cat_idx, model)

def add_gbm_parameters(cat_idx, gbm_model_dict, model):
    model._gbm_models = gbm_model_dict

    model._gbm_set = set(gbm_model_dict.keys())
    model._num_trees = lambda label: \
        gbm_model_dict[label].n_trees
    
    model._leaves = lambda label, tree: \
            tuple(gbm_model_dict[label].get_leaf_encodings(tree))

    model._leaf_weight = lambda label, tree, leaf: \
            gbm_model_dict[label].get_leaf_weight(tree, leaf)

    vbs = [v.get_var_break_points() for v in gbm_model_dict.values()]

    all_breakpoints = {}
    for i in range(model._n_feat):
        if i in cat_idx:
            continue
        else:
            s = set()
            for vb in vbs:
                try:
                    s = s.union(set(vb[i]))
                except KeyError:
                    pass
            if s:
                all_breakpoints[i] = sorted(s)

    model._breakpoint_index = list(all_breakpoints.keys())

    model._breakpoints = lambda i: all_breakpoints[i]

    model._leaf_vars = lambda label, tree, leaf: \
            tuple(i
                for i in gbm_model_dict[label].get_participating_variables(
                    tree, leaf))

def add_gbm_variables(model):
    model._z_l = model.addVars(
                    leaf_index(model), 
                    lb=0,
                    ub=GRB.INFINITY, 
                    name="z_l", vtype='C'
                )
                            
    model._y = model.addVars(
                    misic_interval_index(model), 
                    name="y", 
                    vtype=GRB.BINARY
                )
    model.update()

def add_gbm_constraints(cat_idx, model):

    def single_leaf_rule(model_, label, tree):
        z_l, leaves = model_._z_l, model_._leaves
        return (quicksum(z_l[label, tree, leaf] 
                for leaf in leaves(label, tree))
                == 1)

    model.addConstrs(
        (single_leaf_rule(model, label, tree)
        for (label,tree) in tree_index(model)),
        name="single_leaf"
    )

    def left_split_r(model_, label, tree, split_enc):
        gbt = model_._gbm_models[label]
        split_var, split_val = gbt.get_branch_partition_pair(
            tree,
            split_enc
        )
        y_var = split_var

        if not isinstance(split_val, list):
            # for conti vars
            y_val = model_._breakpoints(y_var).index(split_val)
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_left_leaves(tree, split_enc)
                ) <= \
                    model_._y[y_var, y_val]
        else:
            # for cat vars
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_left_leaves(tree, split_enc)
                ) <= \
                    quicksum(
                        model_._cat_var_dict[split_var][cat]
                        for cat in split_val
                    )

    def right_split_r(model_, label, tree, split_enc):
        gbt = model_._gbm_models[label]
        split_var, split_val = gbt.get_branch_partition_pair(
            tree,
            split_enc
        )
        y_var = split_var
        if not isinstance(split_val, list):
            # for conti vars
            y_val = model_._breakpoints(y_var).index(split_val)
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_right_leaves(tree, split_enc)
            ) <= \
                1 - model_._y[y_var, y_val]
        else:
            # for cat vars
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_right_leaves(tree, split_enc)
            ) <= 1 - \
                quicksum(
                    model_._cat_var_dict[split_var][cat]
                    for cat in split_val
                )

    def y_order_r(model_, i, j):
        if j == len(model_._breakpoints(i)):
            return Constraint.Skip
        return model_._y[i, j] <= model_._y[i, j+1]

    def cat_sums(model_, i):
        return quicksum(
            model_._cat_var_dict[i][cat]
            for cat in model_._cat_var_dict[i].keys()
        ) == 1

    def var_lower_r(model_, i, j):
        lb = model_._cont_var_dict[i].lb
        j_bound = model_._breakpoints(i)[j]
        return model_._cont_var_dict[i] >= lb + (j_bound - lb)*(1-model_._y[i, j])

    def var_upper_r(model_, i, j):
        ub = model_._cont_var_dict[i].ub
        j_bound = model_._breakpoints(i)[j]
        return model_._cont_var_dict[i] <= ub + (j_bound - ub)*(model_._y[i, j])
    
    model.addConstrs(
        (left_split_r(model, label, tree, encoding)
        for (label, tree, encoding) in misic_split_index(model)),
        name="left_split"
    )

    model.addConstrs(
        (right_split_r(model, label, tree, encoding)
        for (label, tree, encoding) in misic_split_index(model)),
        name="right_split"
    )

    # for conti vars
    model.addConstrs(
        (y_order_r(model, var, j)
        for (var, j) in misic_interval_index(model)
        if j != len(model._breakpoints(var))-1),
        name="y_order"
    )

    # for cat vars
    model.addConstrs(
        (cat_sums(model, var)
        for var in cat_idx),
        name="cat_sums"
    )    


    model.addConstrs(
        (var_lower_r(model, var, j)
        for (var, j) in misic_interval_index(model)),
        name="var_lower"
    )

    model.addConstrs(
        (var_upper_r(model, var, j)
        for (var, j) in misic_interval_index(model)),
        name="var_upper"
    )