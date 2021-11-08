import collections as coll
import sys

class GbmModel:
    """Define a gbm model.

    A `GbmModel` is a object-oriented class to enocode tree-based models from
    different libraries. Supported tree training libraries are: LightGBM.

    Use this class as an interface to read in model data from tree model
    training libraries and translate it to the ENTMOOT native structure.

    Parameters
    ----------
    tree_list : list
        Ordered tree list that obeys ENTMOOT input specifications

    Attributes
    ----------
    -

    """
    def __init__(self, tree_list):
        self.load_ordered_tree_dict(tree_list)

    def _build_tree(self, tree):
        """Define `GbmNode` based on single tree.

        Parameters
        ----------
        trees : list of dicts,
            Specifies list of node dicts
        n_trees : number of trees in `tree_list`

        Returns
        -------
        gbm_node : GbmNode
        """
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        return GbmNode(
            split_var=split_var,
            split_code_pred=split_code_pred,
            tree=tree
        )

    def load_ordered_tree_dict(self, tree_list):
        """Define attributes used in `GbmModel`.

        Parameters
        ----------
        tree_list : list of trees,
            Specifies list of trees which contain list of node dicts

        Returns
        -------
        -
        """
        self.n_trees = len(tree_list)
        self.trees = [self._build_tree(tree) for tree in tree_list]

    def get_leaf_encodings(self, tree):
        yield from self.trees[tree].get_leaf_encodings()

    def get_branch_encodings(self, tree):
        yield from self.trees[tree].get_branch_encodings()

    def get_leaf_weight(self, tree, encoding):
        return self.trees[tree].get_leaf_weight(encoding)

    def get_leaf_weights(self, tree):
        return self.trees[tree].get_leaf_weights()

    def get_branch_partition_pair(self, tree, encoding):
        return self.trees[tree].get_branch_partition_pair(encoding)

    def get_left_leaves(self, tree, encoding):
        yield from (
            encoding + s
            for s in self.trees[tree].get_left_leaves(encoding)
        )

    def get_right_leaves(self, tree, encoding):
        yield from (
            encoding + s
            for s in self.trees[tree].get_right_leaves(encoding)
        )

    def get_branch_partition_pairs(self, tree, leaf_encoding):
        yield from self.trees[tree].get_branch_partition_pairs(leaf_encoding)

    def get_participating_variables(self, tree, leaf):
        return set(self.trees[tree].get_participating_variables(leaf))

    def get_all_participating_variables(self, tree):
        return set(self.trees[tree].get_all_participating_variables())

    def get_var_lower(self, tree, leaf, var, lower):
        return self.trees[tree].get_var_lower(leaf, var, lower)

    def get_var_upper(self, tree, leaf, var, upper):
        return self.trees[tree].get_var_upper(leaf, var, upper)

    def get_var_interval(self, tree, leaf, var, var_interval):
        return self.trees[tree].get_var_interval(leaf, var, var_interval)

    def get_var_break_points(self):
        var_breakpoints = {}
        for tree in self.trees:
            for var, breakpoint in tree.get_all_partition_pairs():
                try:
                    if isinstance(breakpoint,list):
                        var_breakpoints[var].append(breakpoint)
                    else:
                        var_breakpoints[var].add(breakpoint)
                except KeyError:
                    if isinstance(breakpoint,list):
                        var_breakpoints[var] = [breakpoint]
                    else:    
                        var_breakpoints[var] = set([breakpoint])

        for k in var_breakpoints.keys():
            if isinstance(var_breakpoints[k], set):
                var_breakpoints[k] = sorted(var_breakpoints[k])

        return var_breakpoints

    def get_leaf_count(self):
        resulting_counter = sum(
            (tree.get_leaf_count() for tree in self.trees),
            coll.Counter()
        )
        del resulting_counter['leaf']
        return resulting_counter

    def get_active_leaves(self, X):
        all_active_leaves = []
        for tree in self.trees:
            active_leaf = []
            tree._populate_active_leaf_encodings(active_leaf, X)
            all_active_leaves.append(''.join(active_leaf))
        return all_active_leaves

    def get_active_area(self, X, cat_idx=[], space=None, volume=False):
            active_splits = {}
            for tree in self.trees:
                tree._populate_active_splits(active_splits, X)

            all_active_splits = {}
            for idx,dim in enumerate(space.dimensions):
                if idx not in cat_idx:

                    if idx in active_splits.keys():
                        # if tree splits on this conti var
                        all_active_splits[idx] = active_splits[idx]
                        all_active_splits[idx].insert(0,dim.transformed_bounds[0])
                        all_active_splits[idx].append(dim.transformed_bounds[1])
                    else:
                        # add actual bounds of var if tree doesn't split on var
                        all_active_splits[idx] = \
                            [
                                dim.transformed_bounds[0],
                                dim.transformed_bounds[1]
                            ]
            # sort all splits and extract modified bounds for vars
            for key in all_active_splits.keys():
                all_active_splits[key] = \
                    sorted( list(set(all_active_splits[key])) )[:2]

            # return hypervolume if required
            if volume:
                hyper_vol = 1
                for key in all_active_splits.keys():
                    hyper_vol *= \
                        abs(all_active_splits[key][0] - all_active_splits[key][1])
                return all_active_splits, hyper_vol
            else:
                return all_active_splits


class GbmType:
    def _populate_active_splits(self, active_splits, X):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                self.left._populate_active_splits(active_splits, X)
            else:
                self.right._populate_active_splits(active_splits, X)
        else:
            if self.split_var != -1:                
                if not self.split_var in active_splits.keys():
                    active_splits[self.split_var] = []

                if X[self.split_var] <= self.split_code_pred:
                    self.left._populate_active_splits(active_splits, X)
                    active_splits[self.split_var].append(self.split_code_pred)
                else:
                    self.right._populate_active_splits(active_splits, X)

    def _populate_active_leaf_encodings(self, active_leaf, X):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                active_leaf.append('0')
                self.left._populate_active_leaf_encodings(active_leaf, X)
            else:
                active_leaf.append('1')
                self.right._populate_active_leaf_encodings(active_leaf, X)
        else:
            if self.split_var != -1:                
                if X[self.split_var] <= self.split_code_pred:
                    active_leaf.append('0')
                    self.left._populate_active_leaf_encodings(active_leaf, X)
                else:
                    active_leaf.append('1')
                    self.right._populate_active_leaf_encodings(active_leaf, X)


class GbmNode(GbmType):
    """Defines a gbm node which can be a split or leaf.

    Initializing `GbmNode` triggers recursive initialization of all nodes that 
    appear in a tree

    Parameters
    ----------
    split_var : int
        Index of split variable used
    split_code_pred : list
        Value that defines split
    tree : list
        List of node dicts that define the tree

    Attributes
    ----------
    split_var : int
        Index of split variable used
    split_code_pred : list
        Value that defines split
    tree : list
        List of node dicts that define the tree
    """
    def __init__(
        self,
        split_var,
        split_code_pred,
        tree
    ):
        self.split_var = split_var
        self.split_code_pred = split_code_pred

        # check if tree is empty
        if not tree:
            print("EmptyTreeModelError: LightGBM was not able to train a "
            "tree model based on your parameter specifications. This can "
            "usually be fixed by increasing the number of `n_initial_points` or "
            "reducing `min_child_samples` via `base_estimator_kwargs` (default "
            "is 2). Alternatively, change `acq_optimizer='sampling'`.")
            import sys
            sys.exit(1)

        # read left node 
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        # split_var value of -1 refers to leaf node
        if split_var == -1:
            self.left = LeafNode(
                split_code_pred = split_code_pred
            )
        else:
            self.left = GbmNode(
                split_var = split_var,
                split_code_pred = split_code_pred,
                tree = tree
            )

        # read right node
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        if split_var == -1:
            self.right = LeafNode(
                split_code_pred=split_code_pred
            )
        else:
            self.right = GbmNode(
                split_var=split_var,
                split_code_pred=split_code_pred,
                tree=tree
            )

    def __repr__(self):
        return ', '.join([
            str(x)
            for x in [
                self.split_var,
                self.split_code_pred]
        ])

    def _get_next_node(self, direction):
        return self.right if int(direction) else self.left

    def get_leaf_encodings(self, current_string=''):
        yield from self.left.get_leaf_encodings(current_string+'0')
        yield from self.right.get_leaf_encodings(current_string+'1')

    def get_branch_encodings(self, current_string=''):
        yield current_string
        yield from self.left.get_branch_encodings(current_string+'0')
        yield from self.right.get_branch_encodings(current_string+'1')

    def get_leaf_weight(self, encoding):
        next_node = self.right if int(encoding[0]) else self.left
        return next_node.get_leaf_weight(encoding[1:])

    def get_leaf_weights(self):
        yield from self.left.get_leaf_weights()
        yield from self.right.get_leaf_weights()

    def get_branch_partition_pair(self, encoding):
        if not encoding:
            return self.split_var, self.split_code_pred
        else:
            next_node = self._get_next_node(encoding[0])
            return next_node.get_branch_partition_pair(encoding[1:])

    def get_left_leaves(self, encoding):
        if encoding:
            next_node = self._get_next_node(encoding[0])
            yield from next_node.get_left_leaves(encoding[1:])
        else:
            yield from self.left.get_leaf_encodings('0')

    def get_right_leaves(self, encoding):
        if encoding:
            next_node = self._get_next_node(encoding[0])
            yield from next_node.get_right_leaves(encoding[1:])
        else:
            yield from self.right.get_leaf_encodings('1')

    def get_branch_partition_pairs(self, encoding):
        yield (self.split_var, self.split_code_pred)
        try:
            next_node = self.right if int(encoding[0]) else self.left
            next_gen = next_node.get_branch_partition_pairs(encoding[1:])
        except IndexError:
            next_gen = []
        yield from next_gen

    def get_participating_variables(self, encoding):
        yield self.split_var
        next_node = self.right if int(encoding[0]) else self.left
        yield from next_node.get_participating_variables(encoding[1:])

    def get_all_participating_variables(self):
        yield self.split_var
        yield from self.left.get_all_participating_variables()
        yield from self.right.get_all_participating_variables()

    def get_all_partition_pairs(self):
        yield (self.split_var, self.split_code_pred)
        yield from self.left.get_all_partition_pairs()
        yield from self.right.get_all_partition_pairs()

    def get_var_lower(self, encoding, var, lower):
        if encoding:
            if self.split_var == var:
                if int(encoding[0]) == 1:
                    assert lower <= self.split_code_pred
                    lower = self.split_code_pred
            next_node = self.right if int(encoding[0]) else self.left
            return next_node.get_var_lower(encoding[1:], var, lower)
        else:
            return lower

    def get_var_upper(self, encoding, var, upper):
        if encoding:
            if self.split_var == var:
                if int(encoding[0]) == 0:
                    assert upper >= self.split_code_pred
                    upper = self.split_code_pred
            next_node = self.right if int(encoding[0]) else self.left
            return next_node.get_var_upper(encoding[1:], var, upper)
        else:
            return upper

    def get_var_interval(self, encoding, var, var_interval):
        if int(encoding[0]):
            next_node = self.right
            if self.split_var == var:
                var_interval = (self.split_code_pred, var_interval[1])
        else:
            next_node = self.left
            if self.split_var == var:
                var_interval = (var_interval[0], self.split_code_pred)
        return next_node.get_var_interval(encoding[1:], var, var_interval)

    def get_leaf_count(self):
        left_count = self.left.get_leaf_count()
        right_count = self.right.get_leaf_count()
        joint_count = left_count+right_count

        left_key = (self.split_var, self.split_code_pred, 0)
        right_key = (self.split_var, self.split_code_pred, 1)
        joint_count[left_key] += left_count['leaf']
        joint_count[right_key] += right_count['leaf']
        return joint_count

class LeafNode(GbmType):
    """Defines a child class of `GbmType`. Leaf nodes have `split_var = -1` and 
    `split_code_pred` as leaf value defined by training."""
    def __init__(self,
                 split_code_pred):
        self.split_var = -1
        self.split_code_pred = split_code_pred

    def __repr__(self):
        return ', '.join([
            str(x)
            for x in [
                'Leaf',
                self.split_code_pred]
        ])

    def switch_to_maximisation(self):
        """Changes the sign of tree model prediction by changing signs of 
        leaf values."""
        self.split_code_pred = -1*self.split_code_pred


    def get_leaf_encodings(self, current_string=''):
        yield current_string

    def get_branch_encodings(self, current_string=''):
        yield from []

    def get_leaf_weight(self, encoding):
        assert not encoding
        return self.split_code_pred

    def get_leaf_weights(self):
        yield self.split_code_pred

    def get_branch_partition_pair(self, encoding):
        raise Exception('Should not get here.')

    def get_left_leaves(self, encoding):
        raise Exception('Should not get here.')

    def get_right_leaves(self, encoding):
        raise Exception('Should not get here.')

    def get_branch_partition_pairs(self, encoding):
        assert not encoding
        yield from []

    def get_participating_variables(self, encoding):
        assert not encoding
        yield from []

    def get_all_participating_variables(self):
        yield from []

    def get_all_partition_pairs(self):
        yield from []

    def get_var_lower(self, encoding, var, lower):
        assert not encoding
        return lower

    def get_var_upper(self, encoding, var, upper):
        assert not encoding
        return upper

    def get_var_interval(self, encoding, var, var_interval):
        assert not encoding
        return var_interval

    def get_leaf_count(self):
        return coll.Counter({'leaf': 1})