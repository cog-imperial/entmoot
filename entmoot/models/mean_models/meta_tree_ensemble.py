class MetaTreeModel:
    def __init__(self, tree_list: list):
        self._num_trees = len(tree_list)
        self.load_ordered_tree_list(tree_list)

    @property
    def num_trees(self):
        return self._num_trees

    def _build_tree(self, tree):
        node = tree.pop(0)
        split_var = node["split_var"]
        split_code_pred = node["split_code_pred"]

        return TreeNode(split_var=split_var, split_code_pred=split_code_pred, tree=tree)

    def load_ordered_tree_list(self, tree_list: list):
        self.trees = [self._build_tree(tree) for tree in tree_list]

    def get_leaf_encodings(self, tree):
        yield from self.trees[tree].get_leaf_encodings()

    def get_leaf_weight(self, tree, encoding):
        return self.trees[tree].get_leaf_weight(encoding)

    def get_var_break_points(self):
        var_breakpoints = {}
        for tree in self.trees:
            for var, breakpoint in tree.get_all_partition_pairs():
                try:
                    if isinstance(breakpoint, list):
                        var_breakpoints[var].append(breakpoint)
                    else:
                        var_breakpoints[var].add(breakpoint)
                except KeyError:
                    if isinstance(breakpoint, list):
                        var_breakpoints[var] = [breakpoint]
                    else:
                        var_breakpoints[var] = set([breakpoint])

        for k in var_breakpoints:
            if isinstance(var_breakpoints[k], set):
                var_breakpoints[k] = sorted(var_breakpoints[k])

        return var_breakpoints

    def get_participating_variables(self, tree, leaf):
        return set(self.trees[tree].get_participating_variables(leaf))

    def get_branch_encodings(self, tree):
        yield from self.trees[tree].get_branch_encodings()

    def get_branch_partition_pair(self, tree, encoding):
        return self.trees[tree].get_branch_partition_pair(encoding)

    def get_left_leaves(self, tree, encoding):
        yield from (encoding + s for s in self.trees[tree].get_left_leaves(encoding))

    def get_right_leaves(self, tree, encoding):
        yield from (encoding + s for s in self.trees[tree].get_right_leaves(encoding))


class TreeType:
    pass


class TreeNode(TreeType):
    def __init__(self, split_var, split_code_pred, tree):
        self.split_var = split_var
        self.split_code_pred = split_code_pred

        # check if tree is empty
        assert (
            tree
        ), "Given the data and train configuration, no tree ensemble could be trained."

        # read left node
        node = tree.pop(0)
        split_var = node["split_var"]
        split_code_pred = node["split_code_pred"]

        # split_var value of -1 refers to leaf node
        if split_var == -1:
            self.left = LeafNode(split_code_pred=split_code_pred)
        else:
            self.left = TreeNode(
                split_var=split_var, split_code_pred=split_code_pred, tree=tree
            )

        # read right node
        node = tree.pop(0)
        split_var = node["split_var"]
        split_code_pred = node["split_code_pred"]

        if split_var == -1:
            self.right = LeafNode(split_code_pred=split_code_pred)
        else:
            self.right = TreeNode(
                split_var=split_var, split_code_pred=split_code_pred, tree=tree
            )

    def __repr__(self):
        return ", ".join([str(x) for x in [self.split_var, self.split_code_pred]])

    def get_leaf_encodings(self, current_string=""):
        yield from self.left.get_leaf_encodings(current_string + "0")
        yield from self.right.get_leaf_encodings(current_string + "1")

    def get_leaf_weight(self, encoding):
        next_node = self.right if int(encoding[0]) else self.left
        return next_node.get_leaf_weight(encoding[1:])

    def get_all_partition_pairs(self):
        yield (self.split_var, self.split_code_pred)
        yield from self.left.get_all_partition_pairs()
        yield from self.right.get_all_partition_pairs()

    def get_participating_variables(self, encoding):
        yield self.split_var
        next_node = self.right if int(encoding[0]) else self.left
        yield from next_node.get_participating_variables(encoding[1:])

    def get_branch_encodings(self, current_string=""):
        yield current_string
        yield from self.left.get_branch_encodings(current_string + "0")
        yield from self.right.get_branch_encodings(current_string + "1")

    def _get_next_node(self, direction):
        return self.right if int(direction) else self.left

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
            yield from self.left.get_leaf_encodings("0")

    def get_right_leaves(self, encoding):
        if encoding:
            next_node = self._get_next_node(encoding[0])
            yield from next_node.get_right_leaves(encoding[1:])
        else:
            yield from self.right.get_leaf_encodings("1")


class LeafNode(TreeType):
    def __init__(self, split_code_pred):
        self.split_var = -1
        self.split_code_pred = split_code_pred

    def __repr__(self):
        return ", ".join([str(x) for x in ["Leaf", self.split_code_pred]])

    def get_leaf_encodings(self, current_string=""):
        yield current_string

    def get_leaf_weight(self, encoding):
        assert not encoding
        return self.split_code_pred

    def get_all_partition_pairs(self):
        yield from []

    def get_participating_variables(self, encoding):
        assert not encoding
        yield from []

    def get_branch_encodings(self, current_string=""):
        yield from []
