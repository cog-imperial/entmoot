import collections as coll


class MetaTreeModel:
    def __init__(self, tree_list: list):
        self.load_ordered_tree_list(tree_list)

    def _build_tree(self, tree):
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        return TreeNode(
            split_var=split_var,
            split_code_pred=split_code_pred,
            tree=tree
        )

    def load_ordered_tree_list(self, tree_list: list):
        self.trees = [self._build_tree(tree) for tree in tree_list]


class TreeType:
    pass


class TreeNode(TreeType):
    def __init__(self, split_var, split_code_pred, tree):
        self.split_var = split_var
        self.split_code_pred = split_code_pred

        # check if tree is empty
        assert tree, "Given the data and train configuration, no tree ensemble could be trained."

        # read left node
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        # split_var value of -1 refers to leaf node
        if split_var == -1:
            self.left = LeafNode(
                split_code_pred=split_code_pred
            )
        else:
            self.left = TreeNode(
                split_var=split_var,
                split_code_pred=split_code_pred,
                tree=tree
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
            self.right = TreeNode(
                split_var=split_var,
                split_code_pred=split_code_pred,
                tree=tree
            )

    def __repr__(self):
        return ', '.join(
            [str(x) for x in [self.split_var, self.split_code_pred]]
        )


class LeafNode(TreeType):
    def __init__(self,
                 split_code_pred):
        self.split_var = -1
        self.split_code_pred = split_code_pred

    def __repr__(self):
        return ', '.join(
            [str(x) for x in ['Leaf', self.split_code_pred]]
        )
