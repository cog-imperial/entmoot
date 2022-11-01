def order_tree_model_dict(tree_model_dict, cat_column=None):
    cat_column = {} if cat_column is None else cat_column
    tree_list = tree_model_dict['tree_info']
    ordered_tree_list = order_tree_list(tree_list, cat_column=cat_column)
    return ordered_tree_list


def order_tree_list(tree_list, cat_column=None):
    ordered_tree_list = []
    for tree in tree_list:
        ordered_node_list = order_node_list(tree, cat_column=cat_column)
        if ordered_node_list:
            ordered_tree_list.append(ordered_node_list)
    return ordered_tree_list


def order_node_list(tree, cat_column=None):
    node = []
    node.append(tree['tree_structure'])
    ordered_node_list = []
    add_next_nodes(ordered_node_list, node, cat_column=cat_column)
    return ordered_node_list


def add_next_nodes(ordered_node_list, node, cat_column=None):
    if node:
        new_node = {}

        try:
            new_node['split_var'] = node[-1]['split_feature']
            if not new_node['split_var'] in cat_column:
                # read numerical variables
                temp_node_val = round(node[-1]['threshold'], 5)
                new_node['split_code_pred'] = temp_node_val  # round(temp_node_val,5)
            else:
                # read categorical variables
                cat_set = node[-1]['threshold'].split("||")
                temp_node_val = [int(cat) for cat in cat_set]
                new_node['split_code_pred'] = temp_node_val

        except KeyError:
            new_node['split_var'] = -1
            temp_node_val = node[-1]['leaf_value']
            new_node['split_code_pred'] = temp_node_val  # round(temp_node_val,5)

        ordered_node_list.append(new_node)

        temp_node = node.pop(-1)
        try:
            node.append(temp_node['right_child'])
        except KeyError:
            pass

        try:
            node.append(temp_node['left_child'])
        except KeyError:
            pass

        add_next_nodes(ordered_node_list, node, cat_column=cat_column)


