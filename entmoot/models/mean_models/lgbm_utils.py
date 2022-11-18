def read_lgbm_tree_model_dict(tree_model_dict, cat_idx):
    ordered_tree_list = []

    for tree in tree_model_dict["tree_info"]:

        # generate list of nodes in tree
        root_node = [tree["tree_structure"]]
        node_list = []

        # populate node_list and add to ordered_tree_list if non-empty
        add_next_nodes(node_list=node_list, node=root_node, cat_idx=cat_idx)

        if node_list:
            ordered_tree_list.append(node_list)

    return ordered_tree_list


def add_next_nodes(node_list, node, cat_idx):
    if node:
        # add new node definition
        new_node = {}

        try:
            new_node["split_var"] = node[-1]["split_feature"]
            if not new_node["split_var"] in cat_idx:
                # read numerical variables, solver accuracy 10e-5
                temp_node_val = round(node[-1]["threshold"], 5)
                new_node["split_code_pred"] = temp_node_val
            else:
                # read categorical variables
                cat_set = node[-1]["threshold"].split("||")
                temp_node_val = [int(cat) for cat in cat_set]
                new_node["split_code_pred"] = temp_node_val

        except KeyError:
            # arrived at leaf node
            new_node["split_var"] = -1
            temp_node_val = node[-1]["leaf_value"]
            new_node["split_code_pred"] = temp_node_val

        node_list.append(new_node)

        # move on to next node in tree
        temp_node = node.pop(-1)
        try:
            node.append(temp_node["right_child"])
        except KeyError:
            pass

        try:
            node.append(temp_node["left_child"])
        except KeyError:
            pass

        add_next_nodes(node_list, node, cat_idx=cat_idx)
