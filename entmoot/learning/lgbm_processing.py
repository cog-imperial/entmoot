
def order_tree_model_dict(tree_model_dict, cat_column=[]):
    """
    Sorts a tree model dict provided by LightGBM from .json file to make it 
    suitable for use in ENTMOOT.

    Key name `tree_info` is specific to LightGBM and needs to be changed for
    other tree training libraries that support model dump to .json format.

    Parameters
    ----------
    tree_model_dict : dict,
        Includes all information on trees trained in LightGBM

    Returns
    -------
    ordered_tree_list : list,
        Ordered tree list compatible with `GbmModel`
    """
    tree_list = tree_model_dict['tree_info']
    ordered_tree_list = order_tree_list(tree_list, cat_column=cat_column)
    return ordered_tree_list

def order_tree_list(tree_list, cat_column=[]):
    """
    Sorts a tree list provided by LightGBM from .json file to make it 
    suitable for use in ENTMOOT.

    Parameters
    ----------
    tree_list : list,
        Unsorted list of trees from LightGBM

    Returns
    -------
    ordered_tree_list : list,
        Ordered tree list compatible with `GbmModel`
    """
    ordered_tree_list = []
    for tree in tree_list:
        ordered_node_list = order_node_list(tree, cat_column=cat_column)
        if ordered_node_list:
            ordered_tree_list.append(ordered_node_list)
    return ordered_tree_list

def order_node_list(tree, cat_column=[]):
    """
    Sorts a list of node dict from a LightGBM instance. Key `tree_structure` is 
    specific for LightGBM.

    Parameters
    ----------
    tree : list,
        Unsorted list of node dicts

    Returns
    -------
    ordered_node_list : list,
        Ordered list of node dicts compatible with `GbmModel`
    """
    node = []
    node.append(tree['tree_structure'])
    ordered_node_list = []
    add_next_nodes(ordered_node_list, node, cat_column=cat_column)
    return ordered_node_list
    
def add_next_nodes(ordered_node_list, node, cat_column=[]):
    """
    Processes LightGBM node and adds sorted node to `ordered_node_list`.

    Parameters
    ----------
    ordered_node_list : list,
        List of nodes to which sorted node is added

    node : list,
        List of unsorted LightGBM nodes for one tree

    Returns
    -------
    -
    """
    if node:   
        new_node = {}
        
        try:
            new_node['split_var'] = node[-1]['split_feature']
            if not new_node['split_var'] in cat_column:
                # read numerical variables
                temp_node_val = node[-1]['threshold']
                new_node['split_code_pred'] = round(temp_node_val,5)
            else:
                # read categorical variables
                cat_set = node[-1]['threshold'].split("||")
                temp_node_val = [int(cat) for cat in cat_set]
                new_node['split_code_pred'] = temp_node_val

        except KeyError:
            new_node['split_var'] = -1
            temp_node_val = node[-1]['leaf_value']
            new_node['split_code_pred'] = round(temp_node_val,5)

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
    
    
