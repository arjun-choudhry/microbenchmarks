from itertools import product

def get_all_combinations(config):
    permuted_list = []
    key_list = []
    val_list = []
    for key, vals in config.items():
        key_list.append(key)
        if isinstance(vals, dict):
            val_list.append(get_all_combinations(vals))
        if isinstance(vals, list):
            val_list.append(vals)

    val_list = list(product(*val_list))
    for item in val_list:
        merged_dict = {}
        for key, val in zip(key_list, item):
            merged_dict[key] = val

        permuted_list.append(merged_dict)

    return permuted_list