import collections
import copy

import torch
from tabulate import tabulate

GROUP_FORMATION_CONFIGS_HEADERS = ["Pipeline Configuration", "NCCL Configs"]
MODEL_CONFIGS_HEADERS = ["Model Configs"]
AVG_TIME = "Avg Time / call"


def table_results(header, results, key):
    print()
    print(f"Metrics for {key} collective")
    print(tabulate(results, headers=header, tablefmt="pipe"))
    print()


def print_rank_0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def get_combined_results(results_dict, header):
    intermittent_dict = copy.deepcopy(results_dict)
    time_idx = header.index(AVG_TIME)
    combined_dict = collections.defaultdict(lambda: 0)

    for key, val_list in intermittent_dict.items():
        for values in val_list:
            time, metric = values[time_idx].split(" ")
            values.pop(time_idx)
            combined_dict[tuple([str(item) for item in values])] += float(time)

    combined_list = []
    for key, val in combined_dict.items():
        keys = list(key)
        keys.insert(time_idx, f"{val} {metric}")
        combined_list.append(keys)

    combined_list.sort(key=lambda x: x[time_idx].split(" ")[0])

    return combined_list

