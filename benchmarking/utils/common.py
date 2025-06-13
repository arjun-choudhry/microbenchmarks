import torch
from tabulate import tabulate

GROUP_FORMATION_CONFIGS_HEADERS = ["Pipeline Configuration", "NCCL Configs"]
AVG_TIME = "Avg Time / call"

def table_results(header, results, key):
    print()
    print(f"Metrics for {key} collective")
    print(tabulate(results, headers=header, tablefmt="pipe"))
    print()

def print_rank_0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)