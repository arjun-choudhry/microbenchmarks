import collections
import torch
import torch.distributed as dist
import tempfile
import yaml

from utils.common import table_results, print_rank_0, GROUP_FORMATION_CONFIGS_HEADERS, AVG_TIME
from utils.parser import get_tuning_parser
from utils.permute import get_all_combinations
from parallel import groups
from collectives.all_to_all import tune_all_to_all
from pathlib import Path


def main():
    parser = get_tuning_parser()
    args = parser.parse_args()

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    root_dir = Path(__file__).resolve().parent.parent
    args.tuning_configs = f"{root_dir}/{args.tuning_configs}"

    with open(args.tuning_configs, "r") as stream:
        tuning_configs = yaml.safe_load(stream)

    results = collections.defaultdict(list)
    error_results = []

    all_combinations = get_all_combinations(tuning_configs)
    total_configs_to_iterate = len(all_combinations)
    invalid_configs = 0

    results_headers = []

    for idx, config in enumerate(all_combinations):
        try:
            group_formation_configs = config['group_formation_configs']
            temp_yaml_path = None
            if 'nccl_group_configs' in group_formation_configs:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                    yaml.dump(group_formation_configs['nccl_group_configs'], tmp)
                    temp_yaml_path = tmp.name

            groups.initialize_model_parallel(
                tensor_model_parallel_size=group_formation_configs['parallel_configs']['tp'],
                pipeline_model_parallel_size=group_formation_configs['parallel_configs']['pp'],
                virtual_pipeline_model_parallel_size=group_formation_configs['parallel_configs']['vpp'],
                context_parallel_size=group_formation_configs['parallel_configs']['cp'],
                expert_model_parallel_size=group_formation_configs['parallel_configs']['ep'],
                expert_tensor_parallel_size=group_formation_configs['parallel_configs']['etp'],
                nccl_communicator_config_path=temp_yaml_path
            )

            if args.collective == "all_to_all":
                results_dict, results_headers = tune_all_to_all(args.payload_size)

            for key, key_results in results_dict.items():
                results[key].append([
                    group_formation_configs['parallel_configs'],
                    group_formation_configs['nccl_group_configs'] if 'nccl_group_configs' else 'Default nccl config',
                    *key_results['results'],
                ])

            groups.destroy_model_parallel()

        except Exception as error_msg:
            error_results.append([
                group_formation_configs['parallel_configs'],
                group_formation_configs['nccl_group_configs'] if 'nccl_group_configs' else 'Default nccl config',
                error_msg
            ])
            invalid_configs += 1

        print_rank_0(f"Iterated {idx+1}/{total_configs_to_iterate} configs. Found {invalid_configs} invalid configs")

    if torch.distributed.get_rank() == 0:
        header = [*GROUP_FORMATION_CONFIGS_HEADERS, *results_headers]
        time_idx = header.index(AVG_TIME)
        for key, result_list in results.items():
            result_list.sort(key=lambda x: float(x[time_idx].split(" ")[0]))
            table_results(header, result_list, key)

        if error_results:
            headers = [*GROUP_FORMATION_CONFIGS_HEADERS, "Error Msg"]
            table_results(headers, error_results, "Error Prone Configs")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
