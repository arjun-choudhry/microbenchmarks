import ast
import collections
import os
import torch
import torch.distributed as dist
import tempfile
import yaml

from utils.common import get_combined_results, table_results, print_rank_0, GROUP_FORMATION_CONFIGS_HEADERS, AVG_TIME
from utils.parser import get_tuning_parser
from utils.permute import get_all_combinations
from parallel import groups
from collectives.all_to_all import AllToAll
from pathlib import Path


def get_instance(collective):
    if collective == "all_to_all":
        return AllToAll()


def executable(config, args, class_instance, results=None):
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

    results_dict = class_instance.tune(args.payload_size)
    if results is not None:
        for key, key_results in results_dict.items():
            results[key].append([
                group_formation_configs['parallel_configs'],
                group_formation_configs['nccl_group_configs'] if 'nccl_group_configs' in group_formation_configs else 'Default nccl config',
                *key_results['results'],
            ])

    groups.destroy_model_parallel()
    torch.cuda.synchronize()


# TODO: Make it generic. This is not scalable
def export_profile(args, best_result, header):
    config_dict = {}
    config_dict['group_formation_configs'] = {}
    config_dict['group_formation_configs']["parallel_configs"] = ast.literal_eval(best_result[header.index('Pipeline Configuration')])
    config_dict['group_formation_configs']["nccl_group_configs"] = ast.literal_eval(best_result[header.index('NCCL Configs')])

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_flops=False,
        with_modules=True,
        with_stack=True,
    )
    trace_file_path = f'{os.environ["TRACE_FILE_PREFIX"]}.json'
    with profiler as prof:
        executable(config_dict, args, get_instance(args.collective))
    if dist.get_rank() == 0:
        print(f"Writing trace file to {trace_file_path}")
        prof.export_chrome_trace(trace_file_path)
    torch.cuda.synchronize()

def accumulate_results(results, error_results, header):
    if torch.distributed.get_rank() == 0:
        time_idx = header.index(AVG_TIME)
        for key, result_list in results.items():
            result_list.sort(key=lambda x: float(x[time_idx].split(" ")[0]))
            table_results(header, result_list, key)

        if len(results) > 1:
            result_list = get_combined_results(results, header)
            table_results(header, result_list, "Combined Results")

        if error_results:
            headers = [*GROUP_FORMATION_CONFIGS_HEADERS, "Error Msg"]
            table_results(headers, error_results, "Error Prone Configs")

def get_profile(args, results, header):
    if args.profile_best:
        result_list = get_combined_results(results, header)
        if torch.distributed.get_rank() == 0:
            broadcast_list = [result_list[0]]
        else:
            broadcast_list = [[]]
        torch.distributed.broadcast_object_list(broadcast_list, 0)
        export_profile(args, broadcast_list[0], header)

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

    class_instance = get_instance(args.collective)

    results_headers = class_instance.get_tuning_result_headers()

    for idx, config in enumerate(all_combinations):
        try:
            executable(config, args, class_instance, results)
        except Exception as error_msg:
            error_results.append([
                config['group_formation_configs']['parallel_configs'],
                config['group_formation_configs']['nccl_group_configs'] if 'nccl_group_configs' in config['group_formation_configs'] else 'Default nccl config',
                error_msg
            ])
            invalid_configs += 1

        print_rank_0(f"Iterated {idx+1}/{total_configs_to_iterate} configs. Found {invalid_configs} invalid configs")

    header = [*GROUP_FORMATION_CONFIGS_HEADERS, *results_headers]
    accumulate_results(results, error_results, header)
    get_profile(args, results, header)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
