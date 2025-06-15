import ast
import os
import torch
import torch.distributed as dist
import tempfile
import yaml

from utils.common import get_combined_results, table_results, print_rank_0, GROUP_FORMATION_CONFIGS_HEADERS, MODEL_CONFIGS_HEADERS, AVG_TIME
from utils.parser import get_ops_tuning_parser
from utils.permute import get_all_combinations
from parallel import groups
from ops.gemm import perform_gemm
from scripts.ops.model_utils import create_model_from_configs
from trainer.fit import train
from pathlib import Path


def executable(tuning_configs, model_configs, results=None):
    group_formation_configs = tuning_configs['group_formation_configs']

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

    model = create_model_from_configs(model_configs)
    data = torch.rand(size=(model_configs['data']['num_sequences'], model_configs['data']['features']), device='cuda')
    avg_time = train(model, data, 1)

    if results is not None:
        results.append([
            group_formation_configs['parallel_configs'],
            group_formation_configs['nccl_group_configs'] if 'nccl_group_configs' in group_formation_configs else 'Default nccl config',
            model_configs['layers'],
            f"{avg_time.item():.3f} ms",
        ])

    groups.destroy_model_parallel()
    torch.cuda.synchronize()


# TODO: Make it generic. This is not scalable
def export_profile(best_result, model_configs, header):
    config_dict = {}
    # print('best_result', best_result)
    config_dict['group_formation_configs'] = {}
    config_dict['group_formation_configs']["parallel_configs"] = best_result[header.index('Pipeline Configuration')]
    config_dict['group_formation_configs']["nccl_group_configs"] = best_result[header.index('NCCL Configs')]

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
        executable(config_dict, model_configs)
    if dist.get_rank() == 0:
        print(f"Writing trace file to {trace_file_path}")
        prof.export_chrome_trace(trace_file_path)
    torch.cuda.synchronize()


def accumulate_results(results, error_results, header):
    if torch.distributed.get_rank() == 0:
        time_idx = header.index(AVG_TIME)
        results.sort(key=lambda x: float(x[time_idx].split(" ")[0]))
        table_results(header, results, "Overall Results")

        if error_results:
            headers = [*GROUP_FORMATION_CONFIGS_HEADERS, "Error Msg"]
            table_results(headers, error_results, "Error Prone Configs")


def get_profile(args, model_configs, results, header):
    if args.profile_best:
        time_idx = header.index(AVG_TIME)
        results.sort(key=lambda x: x[time_idx].split(" ")[0])
        if torch.distributed.get_rank() == 0:
            broadcast_list = [results[0]]
        else:
            broadcast_list = [[]]
        torch.distributed.broadcast_object_list(broadcast_list, 0)
        # print(broadcast_list)
        export_profile(broadcast_list[0], model_configs, header)

def main():
    parser = get_ops_tuning_parser()
    args = parser.parse_args()

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    rank = dist.get_rank()

    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    with open(args.tuning_configs, "r") as stream:
        tuning_configs = yaml.safe_load(stream)

    with open(args.model_configs, "r") as stream:
        model_configs = yaml.safe_load(stream)

    results = []
    error_results = []

    all_combinations = get_all_combinations(tuning_configs)
    total_configs_to_iterate = len(all_combinations)
    invalid_configs = 0

    # func_to_perform = get_layer_func("gemm")

    # parallel_configs = {
    #     'tp': {
    #         'type': "column_parallel",
    #         "size": 8
    #     }
    #
    # }

    # profiler = torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     record_shapes=True,
    #     with_flops=False,
    #     with_modules=True,
    #     with_stack=True,
    # )
    # trace_file_path = f'{os.environ["TRACE_FILE_PREFIX"]}.json'
    #
    # with profiler as prof:
    #     func_to_perform(5, 8, True, 4, parallel_configs, device='cuda')
    # if dist.get_rank() == 0:
    #     print(f"Writing trace file to {trace_file_path}")
    #     prof.export_chrome_trace(trace_file_path)
    # torch.cuda.synchronize()


    # class_instance = get_instance(args.collective)
    #
    # results_headers = class_instance.get_tuning_result_headers()

    for idx, tuning_configs in enumerate(all_combinations):
        try:
            executable(tuning_configs, model_configs, results)
        except Exception as error_msg:
            if groups.is_initialized():
                groups.destroy_model_parallel()
            error_results.append([
                tuning_configs['group_formation_configs']['parallel_configs'],
                tuning_configs['group_formation_configs']['nccl_group_configs'] if 'nccl_group_configs' in tuning_configs['group_formation_configs'] else 'Default nccl config',
                model_configs['layers'],
                error_msg
            ])
            invalid_configs += 1

        print_rank_0(f"Iterated {idx+1}/{total_configs_to_iterate} configs. Found {invalid_configs} invalid configs")

    header = [*GROUP_FORMATION_CONFIGS_HEADERS, *MODEL_CONFIGS_HEADERS, AVG_TIME]
    accumulate_results(results, error_results, header)
    get_profile(args, model_configs, results, header)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
