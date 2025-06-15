import torch

from building_blocks.parallel_linear import ColumnParallelLinear
from trainer.fit import train


def partition_input(parallel_configs, input_data):
    if parallel_configs['tp']['type'] == 'column_parallel':
        torch.distributed.broadcast(input_data, src=0)

    return input_data


def perform_gemm(input_features, output_features, bias_flag, num_sequences, parallel_configs, device):
    input_data = torch.rand(size=(num_sequences, input_features), device=device)
    input_data = partition_input(parallel_configs, input_data)

    gemm_layer = ColumnParallelLinear(
        input_features=input_features,
        output_features=output_features,
        bias_flag=bias_flag,
        device=device,
        num_partitions=parallel_configs['tp']['size']
    )

    train(gemm_layer, input_data, 1)

    gather_weights = [torch.empty_like(gemm_layer.weights) for _ in range(parallel_configs['tp']['size'])]
    torch.distributed.all_gather(gather_weights, gemm_layer.weights)






