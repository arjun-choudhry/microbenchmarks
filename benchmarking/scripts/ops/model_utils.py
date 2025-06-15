import torch.nn

from building_blocks.parallel_linear import ColumnParallelLinear
from parallel.groups import get_tensor_model_parallel_world_size

def get_layer(layer_config):
    if layer_config['layer_type'] == "linear":
        if layer_config['parallel_type'] == "column":
            return ColumnParallelLinear(
                input_features=layer_config['args']['input_features'],
                output_features=layer_config['args']['output_features'],
                num_partitions=get_tensor_model_parallel_world_size(),
                bias_flag=layer_config['args']['bias']
            )


def create_model_from_configs(configs):
    model = []
    for item in configs['layers']:
        model.append(get_layer(item))

    return torch.nn.Sequential(*model)

