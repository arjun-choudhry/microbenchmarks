import torch
from parallel.groups import get_tensor_model_parallel_world_size

class ParallelLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, num_partitions, bias_flag, device):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.num_partitions = num_partitions
        self.bias_flag = bias_flag
        self.device = device


class ColumnParallelLinear(ParallelLinear):
    def __init__(self, input_features, output_features, num_partitions, bias_flag, device='cuda'):
        super().__init__(input_features, output_features, num_partitions, bias_flag, device)

        assert (self.output_features % self.num_partitions) == 0, "Please ensure that num_output_features is fully divisible by num_partitions"

        self.weights = torch.nn.Parameter(
            torch.rand(size=(input_features, output_features // num_partitions), device=device),
            requires_grad=True
        )

        if bias_flag:
            self.bias = torch.nn.Parameter(
                torch.rand(size=(output_features // num_partitions,), device=device)
            )

    def forward(self, x):
        # if x.shape() ==

        results = torch.matmul(x, self.weights)
        if self.bias_flag:
            results += self.bias

        return results


    def combine_weight_matrices(self):
        weights = [torch.empty_like(self.weights, device=self.device) for _ in get_tensor_model_parallel_world_size()]
        torch.distributed.all_gather(weights, self.weights)


