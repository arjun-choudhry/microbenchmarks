import torch
import torch.distributed as dist

from functools import partial
from utils.parser import get_parser
from parallel import groups
from collectives.all_to_all import benchmark_all_to_all
from pathlib import Path


def main():
    parser = get_parser()
    args = parser.parse_args()

    dist.init_process_group(
        backend=args.backend,
        init_method="env://"
    )

    if args.nccl_comms:
        root_dir = Path(__file__).resolve().parent
        args.nccl_comms = f"{root_dir}/{args.nccl_comms}"

    groups.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        virtual_pipeline_model_parallel_size=args.vpp,
        context_parallel_size=args.cp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        nccl_communicator_config_path=args.nccl_comms
    )

    if args.profile_last:
        profiler = partial(
            torch.profiler.profile,
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_flops=False,
            with_modules=True,
            with_stack=True,
        )
        print("profile is turned on")
        print("++++++++++++++++++++++++++")
    else:
        profiler=None

    if args.collective == "all_to_all":
        benchmark_all_to_all(args.min_size, args.max_size, args.step, args.iters, profiler)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
