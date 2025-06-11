import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-size", type=int, default=128, help="Minimum message size (in floats)")
    parser.add_argument("--max-size", type=int, default=128*128*64, help="Maximum message size (in floats)")
    parser.add_argument("--step", type=int, default=128*64, help="Step size (in floats)")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations per size")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--collective", type=str, default="all_to_all", help="Type of collective to benchmark")
    parser.add_argument("--nccl-comms", type=str, default=None, help="file path containing the nccl configs to apply")
    parser.add_argument("--profile-last", action='store_true')

    parser.add_argument("--tp", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--vpp", type=int)
    parser.add_argument("--cp", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--etp", type=int)

    return parser