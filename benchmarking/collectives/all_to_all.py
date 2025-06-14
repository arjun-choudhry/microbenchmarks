import torch
import torch.distributed as dist
import time
import os

from utils.common import table_results, AVG_TIME
from parallel import groups


class AllToAll:
    def form_grps(self):
        ep_grp = groups.get_expert_model_parallel_group(check_initialized=True)
        ep_grp_size = dist.get_world_size(group=ep_grp)

        etp_grp = groups.get_expert_tensor_parallel_group(check_initialized=True)
        etp_grp_size = dist.get_world_size(group=etp_grp)


        results_grp = {
            'ep': {
                'grp': ep_grp,
                'grp_size': ep_grp_size,
                'results': []
            },
            'etp': {
                'grp': etp_grp,
                'grp_size': etp_grp_size,
                'results': []
            },
        }

        return results_grp

    def perform_collective(self, output_split, input_split, pg, key="", profiler=None):
        if profiler == None:
            dist.all_to_all(output_split, input_split, group=pg)
        else:
            trace_file_path = f'{os.environ["TRACE_FILE_PREFIX"]}_{key}.json'
            with profiler() as prof:
                dist.all_to_all(output_split, input_split, group=pg)

            if dist.get_rank() == 0:
                print(f"Writing trace file to {trace_file_path}")
                prof.export_chrome_trace(trace_file_path)

            torch.cuda.synchronize()

    def benchmark(self, min_size, max_size, step, num_iters, profiler):
        rank = dist.get_rank()

        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)

        sizes = list(range(min_size, max_size + 1, step))

        results_grp = self.form_grps()

        for key, parallel_grp in results_grp.items():
            for idx, size in enumerate(sizes):
                input_tensor = torch.randn(size, device=device)
                output_tensor = torch.empty_like(input_tensor)

                input_split = list(input_tensor.chunk(parallel_grp['grp_size']))
                output_split = list(output_tensor.chunk(parallel_grp['grp_size']))

                for _ in range(5):
                    self.perform_collective(output_split, input_split, parallel_grp['grp'])

                torch.cuda.synchronize()
                start = time.time()

                for _ in range(num_iters):
                    self.perform_collective(output_split, input_split, parallel_grp['grp'])

                torch.cuda.synchronize()
                end = time.time()

                if idx == len(sizes)-1 and profiler != None:
                    self.perform_collective(output_split, input_split, parallel_grp['grp'], key, profiler)

                avg_time = torch.tensor([(end - start) / num_iters * 1000], device=device)  # ms

                # gather_list = [torch.ones(1, device=device) for _ in range(dist.get_world_size())] if rank==0 else None

                # dist.gather(avg_time, gather_list)
                dist.reduce(avg_time, dst=0, op=dist.ReduceOp.MAX)
                torch.cuda.synchronize()

                if rank == 0:
                    parallel_grp['results'].append([
                        size,
                        f"{size * 4 / 1024:.2f} KB",
                        f"{avg_time.item():.3f} ms",
                        # f"{torch.cat(gather_list).max().item()}"
                    ])

        if rank == 0:
            headers = ["Size (floats)", "Size (KB)", "Avg Time / call", "Avg Time on all ranks"]
            table_results(headers, results_grp['ep']['results'], "EP")
            table_results(headers, results_grp['etp']['results'], "ETP")


    def tune(self, payload_size, num_iters=5):
        rank = dist.get_rank()

        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)

        results_grp = self.form_grps()

        for key, parallel_grp in results_grp.items():
            input_tensor = torch.randn(payload_size, device=device)
            output_tensor = torch.empty_like(input_tensor)

            input_split = list(input_tensor.chunk(parallel_grp['grp_size']))
            output_split = list(output_tensor.chunk(parallel_grp['grp_size']))

            for _ in range(5):
                self.perform_collective(output_split, input_split, parallel_grp['grp'])

            torch.cuda.synchronize()
            start = time.time()

            for _ in range(num_iters):
                self.perform_collective(output_split, input_split, parallel_grp['grp'])

            torch.cuda.synchronize()
            end = time.time()

            avg_time = torch.tensor([(end - start) / num_iters * 1000], device=device)  # ms

            dist.reduce(avg_time, dst=0, op=dist.ReduceOp.MAX)
            torch.cuda.synchronize()

            parallel_grp["results"] = ([
                    payload_size,
                    f"{payload_size * 4 / 1024:.2f} KB",
                    f"{avg_time.item():.3f} ms",
            ])

        return results_grp

    def get_tuning_result_headers(self):
        return ["Size (floats)", "Size (KB)", AVG_TIME]