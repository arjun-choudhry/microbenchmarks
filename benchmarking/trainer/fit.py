import torch
import time

def train(model, input_data, iter_steps):
    start = time.time()
    for _ in range(iter_steps):
        output = model(input_data)

        #dummy target / loss_fn
        target = torch.ones_like(output)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, target)

        loss.backward()
        for parameter in model.parameters():
            parameter.grad.zero_()

    end = time.time()
    avg_time = torch.tensor([(end - start) / iter_steps * 1000], device='cuda')  # ms

    torch.distributed.reduce(avg_time, dst=0, op=torch.distributed.ReduceOp.MAX)
    return avg_time
