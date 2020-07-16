#!/usr/bin/env python3

from mpify import in_torchddp, ddp_rank, ddp_worldsize
import os
import torch
import torch.distributed as dist

"""
    Applying mpify to the examples in PyTorch Distributed tutorial at
    https://pytorch.org/tutorials/intermediate/dist_tuto.html
"""

"""Blocking point-to-point communication."""
def run_blocking(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        for r in range(1,size):
            tensor += 1
            # Send the tensor to process r
            dist.send(tensor=tensor, dst=r)
            print(f'Rank 0 started blocking send to rank {r}')
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

"""Non-blocking point-to-point communication."""
def run_nonblocking(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        for r in range(1,size):
            tensor = tensor+1
            # Send the tensor to process r
            req = dist.isend(tensor=tensor, dst=r)
            print(f'Rank 0 started nonblocking send to rank {r}')
            req.wait() # Must call req.wait() before next iteration, to avoid data corruption
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print(f'Rank {rank} started nonblocking receive')
        req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

""" All-Reduce example."""
def run_allreduce(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('AllReduce: Rank ', rank, ' has data ', tensor[0])

def runner(fn):
    return fn(ddp_rank(), ddp_worldsize())

if __name__ == "__main__":
    size = torch.cuda.device_count() if torch.cuda.is_available() else 5
    "Pick one of run_blocking, run_nonblocking, and run_allreduce"
    in_torchddp(size, runner, run_blocking)