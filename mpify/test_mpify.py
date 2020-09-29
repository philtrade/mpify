import pytest

from mpify import *
    
def fn(*args, **kwargs):
    import os
    r = int(os.environ['LOCAL_RANK'])
    ws = int(os.environ['LOCAL_WORLD_SIZE'])
    return { 'local_rank': r, 'local_ws': ws, 'pid': os.getpid(),
        '_args' : args, **kwargs }

@pytest.mark.parametrize(("nprocs", "caller_rank"), [ (1, None), (3, 0), (3,2) ])
@pytest.mark.parametrize("args", [('arg_1', 20, 'arg_3'), () ])
@pytest.mark.parametrize("kwargs", [{}, {'a': 'foo', 'b':'goo', 'c':100}])
def test_env_setup(nprocs, caller_rank, args, kwargs):
    import os
    # nprocs = 5
    # args = ('arg_1', 'arg_2', 'arg_3')
    res = ranch(nprocs, fn, *args, caller_rank=caller_rank, gather=True, imports='import os', **kwargs)
    pids = set()

    for i in range(nprocs):
        r = res[i]
        pids.add(r['pid'])
        if caller_rank is not None and i == caller_rank:
            assert r['pid'] == os.getpid()
        assert r['local_rank'] == i
        assert r['local_ws'] == nprocs
        assert r['_args'] == args

    assert len(pids) == nprocs

""" All-Reduce example."""
def run_allreduce(rank, size):
    import torch
    """ Simple point-to-point communication. """
    tensor = torch.tensor(rank)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # print('AllReduce: Rank ', rank, ' has data ', tensor.item(), flush=True)
    return tensor.item()

def rank_size_wrapper(fn):
    def new_fn(*args, **kwargs):
        import os
        return fn(int(os.environ.get('RANK')), int(os.environ.get('WORLD_SIZE')), *args, **kwargs)
    return new_fn

@pytest.mark.parametrize("nprocs", [5])
def test_torch_ddp(nprocs):
    "Pick one of run_blocking, run_nonblocking, and run_allreduce"
    fn = rank_size_wrapper(run_allreduce)
    t = in_torchddp(nprocs, fn, use_gpu=False)
    assert t == sum(range(nprocs))

