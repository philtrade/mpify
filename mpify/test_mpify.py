import pytest

from mpify import *

def fn(*args, **kwargs):
    import os
    r = int(os.environ['LOCAL_RANK'])
    ws = int(os.environ['LOCAL_WORLD_SIZE'])
    return { 'local_rank': r, 'local_ws': ws, 'pid': os.getpid(),
        '_args' : args, **kwargs }

@pytest.mark.parametrize(("nprocs", "caller_rank"), [ (1, 0), (1, None), (3, 0), (3,2) ])
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

    
    