import os, inspect, multiprocess as mp
from typing import Callable
from contextlib import AbstractContextManager, nullcontext
import torch

__all__ = ['import_star', 'ranch', 'TorchDDPCtx', 'in_torchddp']

def import_star(modules=[]):
    "Apply `from module import '*'` into caller's frame from a list of modules."
    cf = inspect.currentframe()
    g = cf.f_back.f_globals
    try:
        for mod in modules:
            try:
                m = __import__(mod, fromlist=['*'])
                to_import = {}
                for name in getattr(m, "__all__", [n for n in dir(m) if not n.startswith('_')]):
                    to_import[name] = getattr(m, name)
                g.update(to_import)
            except Exception as e: raise ImportError(f"Failed to import module {mod}") from e
    finally:
        del cf   # Recommendation from https://docs.python.org/3/library/inspect.html#the-interpreter-stack

def _contextualize(i, g, ws, fn:Callable, cm:AbstractContextManager, l=None):
    "Return a function that will setup os.environ and execute a target function within a context manager."
    if l: assert i < len(l), ValueError("Invalid index {i} > result list size: {len(l)}")
    def _cfn(*args, **kwargs):
        os.environ.update({"LOCAL_RANK":str(i), "RANK":str(g), "WORLD_SIZE":str(ws)})
        with cm or nullcontext(): r = fn(*args, **kwargs)
        if l: l[i] = r
        for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"): os.environ.pop(k) 
        return r
    return _cfn

def ranch(nprocs:int, fn:Callable, *args, parent_rank:int=0, catchall:bool=True, host_rank:int=0, ctx=None, **kwargs):
    '''Spawn `nprocs` ranked process and launch `fn(*args, **kwargs)`. Caller process can participate as `parent_rank`.
    Apply ctx mgr if provided. `catchall`: If True (default), return list of all function return values; otherwise return that executed in the parent rank process, and that requires `parent_rank` be defined.
    '''
    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    
    children_ranks = list(range(nprocs))
    if parent_rank is not None:
        assert 0 <= parent_rank < nprocs, ValueError(f"Out of range parent_rank:{parent_rank}, must be 0 <= parent_rank < {nprocs}")
        children_ranks.pop(parent_rank)

    multiproc_ctx = mp.get_context("spawn")
    p_res, result_list = None, multiproc_ctx.Manager().list([None] * nprocs) if catchall else None
    procs, base_rank = [], host_rank * nprocs
    try:
        for rank in children_ranks:
            target_fn = _contextualize(rank, rank+base_rank, nprocs, fn, ctx, result_list)
            p = multiproc_ctx.Process(target=target_fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()
        if parent_rank is not None: # also run target in current process at a rank
            p_res = (_contextualize(parent_rank, parent_rank+base_rank, nprocs, fn, ctx, result_list))(*args, **kwargs)
        return result_list if catchall else p_res
    except Exception as e:
        raise Exception(e) from e
    finally:
        for p in procs: p.join()

class TorchDDPCtx(AbstractContextManager):
    "Setup/teardown Torch DDP when entering/exiting a `with` clause."
    def __init__(self, *args, addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs):
        self._a, self._p, self._nt = addr, str(port), str(num_threads)
        self._myddp, self._backend = False, 'gloo' # default to CPU backend

    def __enter__(self):
        os.environ.update({"MASTER_ADDR":self._a, "MASTER_PORT":self._p, "OMP_NUM_THREADS":self._nt})
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
            self._backend = 'nccl'
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self._backend, init_method='env://')
            self._myddp = torch.distributed.is_initialized()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._myddp: torch.distributed.destroy_process_group()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        for k in ["MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]: os.environ.pop(k, None)
        return exc_type is None

def in_torchddp(nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, **kwargs):
    "Launch `fn(*args, **kwargs)` in Torch DDP group of `nprocs` processes.  Can customize the TorchddpCtx context."
    if ctx is None: ctx = TorchDDPCtx()
    return ranch(nprocs, fn, *args, parent_rank=0, catchall=False, ctx=ctx,  **kwargs)