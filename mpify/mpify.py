import os, sys, inspect, multiprocess as mp
from typing import Callable
from contextlib import AbstractContextManager, nullcontext
import torch
'''
TODO: if in IPython, user defined local functions can be obtained, thus exportable to child process via kwargss:
     fs = [ f for k,f in shell.user_ns.items() 
        if type(f) == types.FunctionType and not k.startswith('_') and k == f.__name__ and f.__module__ == '__main__' ]

    Can imported modules lines be discovered and sent over as well?
    - yes, if in IPython, examine shell.user_ns dict, for all "_i{n}", look for:
      "from x import y" and "import x as y", build the list and pass to contextualize.

    The following seems to work in the target function of Process():
    import importlib

    # import torch
    globals()['torch'] = importlib.import_module('torch')

    # import torch.distributed as dist
    globals()['dist' ] = importlib.import_module('torch.distributed')

'''
__all__ = ['import_star', 'ranch', 'TorchDDPCtx', 'in_torchddp', 'ddp_rank', 'ddp_worldsize']

def import_star(modules:[str]=[], g:dict=None):
    "Apply `from module import '*'` into caller's frame from a list of modules."
    cf = inspect.currentframe()
    if g is None: g = cf.f_back.f_globals
    try:
        for mod in modules:
            try:
                m = __import__(mod, fromlist=['*'])
                to_import = {}
                for name in getattr(m, "__all__", [n for n in dir(m) if not n.startswith('_')]):
                    to_import[name] = getattr(m, name)
                g.update(to_import)
            except Exception as e: raise ImportError(f"Failed to import module {mod}") from e
    finally: del cf   # per https://docs.python.org/3/library/inspect.html#the-interpreter-stack

def _contextualize(i:int, nprocs:int, fn:Callable, cm:AbstractContextManager, l=None, env:dict={}):
    "Return a function that will setup os.environ and execute a target function within a context manager."
    if l: assert i < len(l), ValueError("Invalid index {i} > result list size: {len(l)}")
    def _cfn(*args, **kwargs):
        os.environ.update({"LOCAL_RANK":str(i), "LOCAL_WORLD_SIZE":str(nprocs)})
        try:
            '''
            The following snippet import modules into function 'fn()''s globals scope.

            import inspect, importlib
            try: g_member = next(filter(lambda t: t[0] == '__globals__', inspect.getmembers(fn)))
            except StopIteration: raise KeyError("Wow what kind of object is fn?! no '__globals__'? ")
            m = __import__('torch')

            # g_member[1] is the same as "globals()" inside "fn()"
            g_member[1]['torch'] = m
            import_star(['numpy'], g_member[1])
            g_member[1]['dist' ] = importlib.import_module('torch.distributed')

            for o in objlist: g_member[1][o] = outside[o]
            '''
            import inspect, importlib
            try: g_member = next(filter(lambda t: t[0] == '__globals__', inspect.getmembers(fn)))
            except StopIteration: raise KeyError("Wow what kind of object is fn?! no '__globals__'? ")
            # m = __import__('torch')
            # g_member[1] is the same as "globals()" inside "fn()"
            # g_member[1]['torch'] = m
            # import_star(['numpy'], g_member[1])

            for k,v in env.items(): g_member[1][k] = v # pass in explicit objects

            with cm or nullcontext(): r = fn(*args, **kwargs)
            if l: l[i] = r
            return r
        finally: map(lambda k: os.environ.pop(k, None), ("LOCAL_RANK", "LOCAL_WORLD_SIZE"))               
    return _cfn

def ranch(nprocs:int, fn:Callable, *args, caller_rank:int=0, gather:bool=True, ctx=None, need:str=None, **kwargs):
    '''Spawn `nprocs` ranked process and launch `fn(*args, **kwargs)`. Caller process can participate as `caller_rank`.
    Apply ctx mgr if provided. `gather`: If True (default), return list of all function return values; otherwise return that executed in the parent rank process, and that requires `caller_rank` be defined.
    '''
    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    
    children_ranks = list(range(nprocs))
    if caller_rank is not None:
        assert 0 <= caller_rank < nprocs, ValueError(f"Invalid caller_rank {caller_rank}, must satisfy 0 <= caller_rank < {nprocs}")
        children_ranks.pop(caller_rank)

    print(f"Spawning ranks: [{children_ranks}]")

    multiproc_ctx, procs = mp.get_context("spawn"), []
    result_list = multiproc_ctx.Manager().list([None] * nprocs) if gather else None
    try: # First launch the execution in the subprocesses, then in current process if caller_rank is not None
        env = {}
        if need is not None:
            cf = inspect.currentframe()
            caller_g = cf.f_back.f_globals
            objs = need.split()
            for o in objs: env[o] = caller_g[o]
            del cf

        for rank in children_ranks:
            target_fn = _contextualize(rank, nprocs, fn, cm=ctx, l=result_list, env=env)
            p = multiproc_ctx.Process(target=target_fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()
        p_res = (_contextualize(caller_rank, nprocs, fn, cm=ctx, l=result_list))(*args, **kwargs) if caller_rank is not None else None
        for p in procs: p.join()
        return result_list if gather else p_res
    finally:
        for p in procs: p.terminate(), p.join()

class TorchDDPCtx(AbstractContextManager):
    "Setup/teardown Torch DDP when entering/exiting a `with` clause. os.environ[] must define 'LOCAL_RANK' prior to __enter__() "
    def __init__(self, *args, world_size:int=None, base_rank:int=0, use_gpu:bool=True,
                 addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs):
        assert world_size and (base_rank >= 0 and world_size > base_rank), ValueError(f"Invalid world_size {world_size} or base_rank {base_rank}. Need to be: world_size > base_rank >=0 ")
        self._ws, self._base_rank = world_size, base_rank
        self._a, self._p, self._nt = addr, str(port), str(num_threads)
        self._use_gpu, self._myddp, self._backend = use_gpu, False, 'gloo' # default to CPU backend

    def __enter__(self):
        try: local_rank, local_ws = int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])
        except KeyError as e: raise KeyError(f"os.environ['LOCAL_RANK'] and os.environ['LOCAL_RANK'] must be set prior to entering the context.") from e

        assert 0 < local_ws <= self._ws, ValueError(f"Invalid os.environ['LOCAL_WORLD_SIZE']: {local_ws}, should be 0 < ws <= {self._ws}!")

        rank = local_rank + self._base_rank
        assert rank < self._ws, ValueError(f"local_rank {local_rank} + base_rank {self._base_rank}, should be < ({self._ws})")

        if self._use_gpu and torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
                self._backend = 'nccl'
            except RuntimeError as e: self._use_gpu = False
            print(f"Rank [{rank}] using CUDA GPU {local_rank}" if self._use_gpu else f"Failed to set CUDA device to {local_rank}. Invalid os.environ['LOCAL_RANK']?", file=sys.stderr, flush=True)

        os.environ.update({"WORLD_SIZE":str(self._ws), "RANK":str(rank),
            "MASTER_ADDR":self._a, "MASTER_PORT":self._p, "OMP_NUM_THREADS":self._nt})
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self._backend, init_method='env://')
            self._myddp = torch.distributed.is_initialized()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._myddp: torch.distributed.destroy_process_group()
        if self._use_gpu and torch.cuda.is_available(): torch.cuda.empty_cache()
        for k in ["WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]: os.environ.pop(k, None)
        return exc_type is None

def ddp_rank(): return int(os.environ['RANK'])
def ddp_worldsize(): return int(os.environ['WORLD_SIZE'])

def in_torchddp(nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, world_size:int=None, base_rank:int=0, need:str=None, **kwargs):
    "Launch `fn(*args, **kwargs)` in Torch DDP group of 'world_size' members on `nprocs` local processes RANK from ['base_rank'..'nprocs'-1]"
    if world_size is None: world_size = nprocs
    assert base_rank + nprocs <= world_size, ValueError(f"nprocs({nprocs}) + base_rank({base_rank}) must be < world_size({world_size})")
    if ctx is None: ctx = TorchDDPCtx(world_size=world_size, base_rank=base_rank, need=need)
    return ranch(nprocs, fn, *args, caller_rank=0, gather=False, ctx=ctx,  **kwargs)