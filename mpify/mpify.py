import os, sys, re, multiprocess as mp
from typing import Callable
from contextlib import AbstractContextManager, nullcontext
import torch

__all__ = ['import_star', 'global_imports', 'ranch', 'TorchDDPCtx', 'in_torchddp']

#  - globals() doesn't necessarily return the '__main__' global scope when inside package function,
#    thus we use sys.modules['__main__'].__dict__.

def import_star(modules:[str], ns:dict=None):
    """Import ``*`` from a list of module, into namespace ns (default to '__main__')

    Args:
        modules: list of modules or packages
        ns: destination namespace, optional. If not provided, will default to '__main__'

    """
    global_imports([f"from {m} import *" for m in modules], ns)

def global_imports(imports:[str], ns:dict=None):
    """
    Parse and execute multiple import statements, and import into target namespace 'ns'

    Args:
        imports: list of import statements, as in Python code.  Supported formats include:

            * import x, y, z as z_alias

            * from A import x

            * from A import z as z_alias

            * from A import x, y, z as z_alias

            Not supported: 'from A import (a, b)'

        ns: target namespace to import into.  Default to '__main__'
    """

    if ns is None:
        import sys
        ns = sys.modules['__main__'].__dict__
    pat = re.compile(r'^\s*?(?:from\s+?(\S+?)\s+?)?import\s+?(.+)$')
    pat_as = re.compile(r'^\s*?(\S+?)(?:\s*?as\s+?(\S+?))?\s*$')
    for parsed in filter(lambda p:p,[pat.match(i) for i in imports]):
        (from_, imp_)  = parsed.groups()
        imps = imp_.split(',')

        # Parse "from X import ..."
        from_mod = __import__(from_, fromlist=['']) if from_ else None

        for name in imps: # each comma-separated item in import a, b, x as y
            (x, y) = pat_as.match(name).groups()
            if y is None: y=x
            if x == '*': # Handle starred import: 'from X import *'
                assert from_, SyntaxError(f"From what <module> are you trying to 'import *': {parsed.string}")
                importables = getattr(from_mod, "__all__", [n for n in dir(from_mod) if not n.startswith('_')])
                for o in importables: ns[o] = getattr(from_mod, o)
            else: # x is either a name in 1 module, OR a module itself
                ns[y] = getattr(from_mod, x) if from_ else __import__(x, fromlist=[''])

def _contextualize(i:int, nprocs:int, fn:Callable, cm:AbstractContextManager, l=None, env:dict={}, imports=""):
    "Return a function that will setup os.environ and execute a target function within a context manager."
    if l: assert i < len(l), ValueError("Invalid index {i}, exceeds size of the result list: {len(l)}")
    def _cfn(*args, **kwargs):
        import os
        os.environ.update({"LOCAL_RANK":str(i), "LOCAL_WORLD_SIZE":str(nprocs)})
        try:
            import sys
            from mpify import global_imports
            # import env into '__main__', which can be in a subprocess here.
            g = sys.modules['__main__'].__dict__
            global_imports(imports.split('\n'), g)
            g.update(env)
            with cm or nullcontext(): r = fn(*args, **kwargs)
            if l: l[i] = r
            return r
        finally: map(lambda k: os.environ.pop(k, None), ("LOCAL_RANK", "LOCAL_WORLD_SIZE"))               
    return _cfn

def ranch(nprocs:int, fn:Callable, *args, caller_rank:int=0, gather:bool=True, ctx:AbstractContextManager=None, need:str="", imports="", **kwargs):
    """ Execute `fn(\*args, \*\*kwargs)` distributedly in `nprocs` processes.  User can
    serialize over objects and functions, spell out import statements, manage execution
    context, gather results, and the parent process can participate as one of the workers.

    If `caller_rank` is `0 <= caller_rank < nprocs`, only `nprocs - 1` processes will be forked, and the caller process will be a worker to run its share of `fn(..)`.

    If `caller_rank` is ``None``, `nprocs` processes will be forked.

    Inside each worker process, its relative rank among all workers is set up in `os.environ['LOCAL_RANK']`, and the total
    number of workers is set up in `os.environ['LOCAL_WORLD_SIZE']`, both as strings.

    Then import statements in `imports`, followed by any objects/functions in `need`, are brought
    into the python global namespace.

    Then, context manager `ctx` is applied around the call `fn(\*args, \*\*kwargs)`.

    Return value of each worker can be gathered in a list (indexed by the process's rank)
    and returned to the caller of `ranch()`.

    Args:
        nprocs: Number of processes to fork.  Visible as a string in `os.environ['LOCAL_WORLD_SIZE']`
            in all worker processes.
        fn: Function to execute on the worker pool
        \*args: Positional arguments by values to `fn(\*args....)`
        \*\*kwargs: Named parameters to `fn(x=..., y=....)`
        caller_rank: Rank of the parent process.  ``0 <= caller_rank < nprocs`` to join, ``None`` to opt out. Default to ``0``.

            In distributed data parallel, 0 means the leading process.
        gather: if ``True``, `ranch` will return a list of return values from each worker, indexed by their ranks.
            If ``False``, and if 'caller_rank' is not None (meaning parent process is a worker),
            `ranch()` will return whatever the parent process' `fn(...)` returns.
        ctx: User defined context manager to be used in a 'with'-clause around the 'fn(...)' call in worker processes.
            Subclassed from AbstractContextManager, ctx needs to define '__enter__()' and '__exit__()' methods.
        need: Space-separated names of objects/functions to be serialized over to the subprocesses.
        imports: A multiline string of `import` statements to execute in the subprocesses
            before `fn()` execution.  Supported formats:

            * `import x, y, z as zoo`

            * `from A import x`

            * `from A import z as zoo`

            * `from A import x, y, z as zoo`

            * Not supported: `from A import (x, y)`

    Returns:
        ``None``, or list of results from worker processes, indexed by their `LOCAL_RANK`: ``[res_0, res_1, .... res_{nprocs-1}]``
    """

    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    
    children_ranks = list(range(nprocs))
    if caller_rank is not None:
        assert 0 <= caller_rank < nprocs, ValueError(f"Invalid caller_rank {caller_rank}, must satisfy 0 <= caller_rank < {nprocs}")
        children_ranks.pop(caller_rank)
    multiproc_ctx, procs = mp.get_context("spawn"), []
    result_list = multiproc_ctx.Manager().list([None] * nprocs) if gather else None
    try:
        # pass globals in this process to subprocess via fn's wrapper, 'target_fn'
        env = {k : sys.modules['__main__'].__dict__[k] for k in need.split()}
        for rank in children_ranks:
            target_fn = _contextualize(rank, nprocs, fn, cm=ctx, l=result_list, env=env, imports=imports)
            p = multiproc_ctx.Process(target=target_fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()
        p_res = (_contextualize(caller_rank, nprocs, fn, cm=ctx, l=result_list, env=env, imports=imports))(*args, **kwargs) if caller_rank is not None else None
        for p in procs: p.join()
        return result_list if gather else p_res
    finally:
        for p in procs: p.terminate(), p.join()

class TorchDDPCtx(AbstractContextManager):
    """
    A context manager to set up and tear down a PyTorch distributed data parallel process group.
    `os.environ['LOCAL_RANK']` must be defined prior to `__enter__()`.

    Args:
        world_size: total number of members in the DDP group
        base_rank: the starting, lowest rank value of among the forked local processes
        use_gpu: if True, will set the default CUDA device base on `os.environ['LOCAL_RANK']`
        addr, port, num_threads: see PyTorch distributed data parallel documentation.
    """
    def __init__(self, *args, world_size:int=None, base_rank:int=0, use_gpu:bool=True,
                 addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs):
        assert world_size and (base_rank >= 0 and world_size > base_rank), ValueError(f"Invalid world_size {world_size} or base_rank {base_rank}. Need to be: world_size > base_rank >=0 ")

        self._ws, self._base_rank = world_size, base_rank
        self._a, self._p, self._nt = addr, str(port), str(num_threads)
        self._use_gpu = use_gpu and torch.cuda.is_available()
        self._myddp, self._backend = False, 'gloo' # default to CPU backend

    def __enter__(self):
        import os
        try: local_rank, local_ws = int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])
        except KeyError: raise KeyError(f"'LOCAL_RANK' or 'LOCAL_RANK' not found in os.environ")

        assert 0 < local_ws <= self._ws, ValueError(f"Invalid 'LOCAL_WORLD_SIZE': {local_ws}, should be 0 < ws <= {self._ws}!")

        rank = local_rank + self._base_rank
        assert rank < self._ws, ValueError(f"local_rank {local_rank} + base_rank {self._base_rank}, should be < ({self._ws})")

        if self._use_gpu:
            assert local_rank<torch.cuda.device_count(), ValueError(f"LOCAL_RANK {local_rank} > available CUDA devices")
            try:
                torch.cuda.set_device(local_rank)
                self._backend = 'nccl'
                print(f"Rank [{rank}] using CUDA GPU {local_rank}", flush=True)
            except RuntimeError as e:
                self._use_gpu = False;
                print(f"Unable to set cuda device {local_rank}, using CPU. {e}", flush=True)

        os.environ.update({"WORLD_SIZE":str(self._ws), "RANK":str(rank),
            "MASTER_ADDR":self._a, "MASTER_PORT":self._p, "OMP_NUM_THREADS":self._nt})
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self._backend, init_method='env://')
            self._myddp = torch.distributed.is_initialized()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._myddp: torch.distributed.destroy_process_group()
        if self._use_gpu: torch.cuda.empty_cache()
        for k in ["WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]: os.environ.pop(k, None)
        return exc_type is None

def in_torchddp(nprocs:int, fn:Callable, *args, world_size:int=None, base_rank:int=0,
                ctx:TorchDDPCtx=None, need:str="", imports:str="", **kwargs):
    """A convenience routine to prepare a context manager for PyTorch Distributed Data Parallel group setup/teardown,
    then calls `ranch()` to fork and execute `fn(*args, **kwargs)`

    Args:
        nprocs: Number of local processes to fork
        fn, \*args, \*\*kwargs: the functions and its arguments
        world_size: total number of members in the entire PyTorch DDP group
        base_rank: the lowest, starting rank of in the local processes
        ctx: by default will use `mpify.TorchDDPCtx` to set up torch distributed group,
            but user can override it with their own if necessary.
        need: names of local objects to serialize over, comma-separated
        imports: multi-line import statements, to apply in each forked process.
    
    Returns:
        The result of `fn(*args, **kwargs)` in the rank `base_rank` execution.
    """
    if world_size is None: world_size = nprocs
    assert base_rank + nprocs <= world_size, ValueError(f"nprocs({nprocs}) + base_rank({base_rank}) must be < world_size({world_size})")
    if ctx is None: ctx = TorchDDPCtx(world_size=world_size, base_rank=base_rank)
    return ranch(nprocs, fn, *args, caller_rank=0, gather=False, ctx=ctx, need=need, imports=imports, **kwargs)