## Overview 

**`mpify`** is a simple API to launch a "target function" parallelly on a group of *ranked* processes via a single blocking call.  It overcomes the few quirks when *multiple python processes* meets *interactive Jupyter/IPython* meets *multiple CUDA GPUs*, and has the following features:
   * **Caller process can participate** as a ranked worker (by default as local rank 0)
   * **Collect return values from any or all worker procs.**
   * **Worker procs will exit upon function completion**, freeing up resources (e.g. GPUs).
   * **Multi-GPUs friendly**, since subprocesses are spawned not forked, thus immune from any existing CUDA state in caller.
   * **Jupyter-friendly**: modules to import, locally defined functions/objects can be passed to spawned subprocesses, thanks to the `multiprocess` module, a fork of the standard Python `multiprocessing`.
   * **Customizable execution environment around function call** via user defined context manager, and
   * **Minimal changes, if any at all, to existing function**,
   * **A helper routine to "`from X import *`" within a Python function**.

`mpify` hopes to make multiprocessing tasks in Jupyter notebook easier.  It works outside of Jupyter as well, check out the *`examples/`* subdir.


## API and Usage Guide

To run an existing function on multiple spawned processes in Jupyter, user has to 1. import all necessary modules in the spawned processes, and 2. pass any locally defined functions and objects to them as well.

`mpify` provides `ranch()` (ranked launch) to address these two issues, via the `imports:str` and `need:str` parameters.


### 1.  <b>ranch</b>(<i>nprocs:int, fn:Callable, *args, caller_rank:int=0, gather:bool=True, ctx:AbstractContextManager=None, imports:str=None, need:str="", **kwargs</i>)
  > Launch `nprocs` ranked process to execute target function `fn(*args, **kwargs)` in parallel.  Each process will see its rank in `os.environ['LOCAL_RANK']`, and total number of processes `nprocs` in `os.environ['LOCAL_WORLD_SIZE']`, as strings not `int`, like all things in `os.environ`
  > 
  > *Returns*: a list of return values from each process, or that from the caller process.  See the documentation on `caller_rank` and `gather` below.
  
  ***`nprocs`***: Number of worker processes

  ***`fn, *args, and **kwargs`***: Target function and its calling parameters, i.e. original target function usage is `fn(*args, **kwargs)` in single process.

  <i>**`imports:str`**</i>: Multi-line `import` statements, to import modules that the target function needs, in the subprocess.  e.g.:
  
  <i>**`need:str`**</i>: Names of locally defined objects/functions in a string, space-separated.  E.g.
  
  ***`caller_rank:int`*** If it is not None and is an integer between 0..`nprocs`-1, the caller process will participate as a worker in the group at this rank value. If `None`, caller doesn't participate.  Default to `0`, i.e. caller participates as local rank `0`.

  ***`gather:bool`***: If `True`, `ranch()` will return *a list of return values from `fn(*args, **kwargs)` from worker processes*, indexed by each process' local rank.  If `False`, and if 0 <= `caller_rank` < `nprocs-1` , will return the single output from caller's execution of the function, otherwise return `None` when `gather==False` and `caller_rank=None`.

  ***`ctx:AbstractContextManager`***: When provided, the execution flow of each process becomes:

  > Spawn -> *[`ctx.__enter__()`]*-> Run function -> *[`ctx.__exit__()`]* -> Terminate

A convenient helper  **`in_torchddp()`** constructs a `mpify.TorchDDPCtx` context manager then calls `ranch()` with it.  See below.

### 2. <b>in_torchddp</b>(<i>nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, world_size:int=None, base_rank:int=0, **kwargs</i>):
  
  > Initialized/join a PyTorch DDP group of `world_size` members, and launch `fn(*args, **kwargs)` in `nprocs` member processes, starting at global rank `base_rank`.  The DDP group will be destroyed upon exit.
  > 
  > *Returns:* Only the `rank-{base_rank}` execution result.


  ***`nprocs`*** - number of worker processes to spawn on this node/host.

  ***`world_size`*** - the group size, is stored in `os.environ['WORLD_SIZE']`.  Default to `nprocs`.

  ***`base_rank`*** - the starting global rank number of the first process in this node/host.  Each process' own global rank is stored in `os.environ['RANK']` before function begins execution.


  ***`ctx`***: custom TorchDDPCtx object. Otherwise a default `TorchDDPCtx()` instance will be used.
    
#### 3. <b>TorchDDPCtx(world_size:int=None, base_rank:int=0, use_gpu:bool=True, addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs)</b>

> A context manager to set-up/tear-down PyTorch's distributed data-parallel group of  `world_size` processes, starting at `base_rank` on this node.
> 
> Requirement: Before `__enter__()` the context, user must set the total number of local participating processes in `os.environ['LOCAL_WORLD_SIZE']`, and the local rank of the current process in `os.environ['LOCAL_RANK']`.  Note: all `os.environ` values are strings.
> 
> Once in context, PyTorch's DDP attributes: `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT` etc, will be available in `os.environ`.

***`use_gpu`***: if `True`, will also try to set up `torch.cuda` to use CUDA GPU according to `LOCAL_RANK`.

-----

More notebook examples will come along in the future.


## References:

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate. <include link to blog when available> It is also a thinner/lighter/more flexible follow-up to my other repo [`Ddip`](https://github.com/philtrade/Ddip), which uses `ipyparallel`.

* The [`multiprocess` libray](https://github.com/uqfoundation/multiprocess), an actively supported alternative to Python's default `multiprocessing` and `torch.multiprocessing`. 
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

