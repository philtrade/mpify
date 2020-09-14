## Overview 

**`mpify`** is a thin library to run function on multiple processes.  It enables multi-GPU distributed training *inside a Jupyter notebook* by:
  * allowing the Jupyter foreground process to participate as a worker and collect results
  * making objects and functions defined in the Jupyter notebook accessible to subprocesses

The wrapper `mpify.ranch(n, fn, ...)` returns when `fn` completes in all processes, and accepts a context manager to be applied around `fn(...)`.  Each subprocess is `spawn`ed (not forked), and is pre-assigned a rank in `os.environ['LOCAL_RANK']`.

With these, `mpify.in_torchddp(n, fn,...)` sets up/tears down a Torch DDP group for distributed training behind the scene.

Although `mpify` works standalone Python app too, it is designed with single-node, multi-GPUs usage in mind.  For asynchronous, distributed workloads on multiple nodes (cluster), please use `ipyparallel`, `dask`, or `ray`, as they take care of scheduling, fault tolerance etc..


### Example: Porting the first training loop in Fastai2's course-v4 chapter 01_intro notebook to train on 3 GPUs in Torch's DDP mode:

Original:

<img src="/images/01_intro_train_cnn_orig.png" height="350">

`mpify`-ed:

<img src="/images/01_intro_train_cnn_mpify.png" height="450">

## More Examples
The [examples/](/examples) directory contains:
  * A PyTorch tutorial on DDP, ported to use `mpify` both in Jupyter, or as a standalone script.
  * Several `fastai2 course-v4` notebooks ported to use `mpify`, and to train in distributed data-parallel.

Interesting use cases you wish to share, and bug reports are welcome.

## Install

**Latest Release is v0.1.0**: python3 -m pip install git+https://github.com/philtrade/mpify@v0.1.0

**Latest Dev. version**: python3 -m pip install git+https://github.com/philtrade/mpify


## API and Usage Guide

To run an existing function on multiple spawned processes in Jupyter, user has to import all necessary modules in the spawned processes, and pass any locally defined functions and objects to them.  It's often necessary to acquire resources, set up environment anew in each subprocess.  Finally the Jupyter caller shell might want to collect the function results for further manipulation in the shell (i.e. not in the subprocesses).

`mpify` provides `ranch()` ("ranked launch"), a sample context manager `TorchDDPCtx`, and `in_torchddp()` to address these issues.


#### 1.  <b>ranch</b>(<i>nprocs:int, fn:Callable, *args, caller_rank:int=0, gather:bool=True, ctx:AbstractContextManager=None, imports:str=None, need:str="", **kwargs</i>)
  > Launch `nprocs` ranked processes to execute target function `fn(*args, **kwargs)` in parallel.  Each process will see its rank in `os.environ['LOCAL_RANK']`, and total number of processes `nprocs` in `os.environ['LOCAL_WORLD_SIZE']`, as strings not `int`, like all things in `os.environ`.
  > 
  > *Returns*: a list of return values from each process, or that from the caller process.  See the documentation on `caller_rank` and `gather` below.
  
  ***`nprocs`***: Number of local worker processes.

  ***`fn, *args, and **kwargs`***: Target function and its calling parameters, i.e. original target function usage is `fn(*args, **kwargs)` in single process.

  <i>**`imports:str`**</i>: Multi-line `import` statements, to import modules that the target function needs, in the subprocess.
  
  <i>**`need:str`**</i>: Names of locally defined objects/functions in a string, space-separated.
  
  ***`caller_rank:int`*** If it is an integer between 0..`nprocs`-1, the caller process will participate as a worker in the group at this rank value. If `None`, caller doesn't participate.  Default to `0`, i.e. caller participates as local rank `0`.

  ***`gather:bool`***: If `True`, `ranch()` will return **a list of** return values from `fn(*args, **kwargs)` from worker processes, indexed by each process' local rank.  If `False`, and if 0 <= `caller_rank` < `nprocs-1` , will return **the single output** from caller's execution of the function, otherwise return `None` when `gather==False` and `caller_rank=None`.

  ***`ctx:AbstractContextManager`***: When provided, the execution flow of each process becomes:

  > Spawn -> *[`ctx.__enter__()`]*-> Run function -> *[`ctx.__exit__()`]* -> Terminate

A convenient helper  **`in_torchddp()`** constructs a `mpify.TorchDDPCtx` context manager then calls `ranch()` with it.  See below.

#### 2. <b>in_torchddp</b>(<i>nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, world_size:int=None, base_rank:int=0, **kwargs</i>):
  
  > Initialized/join a PyTorch DDP group of `world_size` members, and launch `fn(*args, **kwargs)` in `nprocs` member processes, starting at global rank `base_rank`.  The DDP group will be destroyed upon exit.
  > 
  > *Returns:* Only the `rank-{base_rank}` execution result.


  ***`nprocs`*** - number of worker processes to spawn on this node/host.

  ***`world_size`*** - the group size, is stored in `os.environ['WORLD_SIZE']`.  Default to `nprocs`.

  ***`base_rank`*** - the starting global rank number of the first process in this node/host.  Each process' own global rank is stored in `os.environ['RANK']` before function begins execution.


  ***`ctx`***: custom TorchDDPCtx object. Otherwise a default `TorchDDPCtx()` instance will be used.
    
#### 3. <b>TorchDDPCtx(<i>world_size:int=None, base_rank:int=0, use_gpu:bool=True, addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs</i>)</b>

> A context manager to set-up/tear-down PyTorch's distributed data-parallel group of  `world_size` processes, starting at `base_rank` on this node.
> 
> Requirement: Before `__enter__()` the context, user must set the total number of local participating processes in `os.environ['LOCAL_WORLD_SIZE']`, and the local rank of the current process in `os.environ['LOCAL_RANK']`.  Note: all `os.environ` values are strings.
> 
> Once in context, PyTorch's DDP attributes: `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT` etc, will be available in `os.environ`.

***`use_gpu`***: if `True`, `__enter__()` will also set up `torch.cuda` by calling `torch.cuda.set_device(int(os.environ['LOCAL_RANK']))`

#### 4. <b>import_star(<i>modules:[str], globals:dict=None</i>)</b>

> Perform `from m import *` for the list of ***`modules`***.  By default, the names will be imported into the `__main__` module's global namespace, but user may provide its own target namespace via ***`globals`***.

-----

## References:

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate. <include link to blog when available> It is also a thinner/lighter/more flexible follow-up to my other repo [`Ddip`](https://github.com/philtrade/Ddip), which uses `ipyparallel`.

* The [`multiprocess` libray](https://github.com/uqfoundation/multiprocess), an actively supported alternative to Python's default `multiprocessing` and `torch.multiprocessing`. 
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3
