## Overview 

**`mpify`** is a simple API to launch a "target function" parallelly on a group of *ranked* processes via a single blocking call, with the following features:
   * **Caller process can participate** as a ranked worker (by default as local rank 0)
   * **Return values from all workers** can be gathered in a list, or only that from the caller process (if it participates).
   * **One-use semantic**: sub-processes are terminated upon function completion, thus freeing up resources (e.g. Multiple GPU contexts)
   * **Jupyter/IPython-friendly**, locally defined functions and objects are accessible in the subprocesses via the *target function parameters*
   * **A mechanism for ("`from X import *`") within a function**, and
   * user can **use context manager to manage resources around the execution** of the target function.

`mpify` hopes to make multiprocess experiments in Jupyter notebook a little easier (although it works in batch script as well.):

E.g. `fastai v2` notebook/Jupyter users can train model using multiple GPUs in distributed data parallel (DDP) using the convenient wrapper `in_torchddp()`, and get back the `learner` object in the interactive session upon completion:

### To adapt [the `fastai v2` notebook on training `imagenette`](https://github.com/fastai/course-v4/blob/master/nbs/07_sizing_and_tta.ipynb) to train in DDP on 3 GPUs, in Jupyter.
From:

<img src="/images/imagenette_07_orig.png" height="270">

To this:

<img src="/images/imagenette_07_mpified.png" height="340">


## API and Usage Guide

#### 1. <b>import_star</b>(<i>module_list:[str]</i>)

  When inside a function, '`from X import *`' is syntactically disallowed by Python. Use `import_star(['X',...])` to get around such limitation.


####  2. <b>ranch</b>(<i>nprocs:int, fn:Callable, *args, caller_rank:int=0, gather:bool=True, ctx=None, **kwargs</i>)

  Launch `nprocs` ranked process to execute target function `fn(*args, **kwargs)` in parallel.  Local ranks are from 0 to `nprocs`-1.

  Upon entering the target function `fn`, `nprocs` is available in `os.environ['LOCAL_WORLD_SIZE']`, and the local rank (`0 <= local rank <= nprocs`) in `os.environ['LOCAL_RANK']`, both as strings not integers.

  *Returns*: a list of return values from each process, or that from the caller process.  See `catchall` and `caller_rank` below. 
  
  <i>`nprocs:`</i>Number of worker processes

  <i>`fn, *args, and **kwargs`</i>: Target function and its calling parameters.

  <i>`caller_rank`</i> determines whether the caller process participates in the worker group or not.  To participate, specify an integer between 0..`nprocs`-1 as its local rank; otherwise use `None`. Default to `0`, i.e. caller participates as local rank `0`.

  <i>`gather:bool`</i>: If `True`, *return values from `fn(*args, **kwargs)` in all worker process* will be returned in a list of `nprocs` items, indexed by each process' rank.  If `False`, and if caller participates, will return *its* target function's return value (1 item).  Otherwise return `None`.

  <i>`ctx`</i>: A context manager to wrap around the execution of the target function.  Each process' execution flow would look like:

  <i>spawn -> [`ctx.__enter__()`]-> run function -> [`ctx.__exit__()`] -> terminate</i>

  `ctx` is useful to set up execution environment, acquire and manage resources around the execution.

  For example, <b>mpify.TorchDDPCtx</b> is a context manager to set up the PyTorch `Distributed Data-Parallel` process group, for distributed training of PyTorch model.

A convenient helper  **`in_torchddp()`** constructs a `TorchDDPCtx` then calls `ranch()` with it.  See below.

#### 3. <b>in_torchddp</b>(<i>nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, world_size:int=None, base_rank:int=0, **kwargs</i>):
  
  Initialized/join a PyTorch DDP group of `world_size` members, and launch `fn(*args, **kwargs)` in `nprocs` member processes, starting at global rank `base_rank`.

  `nprocs` - number of worker processes to spawn on this node/host.

  `world_size` - the group size, is stored in `os.environ['WORLD_SIZE']`.  Default to `nprocs`

  `base_rank` - the starting global rank number of the first process in this node/host.  Each process' own global rank is stored in `os.environ['RANK']` before function begins execution.


  `ctx`: custom TorchDDPCtx object.  Otherwise a default `TorchDDPCtx()` instance will be used.
    
  *Returns:* only the `rank-0` execution result.  The initialized DDP group will be destroyed.

#### 4. <b>TorchDDPCtx(world_size:int=None, base_rank:int=0, use_gpu:bool=True, addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs)</b>

A context manager to set-up/tear-down PyTorch's distributed data parallel environment.  `world_size` is the group size of the entire DDP group (which can cover multiple nodes).  `base_rank` is the global rank number of the first participating process on this node.

Before entering the context, user of this manager must set the total number of participating processes on this node in `os.environ['LOCAL_WORLD_SIZE']`, and the *cardinality of current process* (from `0` to 'local world size' - `1`) in `os.environ['LOCAL_RANK']`.  Note that both must be strings, not integers, as everything in `os.environ`.

**TorchDDPCtx** will then set up the appropriate environment variables for PyTorch's DDP execution: `WORLD_SIZE`, `RANK` (the global rank w.r.t. the entire DDP group), `MASTER_ADDR`, `MASTER_PORT` etc..


`use_gpu`: if `True`, will attempt to set up `torch.cuda` to use CUDA GPU `i` for `LOCAL_RANK` `i`, upon entering the context.



```python
from mpify import in_torchddp, ranch, TorchDDPCtx

nprocs = 3

# Using in_torchddp()
result = in_torchddp(nprocs, create_and_train_model)

# The same t'ing, using ranch()
ctx = TorchDDPCtx(world_size=nprocs, base_rank=0)
result = ranch(nprocs, create_and_train_model, caller_rank=0, gather=False, ctx=ctx)
 ```

-----

## Usage 

The rule of thumb rule is: the target function must be self-contained, well encapsulated because it'll be executed on a fresh Python interpreter process.  Therefore:

> <i> Spell all local objects it needs in the parameter list.  Import all modules its needs at the function beginning (use `mpify.import_star(...)` to handle starred imports.) </i>
>
>  <i>Each process can find out its rank in `os.environ['RANK']`, and the group size in `os.environ['WORLD_SIZE']`.</i>

Example: If originally things were defined like this in a notebook.

```ipython
[1] # At the notebook beginning:
    from utils import *
    from fancy import *
    import numpy as np
    import torch
    from torch.distributed import *
    
# some cells later
[2] def foo():
      ...   
# and later
[3] objA = 100
  
[4] def target_fn(*args, **kwargs):
      x = np.array([objA])        # external
      foo(x)
      ...
```
    
Rewrite target_fn like this:
  
```python
[5] def target_fn(foo, objA, *args, **kwargs):
      from mpify import import_star
      import_star(['utils', 'fancy'])  # *-imports here
      import numpy as np               # then other imports
      import torch
      import_star(['torch.distributed'])
      
      # original function body below
      x = np.array([objA])
      foo(x)
      ...

To launch it to 5 ranked processes, and `r` below will receive a list of 5 return values.  `r[i]` is from `rank-i` execution: 

[6] import mpify
    r = mpify.ranch(5, target_fn,foo, objA, ...other args and kwargs)
```


More notebook examples will come along in the future.


## References:

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate. <include link to blog when available> It is also a thinner/lighter/more flexible follow-up to my other repo [`Ddip`](https://github.com/philtrade/Ddip), which uses `ipyparallel`.

* The [`multiprocess` libray](https://github.com/uqfoundation/multiprocess), an actively supported alternative to Python's default `multiprocessing` and `torch.multiprocessing`. 
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

