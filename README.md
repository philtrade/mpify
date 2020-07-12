## Overview 

**`mpify`** is a simple API to launch a "target function" parallelly to multiple *ranked* processes via a single blocking call, with the following features:
   * **caller process may participate** as a ranked worker and receive the return object from taret function,
   * sub-processes will exit upon function return, thus **freeing up resources (e.g. Multiple GPU contexts)**,
   * when used in Jupyter notebook/IPython, **locally defined functions and objects can be passed to the target function in subprocesses**,
   * provide a mechanism to **support starred import ("`from X import *`") within the target function**, and
   * user can **use context manager to manage resources around the execution** of the target function.

`mpify` hopes to make multiprocess experiments in Jupyter notebook a little easier (although it works in batch script as well.):

E.g. `fastai v2` notebook/Jupyter users can train model using multiple GPUs in distributed data parallel (DDP) using the convenient wrapper `in_torchddp()`, and get back the `learner` object in the interactive session upon completion:

### To adapt [the `fastai v2` notebook on training `imagenette`](https://github.com/fastai/course-v4/blob/master/nbs/07_sizing_and_tta.ipynb) to train in DDP on 3 GPUs, in Jupyter.
From:

<img src="/images/imagenette_07_orig.png" height="270">

To this:

<img src="/images/imagenette_07_mpified.png" height="340">


## API and Usage Guide

#### <b>import_star</b>(<i>module_list:[str]</i>)

  When inside a function, '`from X import *`' is syntactically disallowed by Python. Use `import_star(['X',...])` to get around such limitation.


####  <b>ranch</b>(<i>nprocs:int, fn:Callable, *args, caller_rank:int=0, catchall:bool=True, host_rank:int=0, ctx=None, **kwargs</i>)

  Launch a group of ranked process (0..`nprocs-1` inclusive) to execute target function `fn(*args, **kwargs)` in parallel.

  A few generic distributed attributes will be available to `fn()`. *Local rank*, *global rank*, *world size* as defined in PyTorch's [distributed data parallel convention](https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940), are stored in `os.environ['LOCAL_RANK, RANK, and WORLD_SIZE']`, as strings .

  *Returns*: a list of return values from each process, or that from the caller process.  See `catchall` and `caller_rank` below. 
  
  <i>`nprocs:`</i>Number of worker processes

<i>`fn, *args, and **kwargs`</i>: Target function and its calling parameters.

<i>`caller_rank`</i>: `None`, if the caller does not participate in the worker group, or a *rank* between 0 and `nprocs-1` (inclusive), if the caller process will participate and run the target function as well.  Default to `0`, i.e. the caller process will be the *rank 0* process in the process group.

<i>`catchall:bool`</i>: Determines the return value of the call.  If True, `ranch()` will gather and return the *return values from all processes* in a list.  If `False`, and if `caller_rank` is defined, return the value of the `caller_rank` execution of the target function.  Otherwise `ranch()` will return `None`.

<i>`ctx`</i>: A context manager which wraps the execution of the target function.  Each process' execution flow would look like:

<i>spawn -> [`ctx.__enter__()`]-> run function -> [`ctx.__exit__()`] -> terminate</i>

When combined with custom context manager, user can set up execution envrionment, acquire and manage resources at entry/exit.

For example, to set up a distributed data parallel process group for distributed training of PyTorch models, one may use the provided  context manager <b>TorchDDPCtx</b>, or more conveniently, use **`in_torchddp()`** which is a wrapper to `ranch()` and it uses `TorchDDPCtx`:

#### <b>in_torchddp</b>(<i>nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, **kwargs</i>):

  Launch `fn` to an initialized torch distributed data parallel (DDP) group with `nprocs` processes, and the caller process itself would be `rank-0`.

  *ctx*: use a custom TorchDDPCtx object.  Otherwise a default `TorchDDPCtx()` instance will be used.
    
  *Returns:* only the `rank-0` execution result.  The initialized DDP group will be destroyed.


> The following `in_torchddp(...)` and `ranch(...)` calls are equivalent:

```python
from mpify import in_torchddp, ranch, TorchDDPCtx

result = in_torchddp(world_size, create_and_train_model)

# the same t'ing
result = ranch(world_size, create_and_train_model, caller_rank=0, catchall=False, ctx=TorchDDPCtx())
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

