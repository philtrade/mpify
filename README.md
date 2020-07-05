## Overview 

**`mpify`** is a simple API to launch a "target function" parallelly to multiple *ranked* processes via a single blocking call. With the following features:

   * sub-processes will exit upon function return, thus **freeing up resources (e.g. Multiple GPU contexts)**,
   * **caller process may participate** as a ranked worker and receive the return object from taret function,
   * **locally defined functions and objects in Jupyter can be passed to the target function in subprocesses**,
   * provide mechanism to **support `from X import *` within the target function**,
   * user can **use context manager to manage resources around the execution** of the target function,

`mpify` hopes to make multiprocess experiments in Jupyter notebook a little easier (although it works in batch script as well.):

E.g. `fastai v2` users can train model using multiple GPUs in distributed data parallel (DDP), all inside Jupyter, without any IPython `%%` magic, or custom loop to handle asynchronous multiprocess results, or struggling to pass locally defined function to the training loop and get back trained Learner upon completion..... using the convenient wrapper `in_torchddp()`:

### To adapt [the `fastai v2` notebook on training `imagenette`](https://github.com/fastai/course-v4/blob/master/nbs/07_sizing_and_tta.ipynb) to train in DDP on 3 GPUs, in Jupyter.
From:

<img src="/images/imagenette_07_orig.png" height="270">

To this:

<img src="/images/imagenette_07_mpified.png" height="340">


## API and Usage Guide

> <b>import_star</b>(<i>module_list:[str]</i>)

  Use this helper `mpify.import_star(['X',...])` in the target function to perform `from X import *` --- *a usage banned by Python*.

------

> <b>ranch</b>(<i>nprocs:int, fn:Callable, *args, caller_rank:int=0, catchall:bool=True, host_rank:int=0, ctx=None, **kwargs</i>)

  Launch a group of ranked process (0..`nprocs-1` inclusive) to execute target function `fn(*args, **kwargs)` in parallel.

  Returns: a list of return values from each process, or that from the caller process.  See `catchall` and `caller_rank` below. 
  
  A few generic distributed attributes will be available to `fn()`. *Local rank*, *global rank*, *world size* are stored in `os.environ['LOCAL_RANK, RANK, and WORLD_SIZE']` --- note that they are strings, not integers, as with all things in `os.environ`.  Their definitions follow PyTorch's [distributed data parallel convention](https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940).

<i>`nprocs: int`</i> -- Number of worker processes

<i>`fn, *args, and **kwargs`</i> -- target function and its calling parameters.

<i>`caller_rank: int`</i> -0 The worker process rank of the caller, if not None.
 By default the caller process will participate as rank-0.  If the caller process does not want to participate, set `caller_rank` to None.

<i>`catchall: bool`</i>: If True, `ranch()` returns a list of return values from all processes.  If `False`, and if `caller_rank` is defined, return the value of the `caller_rank` execution of the target function.  Otherwise `ranch()` will return `None`.

------

> <b>in_torchddp</b>(<i>nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, **kwargs</i>):

  Launch `fn` to `nprocs` processes, and caller process participates as `rank-0`.  Set up the torch distributed data parallel process group environment -- initialize, execute `fn(*args, **kwargs)`, tearn down, then return only the `rank-0` execution result.


> <b>TorchDDPCtx</b>

  A sample context manager that manages the Torch DDP setup/teardown around the target function execution.


## Usage 

The rule of thumb rule is: the target function must be self-contained, well encapsulated because it'll be executed on a fresh Python interpreter process.  Thus:

> <i> Spell all local objects it needs (e.g. `foo()` or `objA`) in the parameters.  Import all modules its needs at the top.  To handle `from X import *`, use `mpify.import_star(['X'])` </i>
>
>  <i>Process rank is available from `os.environ['RANK']`.</i>

Example: If originally things were defined like this:
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
[5] def target_fn(arg, ....foo, objA, ... **kwargs):
      from mpify import import_star
      import_star(['utils', 'fancy'])  # *-imports here
      import numpy as np               # then other imports
      import torch
      import_star(['torch.distributed'])
      
      x = np.array([objA])
      foo(x)
      ...
      # can use os.environ['RANK'] to see its process rank.
```

To launch it to 5 ranked processes, and `r` below will receive a list of 5 return values.  `r[i]` is from `rank-i` execution, 
```python
[6] import mpify
    r = mpify.ranch(5, target_fn, arg,... foo, objA, ...other kwargs)
```


### Resources mangement using custom context manager `mpify.ranch(... ctx=Blah ...)`:

`mpify.ranch(ctx=YourCtxMgr)` lets user wrap around the function execution:

<i>spawn -> [`ctx.__enter__()`]-> run function -> [`ctx.__exit__()`] -> terminate</i>


**As an example**, `mpify` provides `TorchDDPCtx` to setup/tear-down of PyTorch's distributed data parallel: 

- `TorchDDPCtx` does the "bold stuff":
  
  spawn -> [**set target GPU, and initialize torch DDP group**]-> run function -> [**cleanup DDP**] -> terminate.

`TorchDDPCtx` can be customized to use different address and port, but a default `TorchDDPCtx()` is used by `mpify`'s  convenience routine `mpify.in_torchddp()`.

The following two calls are equivalent:

```python
    from mpify import in_torchddp, ranch, TorchDDPCtx
    
    # The same t'ing

    result = in_torchddp(world_size, create_and_train_model, *args, **kwargs)

    # OR 
    result = ranch(world_size, create_and_train_model, *args, caller_rank=0, catchall=False, ctx=TorchDDPCtx(), **kwargs)
 ```

More notebook examples will come along in the future.


## References:

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate. <include link to blog when available> It is also a thinner/lighter/more flexible follow-up to my other repo [`Ddip`](https://github.com/philtrade/Ddip), which uses `ipyparallel`.

* The [`multiprocess` libray](https://github.com/uqfoundation/multiprocess), an actively supported alternative to Python's default `multiprocessing` and `torch.multiprocessing`. 
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

