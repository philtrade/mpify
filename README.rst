Introduction
============

**`mpify`** is a thin library to run function on multiple processes, and to help training model
in PyTorch's Distributed Data Parallel mode in a spontaneous manner.

The main features are:

  * Parallel but blocking execution, call returns when the function has finished in all processes.
  * Worker pool is non-persistent.  Subprocesses are forked spontaneously and terminates upon function exit.
  * Each worker can initialize its own GPU context, suitable distributed application on a multi-GPU node.
  * Functions/objects, *including those defined in a Jupyter notebook*, can be serialized to subprocesses by name.
  * Import statements can be executed in the spontaneously created worker processes, before the function body is run.
  * Results from any or all of the workers can be gathered.

Although `mpify` works standalone Python app too, it is designed with single-node,
multi-GPUs usage in mind.  For asynchronous, distributed workloads on multiple nodes (cluster),
please use `ipyparallel`, `dask`, or `ray`, as they take care of scheduling, fault tolerance etc..

Examples
--------

* Porting a training loop in Fastai2's notebook train in distributed data parallel mode:

Original:

|NBexOriginal|

.. |NBexOriginal| image:: https://github.com/philtrade/mpify/blob/master/images/01_intro_train_cnn_orig.png

`mpify`-ed:

|NBexMpified|

.. |NBexMpified| image:: https://github.com/philtrade/mpify/blob/master/images/01_intro_train_cnn_mpify.png

More notebook examples can be found in the [examples/](/examples) directory.


Installation
------------

::
      python3 -m pip install git+https://github.com/philtrade/mpify 

Documentation
-------------
The complete `API documentation <https://mpify.readthedocs.io/en/latest/mpify.html>`_.

References
----------

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate.  It is  a thinner lighter follow-up to my previous attempt at the problem: [`Ddip`](https://github.com/philtrade/Ddip), which uses `ipyparallel`.

* Why use `multiprocess <https://github.com/uqfoundation/multiprocess>`_ instead of `multiprocessing` and `torch.multiprocessing`: <https://hpc-carpentry.github.io/hpc-python/06-parallel/>
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

