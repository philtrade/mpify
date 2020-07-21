#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mpify-philtrade",
    version="0.1.0",
    author='Phillip K.S. Chu',
    author_email='philtrade@winphil.net',
    url='https://github.com/philtrade/mpify',
    description="Adapting/launching function to run on multiple processes in iPython/Jupyter notebook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['mpify'],
    install_requires=['multiprocess', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: Ubuntu",
    ],
    python_requires='>=3.6',
    license="Apache License 2.0",
    zip_safe=False,
    keywords=['multiprocessing', 'distributed data parallel', 'multi GPU', 'Python', 'Parallel Processing', 'Distributed Computing', 'jupyter', 'IPython',],
)
