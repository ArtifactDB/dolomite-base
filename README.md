<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/dolomite.svg?branch=main)](https://cirrus-ci.com/github/<USER>/dolomite)
[![ReadTheDocs](https://readthedocs.org/projects/dolomite/badge/?version=latest)](https://dolomite.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/dolomite/main.svg)](https://coveralls.io/r/<USER>/dolomite)
[![PyPI-Server](https://img.shields.io/pypi/v/dolomite.svg)](https://pypi.org/project/dolomite/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/dolomite.svg)](https://anaconda.org/conda-forge/dolomite)
[![Monthly Downloads](https://pepy.tech/badge/dolomite/month)](https://pepy.tech/project/dolomite)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/dolomite)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# dolomite

> Add a short description here!

A longer description of your project goes here...


## Developer notes

Obtain the headers:

```shell
(cd extern && ./fetch)
```

Build the shared object file:

```shell
CC="ccache clang++" python setup.py build_ext --inplace
```

For installation:

```shell
CC="ccache clang++" python setup.py install --user
```

For quick testing:

```shell
pytest
```

For more complex testing:

```shell
python setup.py build_ext --inplace && tox
```

To rebuild the **ctypes** bindings with [**cpptypes**](https://github.com/BiocPy/ctypes-wrapper):

```shell
cpptypes src/dolomite/lib \
    --py src/dolomite/_cpphelpers.py \
    --cpp src/dolomite/lib/bindings.cpp \
    --dll _core
```

