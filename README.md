<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/dolomite.svg?branch=main)](https://cirrus-ci.com/github/<USER>/dolomite)
[![ReadTheDocs](https://readthedocs.org/projects/dolomite/badge/?version=latest)](https://dolomite.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/dolomite/main.svg)](https://coveralls.io/r/<USER>/dolomite)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/dolomite.svg)](https://anaconda.org/conda-forge/dolomite)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/dolomite)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/dolomite-base.svg)](https://pypi.org/project/dolomite-base/)
[![Monthly Downloads](https://pepy.tech/badge/dolomite-base/month)](https://pepy.tech/project/dolomite-base)
![Unit tests](https://github.com/ArtifactDB/dolomite-base/actions/workflows/run-tests.yml/badge.svg)

# Save and load Bioconductor objects in Python

The **dolomite-base** package is the Python counterpart to the [**alabaster.base**](https://github.com/ArtifactDB/alabaster.base) R package
for language-agnostic reading and writing of Bioconductor objects (see the [**BiocPy**](https://github.com/BiocPy) project).
This is a more robust and portable alternative to the typical approach of pickling Python objects to save them to disk.

- By separating the on-disk representation from the in-memory object structure, we can more easily adapt to changes in class definitions.
  This improves robustness to Python environment updates.
- By using standard file formats like HDF5 and CSV, we ensure that the objects can be easily read from other languages like R and Javascript.
  This improves interoperability between application ecosystems.
- By breaking up complex Bioconductor objects into their components, we enable modular reads and writes to the backing store.
  We can easily read or update part of an object without having to consider the other parts.

The **dolomite-base** package defines the base generics to read and write the file structures along with the associated metadata.
Implementations of these methods for various Bioconductor classes can be found in the other **dolomite** packages like 
[**dolomite-ranges**](https://github.com/ArtifactDB/dolomite-ranges) and [**dolomite-se**](https://github.com/ArtifactDB/dolomite-se).

## Quick start

First, we'll install the **dolomite-base** package.
This package is available from [PyPI](https://pypi.org/project/dolomite-base) so we can use the standard installation process:

```r
pip install dolomite-base
```

The simplest example involves saving a [`BiocFrame`](https://github.com/BiocPy/BiocFrame) inside a staging directory.
Let's mock one up:

```r
import biocframe
df = biocframe.BiocFrame({ 
    "X": list(range(0, 10)),
    "Y": [ "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" ]
})
print(df)
## BiocFrame with 10 rows and 2 columns
##           X      Y
##     <range> <list>
## [0]       0      a
## [1]       1      b
## [2]       2      c
## [3]       3      d
## [4]       4      e
## [5]       5      f
## [6]       6      g
## [7]       7      h
## [8]       8      i
## [9]       9      j
```

Then we can save it to a directory with the `save_object()` function.

```r
import tempfile
import os
tmp = tempfile.mkdtemp()

import dolomite_base
path = os.path.join(tmp, "my_df")
dolomite_base.save_object(df, path)
```

We can copy the directory to another location, over a network, etc., and then easily load it back into a Python session via the `read_object()` function.
Note that the exact Python types for the `BiocFrame` columns may not be preserved by the round trip.

```r
out = dolomite_base.read_object(path)
print(out)
BiocFrame with 10 rows and 2 columns
##                    X            Y
##     <ndarray[int32]> <StringList>
## [0]                0            a
## [1]                1            b
## [2]                2            c
## [3]                3            d
## [4]                4            e
## [5]                5            f
## [6]                6            g
## [7]                7            h
## [8]                8            i
## [9]                9            j
```

Check out the [API reference](https://artifactdb.github.io/dolomite-base/api/modules.html) for more details.

## Supported classes

The saving/reading process can be applied to a range of data structures, provided the appropriate **alabaster** package is installed.

| Package | Object types | PyPI |
|-----|-----|----|----|
| [**dolomite-base**](https://github.com/ArtifactDB/dolomite-base) | [`BiocFrame`](https://github.com/BiocPy/BiocFrame), `list`, `dict`, [`NamedList`](https://github.com/BiocPy/BiocUtils) | [![](https://img.shields.io/pypi/v/dolomite-base.svg)](https://pypi.org/project/dolomite-base/) |
| [**dolomite-matrix**](https://github.com/ArtifactDB/dolomite-matrix) | `numpy.ndarray`, `scipy.sparse.spmatrix`, [`DelayedArray`](https://github.com/BiocPy/DelayedArray) | [![](https://img.shields.io/pypi/v/dolomite-matrix.svg)](https://pypi.org/project/dolomite-matrix/) |
| [**dolomite-ranges**](https://github.com/ArtifactDB/dolomite-ranges) | [`GenomicRanges`](https://github.com/BiocPy/GenomicRanges), `GenomicRangesList` | [![](https://img.shields.io/pypi/v/dolomite-ranges.svg)](https://pypi.org/project/dolomite-ranges/) |
| [**dolomite-se**](https://github.com/ArtifactDB/dolomite-se) | [`SummarizedExperiment`](https://github.com/BiocPy/SummarizedExperiment), `RangedSummarizedExperiment` | [![](https://img.shields.io/pypi/v/dolomite-se.svg)](https://pypi.org/project/dolomite-se/) |
| [**dolomite-sce**](https://github.com/ArtifactDB/dolomite-sce) | [`SingleCellExperiment`](https://github.com/BiocPy/SingleCellExperiment) | [![](https://img.shields.io/pypi/v/dolomite-sce.svg)](https://pypi.org/project/dolomite-sce/) |
| [**dolomite-mae**](https://github.com/ArtifactDB/dolomite-mae) | [`MultiAssayExperiment`](https://bioconductor.org/packages/MultiAssayExperiment) | [![](https://img.shields.io/pypi/v/dolomite-mae.svg)](https://pypi.org/project/dolomite-mae/) |

All packages are available from PyPI and can be installed with the usual `pip install` process.
Alternatively, to install all packages in one go, users can install the [**dolomite**](https://pypi.org/project/dolomite) umbrella package.

## Extensions and applications

Developers can _extend_ this framework to support more R/Bioconductor classes by creating their own **alabaster** package.
Check out the [extension guide](https://bioconductor.org/packages/release/bioc/vignettes/alabaster.base/inst/doc/extensions.html) for more details.

Developers can also _customize_ this framework for specific applications, most typically to save bespoke metadata in the JSON file.
The JSON file can then be indexed by systems like MongoDB and Elasticsearch to provide search capabilities.
Check out the [applications guide](https://bioconductor.org/packages/release/bioc/vignettes/alabaster.base/inst/doc/applications.html) for more details.

## Links

The [**BiocObjectSchemas**](https://github.com/ArtifactDB/BiocObjectSchemas) repository contains schema definitions for many Bioconductor objects.

For use in an R installation, all schemas are packaged in the [**alabaster.schemas**](https://github.com/ArtifactDB/alabaster.schemas) R package.

A [Docker image](https://github.com/ArtifactDB/alabaster-docker/pkgs/container/alabaster-docker%2Fbuilder) is available, containing several pre-installed **alabaster** packages.

