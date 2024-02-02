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

```sh
pip install dolomite-base
```

The simplest example involves saving a [`BiocFrame`](https://github.com/BiocPy/BiocFrame) inside a staging directory.
Let's mock one up:

```python
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

We save our `BiocFrame` to a user-specified directory with the `save_object()` function.
This function saves its input object to file according to the relevant [specification](https://github.com/ArtifactDB/takane).

```python
import tempfile
import os
tmp = tempfile.mkdtemp()

import dolomite_base
path = os.path.join(tmp, "my_df")
dolomite_base.save_object(df, path)

os.listdir(path)
## ['basic_columns.h5', 'OBJECT']
```

We load the contents of the directory back into a Python session by using the `read_object()` function.
Note that the exact Python types for the `BiocFrame` columns may not be preserved by the round trip,
though the contents of the columns will be unchanged.

```python
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

The saving/reading process can be applied to a range of [**BiocPy**](https://github.com/BiocPy) data structures,
provided the appropriate **dolomite** package is installed.
Each package implements a saving and reading function for its associated classes,
which are automatically used from **dolomite-base**'s `save_object()` and `read_object()` functions, respectively.
(That is, there is no need to explicitly `import` a package when calling `save_object()` or `read_object()` for its classes.)

| Package | Object types | PyPI |
|-----|-----|----|
| [**dolomite-base**](https://github.com/ArtifactDB/dolomite-base) | [`BiocFrame`](https://github.com/BiocPy/BiocFrame), `list`, `dict`, [`NamedList`](https://github.com/BiocPy/BiocUtils) | [![](https://img.shields.io/pypi/v/dolomite-base.svg)](https://pypi.org/project/dolomite-base/) |
| [**dolomite-matrix**](https://github.com/ArtifactDB/dolomite-matrix) | `numpy.ndarray`, `scipy.sparse.spmatrix`, [`DelayedArray`](https://github.com/BiocPy/DelayedArray) | [![](https://img.shields.io/pypi/v/dolomite-matrix.svg)](https://pypi.org/project/dolomite-matrix/) |
| [**dolomite-ranges**](https://github.com/ArtifactDB/dolomite-ranges) | [`GenomicRanges`](https://github.com/BiocPy/GenomicRanges), `GenomicRangesList` | [![](https://img.shields.io/pypi/v/dolomite-ranges.svg)](https://pypi.org/project/dolomite-ranges/) |
| [**dolomite-se**](https://github.com/ArtifactDB/dolomite-se) | [`SummarizedExperiment`](https://github.com/BiocPy/SummarizedExperiment), `RangedSummarizedExperiment` | [![](https://img.shields.io/pypi/v/dolomite-se.svg)](https://pypi.org/project/dolomite-se/) |
| [**dolomite-sce**](https://github.com/ArtifactDB/dolomite-sce) | [`SingleCellExperiment`](https://github.com/BiocPy/SingleCellExperiment) | [![](https://img.shields.io/pypi/v/dolomite-sce.svg)](https://pypi.org/project/dolomite-sce/) |
| [**dolomite-mae**](https://github.com/ArtifactDB/dolomite-mae) | [`MultiAssayExperiment`](https://bioconductor.org/packages/MultiAssayExperiment) | [![](https://img.shields.io/pypi/v/dolomite-mae.svg)](https://pypi.org/project/dolomite-mae/) |

All of the listed packages are available from PyPI and can be installed with the usual `pip install` procedure.
Alternatively, to install all packages in one go, users can install the [**dolomite**](https://pypi.org/project/dolomite) umbrella package.

## Operating on directories

Users can move freely rename or relocate directories and the `read_object()` function will still work.
For example, we can easily copy the entire directory to a new file system and everything will still be correctly referenced within the directory.
The simplest way to share objects is to just `zip` or `tar` the staging directory for _ad hoc_ distribution,
though more serious applications will use storage systems like AWS S3 for easier distribution.

```python
# Mocking up an object:
import biocframe
df = biocframe.BiocFrame({
    "X": list(range(0, 10)),
    "Y": [ "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" ]
})

# Saving to one location:
import tempfile
import os
import dolomite_base
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "my_df")
dolomite_base.save_object(df, path)

# Reading from another location:
alt_path = os.path.join(tmp, "foobar")
os.rename(path, alt_path)
alt_out = dolomite_base.read_object(alt_path)
```

That said, it is unwise to manipulate the files inside the directory created by `save_object()`.
Reading functions will usually depend on specific file names or subdirectory structures within the directory, and fiddling with them may cause unexpected results.
Advanced users can exploit this by loading components from subdirectories if the full object is not required:

```python
# Creating a nested DF:
nested = biocframe.BiocFrame({ "A": df })
nest_path = os.path.join(tmp, "nesting")
dolomite_base.save_object(nested, nest_path)

# Now reading in the nested DF:
redf = dolomite_base.read_object(os.path.join(nest_path, "other_columns", "0"))
```

## Validating files

Each Bioconductor class's on-disk representation is determined by the associated [**takane** specification](https://github.com/ArtifactDB/takane).
For example, `save_object()` will save a `BiocFrame` according to the [`data_frame` specification](https://github.com/ArtifactDB/takane/blob/gh-pages/docs/specifications/data_frame/1.0.md). 
More complex objects may be represented by multiple files, possibly including subdirectories with "child" objects.

Each call to `save_object()` will automatically enforce the relevant specification by validating the directory contents with **dolomite-base**'s `validate_object()` function.
Successful validation provides some guarantees on the file structure within the directory, allowing developers to reliably implement readers in other frameworks.
Conversely, the [**alabaster**](https://github.com/ArtifactDB/alabaster.base) suite applies the same validators on directories generated within an R session,
which ensures that **dolomite-base** is able to read those objects into a Python environment.

Users can also call `validate_object()` themselves, if they have modified the directory after calling `save_object()` and they want to check that the contents are still valid:

```python
# Mocking up an object:
import biocframe
df = biocframe.BiocFrame({
    "X": list(range(0, 10)),
    "Y": [ "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" ]
})

# Saving to one location:
import tempfile
import os
import dolomite_base
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "my_df")
dolomite_base.save_object(df, path)

# So far so good...
dolomite_base.validate_object(path)

# Deleting the file to make it invalid:
os.remove(os.path.join(path, "basic_columns.h5"))
dolomite_base.validate_object(path)
## Traceback (most recent call last):
## etc...
```

## Extending to new classes

The **dolomite** framework is easily extended to new classes by:

1. Writing a method for `save_object()`.
   This should accept an instance of the object and a path to a directory, and save the contents of the object inside the directory.
   It should also produce an `OBJECT` file that specifies the type of the object, e.g., `data_frame`, `hdf5_sparse_matrix`.
2. Writing a function for `read_object()` and registering it in the `read_object_registry`.
   This should accept a path to a directory and read its contents to reconstruct the object.
   The registered type should be the same as that used in the `OBJECT` file.
3. Writing a function for `validate_object()` and registering it in the `validate_object_registry`.
   This should accept a path to a directory and read its contents to determine if it is a valid on-disk representation.
   The registered type should be the same as that used in the `OBJECT` file.
   - (optional) Devleopers can alternatively formalize the on-disk representation by adding a specification to the [**takane**](https://github.com/ArtifactDB/takane) repository.
     This aims to provide C++-based validators for each representation, allowing us to enforce consistency across multiple languages (e.g., R).
     Any **takane** validator is automatically used by `validate_object()` so no registration is required.

To illustrate, let's extend **dolomite** to a new custom class:

```python
class Coffee:
    def __init__(self, beans: str, milk: bool):
        self.beans = beans
        self.milk = milk
```

First we implement the saving method.
Note that we add a `@validate_saves` decorator to instruct `save_object()` to automatically run `validate_object()` on the directory by the `Coffee` method.
This confirms that the output is valid according to our (yet to be added) validator method.

```python
import dolomite_base
import os
import json

@dolomite_base.save_object.register
@dolomite_base.validate_saves
def save_object_for_Coffee(x: Coffee, path: str, **kwargs):
    os.mkdir(path)
    with open(os.path.join(path, "bean_type"), "w") as handle:
        handle.write(x.beans)
    with open(os.path.join(path, "has_milk"), "w") as handle:
        handle.write("true" if x.milk else "false")
    with open(os.path.join(path, "OBJECT"), "w") as handle:
        json.dump({ "type": "coffee", "coffee": { "version": "0.1" } }, handle)
```

Then we implement and register the reading method:

```python
from typing import Dict

def read_Coffee(path: str, metadata: Dict, **kwargs) -> Coffee:
    metadata["coffee"]["version"] # possibly do something different based on version
    with open(os.path.join(path, "bean_type"), "r") as handle:
        beans = handle.read()
    with open(os.path.join(path, "has_milk"), "r") as handle:
        milk = (handle.read() == "true")
    return Coffee(beans, milk)

dolomite_base.read_object_registry["coffee"] = read_Coffee
```

And finally, the validation method:

```python
def validate_Coffee(path: str, metadata: Dict):
    metadata["coffee"]["version"] # possibly do something different based on version
    with open(os.path.join(path, "bean_type"), "r") as handle:
        beans = handle.read()
        if not beans in [ "arabica", "robusta", "excelsa", "liberica" ]:
            raise ValueError("wrong bean type '" + beans + "'")
    with open(os.path.join(path, "has_milk"), "r") as handle:
        milk = handle.read()
        if not milk in [ "true", "false" ]:
            raise ValueError("invalid milk '" + milk + "'")

dolomite_base.validate_object_registry["coffee"] = validate_Coffee
```

Let's run them and see how it works:

```python
cup = Coffee("arabica", milk=False)

import tempfile
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "stuff")
dolomite_base.save_object(cup, path)

cup2 = dolomite_base.read_object(path)
print(cup2.beans)
## arabica
```

For more complex objects that are composed of multiple smaller "child" objects, developers should consider saving each of their children in subdirectories of `path`.
This can be achieved by calling `alt_save_object()` and `alt_read_object()` in the saving and loading functions, respectively.
(We use the `alt_*` versions of these functions to respect application overrides, see below.)

## Creating applications

Developers can also create applications that customize the machinery of the _dolomite_ framework for specific needs.
In most cases, this involves storing more metadata to describe the object in more detail.
For example, we might want to remember the identity of the author for each object.
This is achieved by creating an application-specific saving generic with the same signature as `save_object()`:

```python
from functools import singledispatch
from typing import Any, Dict, Optional
import dolomite_base
import json
import os
import getpass
import biocframe

def dump_extra_metadata(path: str, extra: Dict):
    user_id = getpass.getuser()
    # File names with leading underscores are reserved for application-specific
    # use, so they won't clash with anything produced by save_object().
    metapath = os.path.join(path, "_metadata.json")
    with open(metapath, "w") as handle:
        json.dump({ **extra, "author": user_id }, handle)

@singledispatch
def app_save_object(x: Any, path: str, **kwargs):
    dolomite_base.save_object(x, path, **kwargs) # does the real work
    dump_extra_metadata(path, {}) # adding some application-specific metadata

@app_save_object.register
def app_save_object_for_BiocFrame(x: biocframe.BiocFrame, path: str, **kwargs):
    dolomite_base.save_object(x, path, **kwargs) # does the real work
    # We can also override specific methods to add object+application-specific metadata:
    dump_extra_metadata(path, { "columns": x.get_column_names().as_list() })
```

In general, applications should avoid modifying the files created by the `dolomite_base.save_object()` call, to avoid violating any **takane** format specifications
(unless the application maintainer really knows what they're doing).
Applications are free to write to any path starting with an underscore as this will not be used by any specification.

Once a generic is defined, applications should call `alt_save_object_function()` to instruct `alt_save_object()` to use it instead of `dolomite_base.save_object()`.
This ensures that the customizations are applied to all child objects, such as the nested `BiocFrame` below.

```python
# Create a friendly user-visible function to perform the generic override; this
# is reversed on function exit to avoid interfering with other applications.
def save_for_application(x, path: str, **kwargs):
    old = dolomite_base.alt_save_object_function(app_save_object)
    try:
        dolomite_base.alt_save_object(x, path, **kwargs)
    finally:
        dolomite_base.alt_save_object_function(old)

# Saving our nested BiocFrames with our overrides active.
import biocframe
df = biocframe.BiocFrame({
    "A": [1, 2, 3, 4],
    "B": biocframe.BiocFrame({
        "C": ["a", "b", "c", "d"]
    })
})

import tempfile
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "foobar")
save_for_application(df, path)

# Both the parent and child BiocFrames have new metadata.
with open(os.path.join(path, "_metadata.json"), "r") as handle:
    print(handle.read())
## {"columns": ["A", "B"], "author": "aaron"}

with open(os.path.join(path, "other_columns", "1", "_metadata.json"), "r") as handle:
    print(handle.read())
## {"columns": ["C"], "author": "aaron"}
```

The reading function can be similarly overridden by setting `alt_read_object_function()` to instruct all `alt_read_object()` calls to use the override.
This allows applications to, e.g., do something with the metadata that we just added.

```python
def app_read_object(path: str, metadata: Optional[Dict] = None, **kwargs):
    if metadata is None:
        with open(os.path.join(path, "OBJECT"), "r") as handle:
            metadata = json.load(handle)

    # Print custom message based on the type and application-specific metadata.
    with open(os.path.join(path, "_metadata.json"), "r") as handle:
        appmeta = json.load(handle)
        print("I am a " + metadata["type"] + " created by " + appmeta["author"])
        if metadata["type"] == "data_frame":
            print("I have the following columns: " + ", ".join(appmeta["columns"]))

    return dolomite_base.read_object(path, metadata=metadata, **kwargs)

# Creating a user-friendly function to set the override before the read operation.
def read_for_application(path: str, metadata: Optional[Dict] = None, **kwargs):
    old = dolomite_base.alt_read_object_function(app_read_object)
    try:
        return dolomite_base.alt_read_object(path, metadata=metadata, **kwargs)
    finally:
        dolomite_base.alt_read_object_function(old)

# This diverts to the override with printing of custom messages.
read_for_application(path)
## I am a data_frame created by aaron
## I have the following columns: A, B
## I am a data_frame created by aaron
## I have the following columns: C
```

By overriding the saving and reading process for one or more classes, each application can customize the behavior of the **dolomite** framework to their own needs.
