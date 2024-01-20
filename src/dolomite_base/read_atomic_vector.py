from typing import Union
from biocutils import StringList, IntegerList, FloatList, BooleanList
import numpy
import h5py
import os
import warnings

from .load_vector_from_hdf5 import load_vector_from_hdf5

def read_atomic_vector(path: str, metadata: dict, atomic_vector_use_numeric_1darray: bool = False, **kwargs) -> Union[StringList, IntegerList, FloatList, BooleanList, numpy.ndarray]:
    """
    Read an atomic vector from its on-disk representation. In general, this
    function should not be called directly but instead via
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        metadata: Metadata for the object.

        atomic_vector_use_numeric_1darray: 
            Whether numeric vectors should be represented as 1-dimensional
            NumPy arrays. This is more memory-efficient than regular Python
            lists but discards the distinction between vectors and 1-D arrays.
            We set this to ``False`` by default to ensure that we can load
            names via :py:class:`~biocutils.NamedList.NamedList` subclasses.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        An atomic vector, represented as a
        :py:class:`~biocutils.StringList.StringList`,
        :py:class:`~biocutils.IntegerList.IntegerList`,
        :py:class:`~biocutils.FloatList.FloatList`,
        :py:class:`~biocutils.BooleanList.BooleanList`.
    """

    with h5py.File(os.path.join(path, "contents.h5"), "r") as handle:
        ghandle = handle["atomic_vector"]
        vectype = ghandle.attrs["type"]
        dhandle = ghandle["values"]
        values = load_vector_from_hdf5(dhandle, vectype, atomic_vector_use_numeric_1darray)

        if "names" in ghandle:
            if isinstance(values, NamedList):
                output.set_names([a.decode() for a in ghandle["names"][:]], in_place=True)
            else:
                warnings.warn("skipping names when reading atomic vectors as 1-dimensional NumPy arrays")
        return output
