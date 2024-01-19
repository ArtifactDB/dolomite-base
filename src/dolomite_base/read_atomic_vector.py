from typing import Union
from biocutils import StringList, IntegerList, FloatList, BooleanList
import numpy
import h5py
import os
import warnings

def read_atomic_vector(path: str, metadata: dict, **kwargs) -> Union[StringList, numpy.ndarray]:
    """
    Read an atomic vector from disk. In general, this function should not be
    called directly but instead via
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        metadata: Metadata for the object.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        An atomic vector. For string types, this will be a ``StringList``
        object, otherwise it will be a 1-dimensional NumPy array.
    """

    with h5py.File(os.path.join(path, "contents.h5"), "r") as handle:
        ghandle = handle["atomic_vector"]
        vectype = ghandle.attrs["type"]

        has_names = "names" in ghandle
        dhandle = ghandle["values"]
        has_none = "missing-value-placeholder" in dhandle.attrs

        if vectype == "string":
            output = StringList(a.decode() for a in dhandle[:])
            if has_none:
                placeholder = dhandle.attrs["missing-value-placeholder"]
                for i, x in enumerate(output):
                    if x == placeholder:
                        output[i] = None
        else:
            values = dhandle[:]
            if not has_none:
                output = values
            else:
                output = [None] * values.shape[0]
                placeholder = dhandle.attrs["missing-value-placeholder"]
                if numpy.isnan(placeholder):
                    for i, x in enumerate(values):
                        if not numpy.isnan(x):
                            output[i] = x
                else:
                    for i, x in enumerate(values):
                        if x != placeholder:
                            output[i] = x

            if vectype == "integer":
                output = IntegerList(output)
            elif vectype == "number":
                if "_python_original_type" in dhandle.attrs and dhandle.attrs["_python_original_type"] == "biocutils.IntegerList":
                    output = IntegerList(output)
                else:
                    output = FloatList(output)
            elif vectype == "boolean":
                output = BooleanList(output)

        if has_names:
            output.set_names([a.decode() for a in ghandle["names"][:]], in_place=True)
        return output
