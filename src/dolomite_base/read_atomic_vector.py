from typing import Any, Union, Optional
from biocutils import StringList
import numpy
import h5py
import os
import warnings

def read_atomic_vector(path: str, metadata: Optional[dict] = None, **kwargs) -> Union[StringList, numpy.ndarray]:
    """
    Read an atomic vector from disk. In general, this function should not be
    called directly but instead via
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        metadata: Metadata for the object. This is read from the `OBJECT` file if None.

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

            if has_names:
                output.set_names([a.decode() for a in ghandle["names"][:]], in_place=True)
            return output

        if has_names:
            warnings.warn("skipping names when reading a numeric 'atomic_vector'")

        output = dhandle[:]
        if has_none:
            placeholder = dhandle.attrs["missing-value-placeholder"]
            if numpy.isnan(placeholder):
                mask = numpy.isnan(output)
            else:
                mask = (output == placeholder)

        if vectype == "boolean":
            output = output.astype(numpy.bool_)
        elif vectype == "float":
            if not numpy.issubdtype(output.dtype, numpy.floating):
                output = output.astype(numpy.double)

        if has_none:
            return numpy.ma.MaskedArray(output, mask=mask)
        else:
            return output
