from typing import Any, Optional
from biocutils import Factor
import numpy
import h5py
import os


def read_string_factor(path: str, metadata: Optional[dict] = None, **kwargs) -> Factor:
    """
    Read a string factor from disk. In general, this function should not
    be called directly but instead via
    :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        path: Path to the directory containing the object.

        metadata: Metadata for the object. This is read from the `OBJECT` file if None.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A :py:class:`~biocutils.Factor.Factor` object.
    """

    with h5py.File(os.path.join(path, "contents.h5"), "r") as handle:
        ghandle = handle["string_factor"]

        chandle = ghandle["codes"]
        codes = chandle[:].astype(numpy.int32)
        if "missing-value-placeholder" in chandle.attrs:
            placeholder = chandle.attrs["missing-value-placeholder"]
            codes[codes == placeholder] = -1

        levels = [a.decode() for a in ghandle["levels"][:]]

        ordered = False
        if "ordered" in ghandle:
            ordered = ghandle["ordered"][()] != 0

        output = Factor(codes, levels, ordered = ordered)
        if "names" in ghandle:
            output.set_names([a.decode() for a in ghandle["names"][:]], in_place=True)

        return output
