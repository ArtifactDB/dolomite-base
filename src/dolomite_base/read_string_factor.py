from biocutils import Factor
import h5py
import os

from ._utils_factor import load_factor_from_hdf5
from . import _utils_string as strings


def read_string_factor(path: str, metadata: dict, **kwargs) -> Factor:
    """Read a string factor from disk. 
    
    In general, this function should not be called directly 
    but instead via
    :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        path: 
            Path to the directory containing the object.

        metadata: 
            Metadata for the object. 

        kwargs: 
            Further arguments, passed to nested objects.

    Returns:
        A :py:class:`~biocutils.Factor.Factor` object.
    """

    with h5py.File(os.path.join(path, "contents.h5"), "r") as handle:
        ghandle = handle["string_factor"]
        output = load_factor_from_hdf5(ghandle)
        if "names" in ghandle:
            output.set_names(strings.load_string_vector_from_hdf5(ghandle["names"]), in_place=True)
        return output
