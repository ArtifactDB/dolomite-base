from biocutils import Factor
import os
import h5py

from .save_object import save_object, validate_saves
from .save_object_file import save_object_file
from . import _utils_string as strings
from ._utils_factor import save_factor_to_hdf5


@save_object.register
@validate_saves
def save_string_factor(x: Factor, path: str, **kwargs):
    """Method for saving :py:class:`~biocutils.Factor.Factor` objects to their
    corresponding file representation, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to save the object.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)
    save_object_file(path, "string_factor", { "string_factor": { "version": "1.0" } })

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("string_factor")
        save_factor_to_hdf5(ghandle, x)
        nms = x.get_names()
        if nms is not None:
            strings.save_fixed_length_strings(ghandle, "names", nms.as_list())
