from typing import Any
from biocutils import StringList
import os
import h5py

from .save_object import save_object, validate_saves
from . import _utils as ut

@save_object.register
@validate_saves
def save_string_list(x: StringList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.StringList.StringList` objects
    to their corresponding file representation, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to save the object.

        kwargs: Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)
    with open(os.path.join(path, "OBJECT"), 'w', encoding="utf-8") as handle:
        handle.write('{ "type": "atomic_vector", "atomic_vector": { "version": "1.0" } }')

    nms = x.get_names()
    has_none = any(y is None for y in x)
    if has_none:
        x, placeholder = ut._choose_missing_string_placeholder(x)

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "string"
        dset = ut._save_fixed_length_strings(ghandle, "values", x)

        if has_none:
           dset.attrs["missing-value-placeholder"] = placeholder
        if not nms is None:
            ut._save_fixed_length_strings(ghandle, "names", nms)

    return
