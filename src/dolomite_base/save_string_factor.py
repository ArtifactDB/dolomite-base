from typing import Any
from biocutils import Factor
import os
import h5py
import numpy

from .save_object import save_object, validate_saves
from . import _utils as ut

@save_object.register
@validate_saves
def save_string_factor(x: Factor, path: str, **kwargs):
    """Method for saving :py:class:`~biocutils.Factor.Factor` objects to their
    corresponding file representation, see
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
        handle.write('{ "type": "string_factor", "string_factor": { "version": "1.0" } }')

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("string_factor")
        ut.save_fixed_length_strings(ghandle, "levels", x.get_levels())

        codes = x.get_codes()
        is_missing = codes == -1
        has_missing = is_missing.any()
        nlevels = len(x.get_levels())
        if has_missing:
            codes = codes.astype(numpy.uint32, copy=True)
            codes[is_missing] = nlevels

        dhandle = ghandle.create_dataset("codes", data=codes, dtype="u4", compression="gzip", chunks=True)
        if has_missing:
            dhandle.attrs.create("missing-value-placeholder", data=nlevels, dtype="u4")

        nms = x.get_names()
        if not nms is None:
            ut.save_fixed_length_strings(ghandle, "names", nms.as_list())

        if x.get_ordered():
            ghandle.create_dataset("ordered", data=1, dtype="i8")
