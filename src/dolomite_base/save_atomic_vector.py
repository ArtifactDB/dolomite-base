from typing import Any
from biocutils import StringList, IntegerList, FloatList, BooleanList
import os
import h5py
import numpy

from .save_object import save_object, validate_saves
from . import _utils as ut

@save_object.register
@validate_saves
def save_atomic_vector_from_string_list(x: StringList, path: str, **kwargs): 
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
    x = x.as_list()
    has_none = any(y is None for y in x)
    if has_none:
        x, placeholder = ut.choose_missing_string_placeholder(x)

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "string"
        dset = ut.save_fixed_length_strings(ghandle, "values", x)

        if has_none:
           dset.attrs["missing-value-placeholder"] = placeholder
        if not nms is None:
            ut.save_fixed_length_strings(ghandle, "names", nms)

    return


@save_object.register
@validate_saves
def save_atomic_vector_from_integer_list(x: IntegerList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.IntegerList.IntegerList` objects
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
    x = x.as_list()
    has_none = any(y is None for y in x)

    final_type = int
    if ut._is_integer_vector_within_limit(x):
        if has_none:
            x, mask = ut.list_to_numpy_with_mask(x, numpy.int32)
            x, placeholder = ut.choose_missing_integer_placeholder(x, mask, copy=False)
            if numpy.issubdtype(x.dtype, numpy.floating):
                final_type = float
    else:
        final_type = float
        if has_none:
            x, mask = ut.list_to_numpy_with_mask(x, numpy.float64)
            placeholder = numpy.NaN
            x[mask] = placeholder

    if final_type == float:
        dtype = "f8"
        text_type = "number"
    else:
        dtype = "i4"
        text_type = "integer"

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = text_type
        dset = ghandle.create_dataset("values", data=x, dtype=dtype)

        if has_none:
           dset.attrs.create("missing-value-placeholder", placeholder, dtype=dtype)
        if not nms is None:
            ut.save_fixed_length_strings(ghandle, "names", nms)
        if final_type == float:
            dset.attrs.create("_python_original_type", "biocutils.IntegerList")

    return


@save_object.register
@validate_saves
def save_atomic_vector_from_float_list(x: FloatList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.FloatList.FloatList` objects
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
    x = x.as_list()
    has_none = any(y is None for y in x)

    if has_none:
        x, mask = ut.list_to_numpy_with_mask(x, numpy.float64)
        x, placeholder = ut.choose_missing_float_placeholder(x, mask, copy=False)

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "number"
        dset = ghandle.create_dataset("values", data=x, dtype="f8")

        if has_none:
           dset.attrs.create("missing-value-placeholder", placeholder, dtype="f8")
        if not nms is None:
            ut.save_fixed_length_strings(ghandle, "names", nms)

    return


@save_object.register
@validate_saves
def save_atomic_vector_from_boolean_list(x: BooleanList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.BooleanList.BooleanList` objects
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
    x = x.as_list()
    has_none = any(y is None for y in x)

    if has_none:
        x, mask = ut.list_to_numpy_with_mask(x, x_dtype=numpy.uint8, mask_dtype=numpy.bool_)
        x, placeholder = ut.choose_missing_boolean_placeholder(x, mask, copy=False)

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "boolean"
        dset = ghandle.create_dataset("values", data=x, dtype="i1")

        if has_none:
           dset.attrs.create("missing-value-placeholder", placeholder, dtype="i1")
        if not nms is None:
            ut.save_fixed_length_strings(ghandle, "names", nms)

    return
