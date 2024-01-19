from functools import singledispatch
import numpy
import h5py
from biocutils import StringList, IntegerList, FloatList, BooleanList
from . import choose_missing_placeholder as ch
from . import _utils as ut


def write_StringList_to_hdf5(handle: h5py.Group, name: str, x: StringList) -> h5py.Dataset:
    has_none = any(y is None for y in x.as_list())
    if has_none:
        x, placeholder = ch.choose_missing_string_placeholder(x.as_list())

    dset = ut.save_fixed_length_strings(handle, name, x)
    if has_none:
        dset.attrs["missing-value-placeholder"] = placeholder
    return dset


def write_IntegerList_to_hdf5(handle: h5py.Group, name: str, x: IntegerList) -> h5py.Dataset:
    has_none = any(y is None for y in x.as_list())

    final_type = int
    if ut._is_integer_vector_within_limit(x):
        if has_none:
            x, mask = ut.list_to_numpy_with_mask(x, numpy.int32)
            x, placeholder = ch.choose_missing_integer_placeholder(x, mask, copy=False)
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
    else:
        dtype = "i4"

    dset = handle.create_dataset(name, data=x, dtype=dtype, compression="gzip", chunks=True)
    if has_none:
       dset.attrs.create("missing-value-placeholder", placeholder, dtype=dtype)
    return dset


def write_FloatList_to_hdf5(handle: h5py.Group, name: str, x: FloatList) -> h5py.Dataset:
    has_none = any(y is None for y in x)
    if has_none:
        x, mask = ut.list_to_numpy_with_mask(x, numpy.float64)
        x, placeholder = ch.choose_missing_float_placeholder(x, mask, copy=False)

    dset = handle.create_dataset(name, data=x, dtype="f8", compression="gzip", chunks=True)
    if has_none:
       dset.attrs.create("missing-value-placeholder", placeholder, dtype="f8")
    return dset


def write_BooleanList_to_hdf5(handle: h5py.Group, name: str, x: BooleanList) -> h5py.Dataset:
    has_none = any(y is None for y in x)
    if has_none:
        x, mask = ut.list_to_numpy_with_mask(x, x_dtype=numpy.uint8, mask_dtype=numpy.bool_)
        x, placeholder = ch.choose_missing_boolean_placeholder(x, mask, copy=False)

    dset = handle.create_dataset(name, data=x, dtype="i1", compression="gzip", chunks=True)
    if has_none:
       dset.attrs.create("missing-value-placeholder", placeholder, dtype="i1")
    return dset
