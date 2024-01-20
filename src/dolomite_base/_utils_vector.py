from typing import Sequence, Union
import numpy
import h5py
from biocutils import StringList, IntegerList, FloatList, BooleanList

from . import choose_missing_placeholder as ch
from . import _utils_misc as misc
from . import _utils_string as strings


def _list_to_numpy_with_mask(x: Sequence, x_dtype, mask_dtype = numpy.uint8) -> numpy.ndarray:
    mask = numpy.ndarray(len(x), dtype=mask_dtype)
    arr = numpy.ndarray(len(x), dtype=x_dtype)
    for i, y in enumerate(x):
        if y is None:
            arr[i] = 0
            mask[i] = 1
        else:
            arr[i] = y
            mask[i] = 0
    return arr, mask


def write_string_list_to_hdf5(handle: h5py.Group, name: str, x: list) -> h5py.Dataset:
    has_none = any(y is None for y in x)
    if has_none:
        x, placeholder = ch.choose_missing_string_placeholder(x)

    dset = strings.save_fixed_length_strings(handle, name, x)
    if has_none:
        dset.attrs["missing-value-placeholder"] = placeholder
    return dset


def write_integer_list_to_hdf5(handle: h5py.Group, name: str, x: list) -> h5py.Dataset:
    has_none = any(y is None for y in x)

    final_type = int
    if misc.sequence_exceeds_int32(x):
        final_type = float
        if has_none:
            x, mask = _list_to_numpy_with_mask(x, numpy.float64)
            placeholder = numpy.NaN
            x[mask] = placeholder
    else:
        if has_none:
            x, mask = _list_to_numpy_with_mask(x, numpy.int32)
            x, placeholder = ch.choose_missing_integer_placeholder(x, mask, copy=False)
            if numpy.issubdtype(x.dtype, numpy.floating):
                final_type = float

    if final_type == float:
        dtype = "f8"
    else:
        dtype = "i4"

    dset = handle.create_dataset(name, data=x, dtype=dtype, compression="gzip", chunks=True)
    if has_none:
       dset.attrs.create("missing-value-placeholder", placeholder, dtype=dtype)
    return dset


def write_float_list_to_hdf5(handle: h5py.Group, name: str, x: list) -> h5py.Dataset:
    has_none = any(y is None for y in x)
    if has_none:
        x, mask = _list_to_numpy_with_mask(x, numpy.float64)
        x, placeholder = ch.choose_missing_float_placeholder(x, mask, copy=False)

    dset = handle.create_dataset(name, data=x, dtype="f8", compression="gzip", chunks=True)
    if has_none:
       dset.attrs.create("missing-value-placeholder", placeholder, dtype="f8")
    return dset


def write_boolean_list_to_hdf5(handle: h5py.Group, name: str, x: list) -> h5py.Dataset:
    has_none = any(y is None for y in x)
    if has_none:
        x, mask = _list_to_numpy_with_mask(x, x_dtype=numpy.uint8, mask_dtype=numpy.bool_)
        x, placeholder = ch.choose_missing_boolean_placeholder(x, mask, copy=False)

    dset = handle.create_dataset(name, data=x, dtype="i1", compression="gzip", chunks=True)
    if has_none:
       dset.attrs.create("missing-value-placeholder", placeholder, dtype="i1")
    return dset


def write_ndarray_to_hdf5(handle: h5py.Group, name: str, x: numpy.ndarray) -> h5py.Dataset:
    if numpy.issubdtype(x.dtype, numpy.floating):
        dset = handle.create_dataset(name, data=x, dtype="f8", compression="gzip", chunks=True)
    elif x.dtype == numpy.bool_:
        dset = handle.create_dataset(name, data=x, dtype="i1", compression="gzip", chunks=True)
    else:
        if misc.sequence_exceeds_int32(x, check_none=False):
            dset = handle.create_dataset(name, data=x, dtype="f8", compression="gzip", chunks=True)
        else:
            dset = handle.create_dataset(name, data=x, dtype="i4", compression="gzip", chunks=True)
    return dset


def write_MaskedArray_to_hdf5(handle: h5py.Group, name: str, x: numpy.ma.MaskedArray) -> h5py.Dataset:
    mask = x.mask
    if not any(mask):
        return write_ndarray_to_hdf5(handle, name, x.data)

    if numpy.issubdtype(x.dtype, numpy.floating):
        x, placeholder = ch.choose_missing_float_placeholder(x.data, mask)
        dset = handle.create_dataset(name, data=x, dtype="f8", compression="gzip", chunks=True)
        dset.attrs.create("missing-value-placeholder", placeholder, dtype="f8")
    elif x.dtype == numpy.bool_:
        x, placeholder = ch.choose_missing_boolean_placeholder(x.data, mask, copy=False)
        dset = handle.create_dataset(name, data=x, dtype="i1", compression="gzip", chunks=True)
        dset.attrs.create("missing-value-placeholder", placeholder, dtype="i1")
    else:
        final_type = int
        if misc.sequence_exceeds_int32(x):
            final_type = float
            placeholder = numpy.NaN
            x = x.data.astype(numpy.float64)
            x[mask] = placeholder
        else:
            x = x.data.astype(numpy.int32)
            x, placeholder = ch.choose_missing_integer_placeholder(x, mask, copy=False)
            if numpy.issubdtype(x.dtype, numpy.floating):
                final_type = float

        if final_type == float:
            dtype = "f8"
        else:
            dtype = "i4"
        dset = handle.create_dataset(name, data=x, dtype="f8", compression="gzip", chunks=True)
        dset.attrs.create("missing-value-placeholder", placeholder, dtype=dtype)

    return dset


def load_vector_from_hdf5(handle: h5py.Dataset, curtype: str, convert_to_1darray: bool) -> Union[StringList, IntegerList, FloatList, BooleanList, numpy.ndarray]:
    if curtype == "string":
        values = StringList(strings.load_string_vector_from_hdf5(handle))
        if "missing-value-placeholder" in handle.attrs:
            placeholder = strings.load_scalar_string_attribute_from_hdf5(handle, "missing-value-placeholder")
            for j, y in enumerate(values):
                if y == placeholder:
                    values[j] = None
        return values

    values = handle[:]
    if "missing-value-placeholder" in handle.attrs:
        placeholder = handle.attrs["missing-value-placeholder"]
        if numpy.isnan(placeholder):
            mask = numpy.isnan(values)
        else:
            mask = (values == placeholder)

        if convert_to_1darray:
            return numpy.ma.MaskedArray(_coerce_numpy_type(values, curtype), mask=mask)
        else:
            output = []
            for i, y in enumerate(values):
                if mask[i]:
                    output.append(None)
                else:
                    output.append(y)
            return _choose_NamedList_subclass(output, curtype)

    if convert_to_1darray:
        return _coerce_numpy_type(values, curtype)
    else:
        return _choose_NamedList_subclass(values, curtype)


def _coerce_numpy_type(values: numpy.ndarray, curtype: str) -> numpy.ndarray:
    if curtype == "boolean":
        return values.astype(numpy.bool_)
    elif curtype == "number":
        if not numpy.issubdtype(values.dtype, numpy.floating):
            return values.astype(numpy.double)
    return values


def _choose_NamedList_subclass(values: Sequence, curtype: str) -> Union[IntegerList, FloatList, BooleanList]:
    if curtype == "boolean":
        return BooleanList(values)
    elif curtype == "number":
        return FloatList(values)
    else:
        return IntegerList(values)
