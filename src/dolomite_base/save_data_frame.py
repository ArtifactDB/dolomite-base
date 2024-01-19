from typing import Any, Tuple, Optional, Literal
from collections import namedtuple
from functools import singledispatch
import os
from biocframe import BiocFrame
from biocutils import Factor, StringList, get_height
import numpy
import h5py
import gzip

from .save_object import save_object
from .alt_save_object import alt_save_object
from . import _utils as ut


@save_object.register
def save_data_frame(x: BiocFrame, path: str, data_frame_convert_list_to_vector: bool = True, **kwargs) -> dict[str, Any]:
    """Method for saving :py:class:`~biocframe.BiocFrame.BiocFrame`
    objects to the corresponding file representations, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to a directory in which to save `x`.

        data_frame_convert_list_to_vector: 
            Whether data frame columns that are lists should be saved as typed
            vectors, if all entries are of the same basic type (integer,
            string, float, boolean) or None. This is more compact but will
            change the Python representation on loading, e.g., all-integer
            columns will be reloaded as NumPy arrays.

        kwargs: 
            Further arguments, passed to internal
            :py:func:`~dolomite_base.alt_save_object.alt_save_object` calls.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)

    other = []
    full = os.path.join(path, "basic_columns.h5")
    with h5py.File(full, "w") as handle:
        ghandle = handle.create_group("data_frame")
        ghandle.attrs.create("row-count", data=x.shape[0], dtype="u8")
        ghandle.attrs.create("version", data="1.0")

        dhandle = ghandle.create_group("data")
        output = Hdf5ColumnOutput(handle=dhandle, otherable=other, convert_list_to_vector=data_frame_convert_list_to_vector)
        for i in range(x.shape[1]):
            _process_column_for_hdf5(x.get_column(i), i, output)

        ut._save_fixed_length_strings(ghandle, "column_names", x.get_column_names())
        rn = x.get_row_names()
        if rn is not None:
            ut._save_fixed_length_strings(ghandle, "row_names", rn)

    if len(other):
        other_dir = os.path.join(path, "other_columns")
        os.mkdir(other_dir)
        for i in other:
            alt_save_object(x.get_column(i), os.path.join(other_dir, str(i)), data_frame_convert_list_to_vector=data_frame_convert_list_to_vector, **kwargs)

    md = x.get_metadata()
    if md is not None and len(md):
        alt_save_object(x.metadata, os.path.join(path, "other_annotations"), data_frame_convert_list_to_vector=data_frame_convert_list_to_vector, **kwargs)

    cd = x.get_column_data(with_names=False) 
    if cd is not None and cd.shape[1] > 0:
        if cd.get_row_names() is not None:
            cd = cd.set_row_names(None)
        alt_save_object(cd, os.path.join(path, "column_annotations"), data_frame_convert_list_to_vector=data_frame_convert_list_to_vector, **kwargs)

    with open(os.path.join(path, "OBJECT"), "w") as handle:
        handle.write('{ "type": "data_frame", "data_frame": { "version": "1.0" } }')


########################################################


def _select_hdf5_placeholder(current, dtype) -> Tuple:
    if dtype == float:
        copy, placeholder = ut._choose_missing_float_placeholder(current)
    elif dtype == int:
        copy, placeholder, dtype = ut._choose_missing_integer_placeholder(current)
    elif dtype == str:
        copy, placeholder = ut._choose_missing_string_placeholder(current)
    elif dtype == bool:
        copy, placeholder = ut._choose_missing_boolean_placeholder(current)
    else:
        raise NotImplementedError("saving a list of " + str(dtype) + " is not supported yet")
    return copy, placeholder, dtype


Hdf5ColumnOutput = namedtuple('Hdf5ColumnOutput', ['handle', 'otherable', 'convert_list_to_vector'])


def _dump_column_to_hdf5(contents, data_type, placeholder, index: str, output: Hdf5ColumnOutput):
    if data_type == int:
        savetype = 'i4'
        typename = "integer"
    elif data_type == float:
        savetype = 'f8'
        typename = "number"
    elif data_type == str:
        savetype = None
        typename = "string"
    elif data_type == bool:
        savetype = 'i1'
        typename = "boolean"
    else:
        raise NotImplementedError("saving a list of " + str(data_type) + " is not supported yet")

    if savetype: 
        dhandle = output.handle.create_dataset(str(index), data=contents, dtype=savetype, compression="gzip", chunks=True)
    else:
        dhandle = ut._save_fixed_length_strings(output.handle, str(index), contents)

    dhandle.attrs.create("type", data=typename)
    if placeholder:
        dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype=savetype)


def _determine_list_type(x: list) -> Tuple:
    all_types = set()
    has_none = False
    for y in x:
        if isinstance(y, numpy.generic):
            if numpy.issubdtype(y.dtype.type, numpy.integer):
                all_types.add(int)
            elif numpy.issubdtype(y.dtype.type, numpy.floating):
                all_types.add(float)
            else:
                all_types.add(type(y))
        elif ut._is_missing_scalar(y):
            has_none = True
        else:
            all_types.add(type(y))

    final_type = None
    if len(all_types) == 1:
        final_type = list(all_types)[0]
        if final_type == int:
            if not ut._is_integer_vector_within_limit(x):
                final_type = float 
        elif final_type != str and final_type != bool and final_type != float:
            final_type = None
    elif len(all_types) == 2 and int in all_types and float in all_types:
        final_type = float
    elif len(all_types) == 0:
        final_type = None

    return final_type, has_none


def _convert_list_with_missingness(x: list, final_type) -> Tuple:
    if final_type == str: 
        copy = x[:]
        for i, y in enumerate(x):
            if ut._is_missing_scalar(y):
                copy[i] = None
        x = StringList(copy)

    else:
        if final_type == int:
            dtype = numpy.int32
        elif final_type == bool:
            dtype = numpy.int8
        else:
            dtype = numpy.float64

        mask = numpy.ndarray(len(x), dtype=numpy.bool_)
        copy = numpy.ndarray(len(x), dtype=dtype)
        for i, y in enumerate(x):
            if ut._is_missing_scalar(y):
                copy[i] = 0
                mask[i] = True
            else:
                copy[i] = y
                mask[i] = False
        x = numpy.ma.MaskedArray(copy, mask=mask)

    return _select_hdf5_placeholder(x, final_type)


########################################################


@singledispatch
def _process_column_for_hdf5(x: Any, index: int, output: Hdf5ColumnOutput):
    output.otherable.append(index)


@_process_column_for_hdf5.register
def _process_list_column_for_hdf5(x: list, index: int, output: Hdf5ColumnOutput):
    if output.convert_list_to_vector:
        final_type, has_none = _determine_list_type(x)

        if final_type == str:
            placeholder = None
            if has_none:
                x, placeholder = ut.choose_missing_string_placeholder(x) 
            dhandle = ut.save_fixed_length_strings(output.handle, str(index), x)
            dhandle.attrs["type"] = "string"
            dhandle.attrs["_python_original_type"] = "list"
            if placeholder:
                dhandle.attrs.create("missing-value-placeholder", data=placeholder)

            return

        elif final_type == int:
            if ut._is_integer_vector_within_limit(x):
                placeholder = None
                if has_none:
                    x, mask = ut.list_to_numpy_mask(x, numpy.int32)
                    x, placeholder = ut.choose_missing_integer_placeholder(x, mask, copy=False) 
                dhandle = output.handle.create_dataset(str(index), data=x, dtype="i4", compression="gzip", chunks=True)
                dhandle.attrs.create("type", data="integer")
                dhandle.attrs["_python_original_type"] = "list"
                if placeholder:
                    dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype="i4")
                return
            else:
                placeholder = None
                if has_none:
                    x, mask = ut.list_to_numpy_mask(x, numpy.float64)
                    x, placeholder = ut.choose_missing_integer_placeholder(x, mask, copy=False) 
                dhandle = output.handle.create_dataset(str(index), data=x, dtype="f8", compression="gzip", chunks=True)
                dhandle.attrs.create("type", data="number")
                dhandle.attrs["_python_original_type"] = "list"
                if placeholder:
                    dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype="f8")
                return

        elif final_type == float:
            placeholder = None
            if has_none:
                x, mask = ut.list_to_numpy_mask(x, numpy.float64)
                x, placeholder = ut.choose_missing_float_placeholder(x, mask, copy=False) 

            dhandle = ut.save_fixed_length_strings(output.handle, str(index), x)
            dhandle.attrs.create("type", data="integer")
            dhandle.attrs["_python_original_type"] = "list"
            if placeholder:
                dhandle.attrs.create("missing-value-placeholder", data=placeholder)
            return

        elif final_type == bool:
            placeholder = None
            if has_none:
                x, mask = ut.list_to_numpy_mask(x, numpy.float64)
                x, placeholder = ut.choose_missing_float_placeholder(x, mask, copy=False) 

            dhandle = ut.save_fixed_length_strings(output.handle, str(index), x)
            dhandle.attrs.create("type", data="integer")
            dhandle.attrs["_python_original_type"] = "list"
            if placeholder:
                dhandle.attrs.create("missing-value-placeholder", data=placeholder)
            return







    _process_column_for_hdf5.registry[object](x, index, output)


@_process_column_for_hdf5.register
def _process_StringList_column_for_hdf5(x: StringList, index: int, output: Hdf5ColumnOutput):
    placeholder = None
    if any(y is None for y in x):
        x, placeholder = ut.choose_missing_string_placeholder(current)
    dhandle = ut.save_fixed_length_strings(output.handle, str(index), x)
    dhandle.attrs.create("type", data="string")
    if placeholder:
        dhandle.attrs.create("missing-value-placeholder", data=placeholder)


@_process_column_for_hdf5.register
def _process_StringList_column_for_hdf5(x: StringList, index: int, output: Hdf5ColumnOutput):
    placeholder = None
    if any(y is None for y in x):
        x, placeholder = ut._choose_missing_string_placeholder(current)
    _dump_column_to_hdf5(x, str, placeholder, index, output)



@_process_column_for_hdf5.register
def _process_numpy_column_for_hdf5(x: numpy.ndarray, index: int, output: Hdf5ColumnOutput):
    final_type = ut._determine_save_type(x)
    placeholder = None
    if ut._is_actually_masked(x):
        x, placeholder, final_type = _select_hdf5_placeholder(x, final_type)
    _dump_column_to_hdf5(x, final_type, placeholder, index, output)


@_process_column_for_hdf5.register
def _process_factor_column_for_hdf5(x: Factor, index: int, output: Hdf5ColumnOutput):
    ghandle = output.handle.create_group(str(index))
    ghandle.attrs.create("type", data="factor")
    ghandle.attrs.create("ordered", data=x.get_ordered(), dtype="i1")

    ut._save_fixed_length_strings(ghandle, "levels", x.get_levels())

    codes = x.get_codes()
    is_missing = codes == -1
    has_missing = is_missing.any()
    nlevels = len(x.get_levels())
    if has_missing:
        codes = codes.astype(numpy.uint32, copy=True)
        codes[is_missing] = nlevels

    dhandle = ghandle.create_dataset("codes", data=codes, dtype='u4', compression="gzip", chunks=True)
    if has_missing:
        dhandle.attrs.create("missing-value-placeholder", data=nlevels, dtype='u4')
