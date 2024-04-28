from typing import Any, Dict
from collections import namedtuple
from functools import singledispatch
import os
from biocframe import BiocFrame
from biocutils import Factor, StringList, IntegerList, BooleanList, FloatList
import numpy
import h5py

from .save_object import save_object
from .save_object_file import save_object_file
from .alt_save_object import alt_save_object
from . import _utils_string as strings
from . import write_vector_to_hdf5 as write
from ._utils_factor import save_factor_to_hdf5


@save_object.register
def save_data_frame(
    x: BiocFrame, 
    path: str, 
    data_frame_convert_list_to_vector: bool = True, 
    data_frame_convert_1darray_to_vector: bool = True, 
    **kwargs
) -> Dict[str, Any]:
    """Method for saving :py:class:`~biocframe.BiocFrame.BiocFrame`
    objects to the corresponding file representations, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to a directory in which to save ``x``.

        data_frame_convert_list_to_vector: 
            If a column is a regular Python list where all entries are of the
            same basic type (integer, string, float, boolean) or None, should
            it be converted to a typed vector in the on-disk representation?
            This avoids creating a separate file to store this column but
            changes the class of the column when the ``BiocFrame`` is read back
            into a Python session. If ``False``, the list is saved as an
            external object instead.

        data_frame_convert_1darray_to_vector: 
            If a column is a 1D NumPy array, should it be saved as a typed
            vector? This avoids creating a separate file for the column but
            discards the distinction between 1D arrays and vectors. Usually
            this is not an important difference, but nonetheless, users can
            set this flag to ``False`` to save all 1D NumPy arrays as an 
            external "dense array" object instead.

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
        output = Hdf5ColumnOutput(
            handle=dhandle, 
            otherable=other, 
            convert_list_to_vector=data_frame_convert_list_to_vector, 
            convert_1darray_to_vector=data_frame_convert_1darray_to_vector
        )
        for i in range(x.shape[1]):
            _process_column_for_hdf5(x.get_column(i), i, output)

        strings.save_fixed_length_strings(ghandle, "column_names", x.get_column_names())
        rn = x.get_row_names()
        if rn is not None:
            strings.save_fixed_length_strings(ghandle, "row_names", rn)

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

    save_object_file(path, "data_frame", { "data_frame": { "version": "1.0" } })
    return


########################################################


Hdf5ColumnOutput = namedtuple('Hdf5ColumnOutput', ['handle', 'otherable', 'convert_list_to_vector', 'convert_1darray_to_vector'])


@singledispatch
def _process_column_for_hdf5(x: Any, index: int, output: Hdf5ColumnOutput):
    output.otherable.append(index)
    return


@_process_column_for_hdf5.register
def _process_list_column_for_hdf5(x: list, index: int, output: Hdf5ColumnOutput):
    if output.convert_list_to_vector:
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
            elif y is None:
                has_none = True
            else:
                all_types.add(type(y))

        final_type = None
        if len(all_types) == 1:
            final_type = list(all_types)[0]
        elif len(all_types) == 2 and int in all_types and float in all_types:
            final_type = float
        elif len(all_types) == 0:
            final_type = None

        if final_type == str:
            dhandle = write.write_string_vector_to_hdf5(output.handle, str(index), x)
            dhandle.attrs["type"] = "string"
            return

        elif final_type == int:
            dhandle = write.write_integer_vector_to_hdf5(output.handle, str(index), x, allow_float_promotion=True)
            if numpy.issubdtype(dhandle.dtype, numpy.floating):
                dhandle.attrs["type"] = "number"
            else:
                dhandle.attrs["type"] = "integer"
            return

        elif final_type == float:
            dhandle = write.write_float_vector_to_hdf5(output.handle, str(index), x)
            dhandle.attrs["type"] = "number"
            return

        elif final_type == bool:
            dhandle = write.write_boolean_vector_to_hdf5(output.handle, str(index), x)
            dhandle.attrs["type"] = "boolean"
            return

    _process_column_for_hdf5.registry[object](x, index, output)
    return


@_process_column_for_hdf5.register
def _process_StringList_column_for_hdf5(x: StringList, index: int, output: Hdf5ColumnOutput):
    dhandle = write.write_string_vector_to_hdf5(output.handle, str(index), x.as_list())
    dhandle.attrs["type"] = "string"
    return


@_process_column_for_hdf5.register
def _process_IntegerList_column_for_hdf5(x: IntegerList, index: int, output: Hdf5ColumnOutput):
    dhandle = write.write_integer_vector_to_hdf5(output.handle, str(index), x.as_list(), allow_float_promotion=True)
    if numpy.issubdtype(dhandle.dtype, numpy.floating):
        dhandle.attrs["type"] = "number"
    else:
        dhandle.attrs["type"] = "integer"
    return


@_process_column_for_hdf5.register
def _process_FloatList_column_for_hdf5(x: FloatList, index: int, output: Hdf5ColumnOutput):
    dhandle = write.write_float_vector_to_hdf5(output.handle, str(index), x.as_list())
    dhandle.attrs["type"] = "number"
    return


@_process_column_for_hdf5.register
def _process_BooleanList_column_for_hdf5(x: BooleanList, index: int, output: Hdf5ColumnOutput):
    dhandle = write.write_float_vector_to_hdf5(output.handle, str(index), x.as_list())
    dhandle.attrs["type"] = "boolean"
    return


@_process_column_for_hdf5.register
def _process_ndarray_column_for_hdf5(x: numpy.ndarray, index: int, output: Hdf5ColumnOutput):
    if output.convert_1darray_to_vector and len(x.shape) == 1:
        if numpy.issubdtype(x.dtype, numpy.floating):
            dhandle = write.write_float_vector_to_hdf5(output.handle, str(index), x)
            dhandle.attrs["type"] = "number"

        elif x.dtype == numpy.bool_:
            dhandle = write.write_boolean_vector_to_hdf5(output.handle, str(index), x)
            dhandle.attrs["type"] = "boolean"

        elif numpy.issubdtype(x.dtype, numpy.integer):
            dhandle = write.write_integer_vector_to_hdf5(output.handle, str(index), x, allow_float_promotion=True)
            if numpy.issubdtype(dhandle.dtype, numpy.floating):
                dhandle.attrs["type"] = "number"
            else:
                dhandle.attrs["type"] = "integer"

        elif numpy.issubdtype(x.dtype, numpy.str_):
            dhandle = write.write_string_vector_to_hdf5(output.handle, str(index), x)
            dhandle.attrs["type"] = "string"

        else:
            raise NotImplementedError("cannot save column of type '" + x.dtype.name + "'")

        return

    _process_column_for_hdf5.registry[object](x, index, output)
    return


@_process_column_for_hdf5.register
def _process_factor_column_for_hdf5(x: Factor, index: int, output: Hdf5ColumnOutput):
    ghandle = output.handle.create_group(str(index))
    ghandle.attrs.create("type", data="factor")
    save_factor_to_hdf5(ghandle, x)
    return
