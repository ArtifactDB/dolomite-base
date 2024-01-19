from typing import Any, Optional, Sequence
from biocframe import BiocFrame
from biocutils import Factor, StringList, IntegerList, FloatList, BooleanList, NamedList
import numpy
import h5py
import os

from .alt_read_object import alt_read_object

def read_data_frame(path: str, metadata: dict, data_frame_represent_column_as_1darray : bool = True, **kwargs) -> BiocFrame:
    """Load a data frame from a HDF5 file. In general, this function should not
    be called directly but instead via :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        metadata: Metadata for the object.

        data_frame_represent_column_as_1darray: 
            Whether numeric columns should be represented as 1-dimensional
            NumPy arrays. This is more efficient than regular Python lists
            but discards the distinction between vectors and 1-D arrays.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A data frame.
    """
    column_names = []
    contents = {}
    row_names = None
    expected_rows = 0

    with h5py.File(os.path.join(path, "basic_columns.h5"), "r") as handle:
        ghandle = handle["data_frame"]
        expected_rows = ghandle.attrs["row-count"][()]
        column_names = [v.decode() for v in ghandle["column_names"]]
        if "row_names" in ghandle:
            row_names = [v.decode("UTF8") for v in ghandle["row_names"]]

        dhandle = ghandle["data"]
        for i, col in enumerate(column_names):
            name = str(i)
            if name not in dhandle:
                contents[col] = alt_read_object(os.path.join(path, "other_columns", name), **kwargs)
                continue

            xhandle = dhandle[name]
            curtype = xhandle.attrs["type"]

            if curtype == "factor":
                chandle = xhandle["codes"]
                codes = chandle[:].astype(numpy.int32)
                if "missing-value-placeholder" in chandle.attrs:
                    placeholder = chandle.attrs["missing-value-placeholder"]
                    codes[codes == placeholder] = -1

                ordered = False
                if "ordered" in xhandle.attrs:
                    ordered = xhandle.attrs["ordered"][()] != 0

                levels = StringList(v.decode("UTF8") for v in xhandle["levels"])
                contents[col] = Factor(codes, levels, ordered=ordered)
                continue

            values = xhandle[:]
            if curtype == "string":
                values = StringList(v.decode('UTF8') for v in values)
                if "missing-value-placeholder" in xhandle.attrs:
                    placeholder = xhandle.attrs["missing-value-placeholder"]
                    for j, y in enumerate(values):
                        if y == placeholder:
                            values[j] = None
                contents[col] = values
                continue

            if "missing-value-placeholder" in xhandle.attrs:
                placeholder = xhandle.attrs["missing-value-placeholder"]
                if numpy.isnan(placeholder):
                    mask = numpy.isnan(values)
                else:
                    mask = (values == placeholder)
                if data_frame_represent_column_as_1darray:
                    contents[col] = numpy.ma.MaskedArray(_coerce_numpy_type(values, curtype), mask=mask)
                else:
                    values = []
                    for i, y in enumerate(values):
                        if mask[i]:
                            values.append(None)
                        else:
                            values.append(y)
                    contents[col] = _choose_NamedList_subclass(values, curtype)
                continue

            if data_frame_represent_column_as_1darray:
                contents[col] = _coerce_numpy_type(values, curtype)
            else:
                contents[col] = _choose_NamedList_subclass(values, curtype)

    df = BiocFrame(contents, number_of_rows=expected_rows, row_names=row_names, column_names=column_names)

    other_dir = os.path.join(path, "other_annotations")
    if os.path.exists(other_dir):
        df.set_metadata(alt_read_object(other_dir, **kwargs).as_dict(), in_place=True)

    mcol_dir = os.path.join(path, "column_annotations")
    if os.path.exists(mcol_dir):
        df.set_column_data(alt_read_object(mcol_dir, **kwargs), in_place=True)

    return df


def _coerce_numpy_type(values: numpy.ndarray, curtype: str) -> numpy.ndarray:
    if curtype == "boolean":
        return values.astype(numpy.bool_)
    elif curtype == "number":
        if not numpy.issubdtype(values.dtype, numpy.floating):
            return values.astype(numpy.double)
    return values


def _choose_NamedList_subclass(values: Sequence, curtype: str) -> NamedList:
    if curtype == "boolean":
        return BooleanList(values)
    elif curtype == "number":
        return NumberList(values)
    else:
        return IntegerList(values)

