import ctypes as ct
from typing import Any, Optional
from biocframe import BiocFrame
from biocutils import Factor, StringList
import numpy
import h5py
import os

from .alt_read_object import alt_read_object

def read_data_frame(path: str, metadata: dict, **kwargs) -> BiocFrame:
    """Load a data frame from a HDF5 file. In general, this function should not
    be called directly but instead via :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        metadata: Metadata for the object.

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
                values = alt_read_object(os.path.join(path, "other_columns", name), **kwargs)

                # Dicts don't actually satisfy the BiocFrame contract, so while you
                # could stuff a dict in there, it'll fail if you want to, e.g.,
                # slice with repeated rows. So we convert it to a list to be safe.
                # Besides, the row names of the BiocFrame should override any
                # names for the individual columns, so we're not losing much here.
                if isinstance(values, dict):
                    values = list(values.values())

                contents[col] = values
                continue

            xhandle = dhandle[name]
            curtype = xhandle.attrs["type"]

            if curtype == "factor":
                chandle = xhandle["codes"]
                codes = chandle[:]
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
            is_str = (curtype == "string")
            if is_str:
                values = StringList(v.decode('UTF8') for v in values)

            if "missing-value-placeholder" in xhandle.attrs:
                placeholder = xhandle.attrs["missing-value-placeholder"]
                if is_str:
                    for j, y in enumerate(values):
                        if y == placeholder:
                            values[j] = None
                else:
                    if numpy.isnan(placeholder):
                        mask = numpy.isnan(values)
                    else:
                        mask = (values == placeholder)
                    values = numpy.ma.MaskedArray(values, mask=mask)

            if curtype == "boolean":
                values = values.astype(numpy.bool_)
            elif curtype == "number":
                if not numpy.issubdtype(values.dtype, numpy.floating):
                    values = values.astype(numpy.double)

            contents[col] = values

    df = BiocFrame(contents, number_of_rows=expected_rows, row_names=row_names, column_names=column_names)

    other_dir = os.path.join(path, "other_annotations")
    if os.path.exists(other_dir):
        df.set_metadata(alt_read_object(other_dir, **kwargs).as_dict(), in_place=True)

    mcol_dir = os.path.join(path, "column_annotations")
    if os.path.exists(mcol_dir):
        df.set_column_data(alt_read_object(mcol_dir, **kwargs), in_place=True)

    return df
