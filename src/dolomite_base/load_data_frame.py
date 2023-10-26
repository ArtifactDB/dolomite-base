import ctypes as ct
from typing import Any, Optional
from biocframe import BiocFrame
import numpy as np
import h5py
import os

from . import lib_dolomite_base as lib
from .acquire_file import acquire_file
from .acquire_metadata import acquire_metadata
from .load_object import load_object
from ._utils import _is_gzip_compressed


def load_csv_data_frame(meta: dict[str, Any], project: Any, **kwargs) -> BiocFrame:
    """Load a data frame from a (possibly Gzip-compressed) CSV file in the
    **comservatory** format. In general, this function should not be called
    directly but instead via :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this CSV data frame.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A data frame.
    """
    full_path = acquire_file(project, meta["path"])
    expected_rows = meta["data_frame"]["dimensions"][0]

    cnames, contents = lib.load_csv(
        full_path, 
        expected_rows,
        _is_gzip_compressed(meta, "csv_data_frame"),
        True
    )

    has_row_names = "row_names" in meta["data_frame"] and meta["data_frame"]["row_names"]
    columns = meta["data_frame"]["columns"]
    if len(columns) + has_row_names != len(contents):
        raise ValueError("difference between the observed and expected number of CSV columns (" + str(observed_cols) + " to " + str(expected_cols) + ")")

    row_names = None
    if has_row_names:
        row_names = contents[0]
        contents = contents[1:]

    return _create_BiocFrame(expected_rows, row_names, columns, contents, project, **kwargs)


def _create_BiocFrame(expected_rows: int, row_names: Optional[list], columns: list, contents: list, project, **kwargs) -> BiocFrame:
    output = BiocFrame({}, number_of_rows=expected_rows, row_names=row_names)

    for i, c in enumerate(contents):
        curval = columns[i]
        if curval["type"] == "other":
            child_meta = acquire_metadata(project, curval["resource"]["path"])
            c = load_object(child_meta, project, **kwargs)

        elif curval["type"] == "integer":
            if not np.issubdtype(c.dtype, np.integer):
                c = c.astype(np.int32)

        elif curval["type"] == "number":
            if not np.issubdtype(c.dtype, np.floating):
                c = c.astype(np.float64)

        elif curval["type"] == "boolean":
            c = c.astype(np.bool_, copy=False)

        elif curval["type"] == "string":
            if not isinstance(c, list): # only happens if the entire column is NA.
                c = [None] * len(c)

        output[curval["name"]] = c

    return output


def load_hdf5_data_frame(meta: dict[str, Any], project: Any, **kwargs) -> BiocFrame:
    """Load a data frame from a HDF5 file. In general, this function should not
    be called directly but instead via :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this HDF5 data frame.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A data frame.
    """
    full_path = acquire_file(project, meta["path"])

    has_row_names = "row_names" in meta["data_frame"] and meta["data_frame"]["row_names"]
    columns = meta["data_frame"]["columns"]

    contents = [None] * len(columns)
    row_names = None
    with h5py.File(full_path, "r") as handle:
        ghandle = handle[meta["hdf5_data_frame"]["group"]]
        if has_row_names:
            row_names = [v.decode("UTF8") for v in ghandle["row_names"]]

        dhandle = ghandle["data"]
        for i in range(len(columns)):
            name = str(i)
            if name not in dhandle:
                continue
            xhandle = dhandle[name]
            values = xhandle[:]

            is_str = columns[i]["type"] == "string"
            if is_str:
                values = [v.decode('UTF8') for v in values]

            if "missing-value-placeholder" in xhandle.attrs:
                placeholder = xhandle.attrs["missing-value-placeholder"]
                if is_str:
                    for j, y in enumerate(values):
                        if y == placeholder:
                            values[j] = None
                else:
                    if isinstance(placeholder, float) and np.isnan(placeholder): # need to handle NaNs with weird payloads.
                        if values.dtype != placeholder.dtype:
                            raise ValueError("types of missing placeholder and array should be the same")
                        tmp = np.array(placeholder, dtype=placeholder.dtype)
                        mask = lib.create_nan_mask(values.ctypes.data, len(values), values.dtype.itemsize, tmp.ctypes.data)
                    else:
                        mask = (values == placeholder)
                    values = np.ma.array(values, mask=mask)

            contents[i] = values 

    expected_rows = meta["data_frame"]["dimensions"][0]
    return _create_BiocFrame(expected_rows, row_names, columns, contents, project, **kwargs)
