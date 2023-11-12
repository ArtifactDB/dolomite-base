import ctypes as ct
from typing import Any, Optional
from biocframe import BiocFrame
from biocutils import Factor, StringList
import numpy as np
import h5py
import os

from . import lib_dolomite_base as lib
from .acquire_file import acquire_file
from .acquire_metadata import acquire_metadata
from .alt_load_object import alt_load_object
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

    df = _create_BiocFrame(expected_rows, row_names, columns, contents, project, **kwargs)
    _attach_metadata(meta, df, project)
    return df


def _create_BiocFrame(expected_rows: int, row_names: Optional[list], columns: list, contents: list, project, **kwargs) -> BiocFrame:
    output = BiocFrame({}, number_of_rows=expected_rows, row_names=row_names)

    for i, c in enumerate(contents):
        curval = columns[i]
        if curval["type"] == "other":
            child_meta = acquire_metadata(project, curval["resource"]["path"])
            c = alt_load_object(child_meta, project, **kwargs)

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
            if not isinstance(c, StringList):
                c = StringList(c)

        elif curval["type"] == "factor":
            if "levels" in curval:
                lev_meta = acquire_metadata(project, curval["levels"]["resource"]["path"])
                levels = alt_load_object(lev_meta, project, **kwargs)
                c = Factor(c, levels, ordered=(curval["ordered"] if "ordered" in curval else False))

        output.set_column(curval["name"], c, in_place=True)

    return output


def _attach_metadata(meta: dict[str, Any], df: BiocFrame, project):
    dmeta = meta["data_frame"]
    if "other_data" in dmeta:
        mmeta = acquire_metadata(project, dmeta["other_data"]["resource"]["path"])
        df.set_metadata(alt_load_object(mmeta, project), in_place=True)
    if "column_data" in dmeta:
        mmeta = acquire_metadata(project, dmeta["column_data"]["resource"]["path"])
        df.set_column_data(alt_load_object(mmeta, project), in_place=True)


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
            row_names = StringList(v.decode("UTF8") for v in ghandle["row_names"])

        dhandle = ghandle["data"]
        for i in range(len(columns)):
            name = str(i)
            if name not in dhandle:
                continue
            xhandle = dhandle[name]

            curtype = columns[i]["type"] 
            if curtype == "factor":
                chandle = xhandle["codes"]
                codes = chandle[:]
                if "missing-value-placeholder" in chandle.attrs:
                    placeholder = chandle.attrs["missing-value-placeholder"]
                    if placeholder != -1:
                        values[values == placeholder] = -1
                levels = StringList(v.decode("UTF8") for v in xhandle["levels"])
                values = Factor(codes, levels, ordered=(columns[i]["ordered"] if "ordered" in columns[i] else False))
            else:
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
                        if np.isnan(placeholder):
                            mask = np.isnan(values)
                        else:
                            mask = (values == placeholder)
                        values = np.ma.array(values, mask=mask)

            contents[i] = values 

    expected_rows = meta["data_frame"]["dimensions"][0]
    df = _create_BiocFrame(expected_rows, row_names, columns, contents, project, **kwargs)
    _attach_metadata(meta, df, project)
    return df
