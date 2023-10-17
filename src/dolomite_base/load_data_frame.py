import ctypes as ct
from typing import Any, Optional
from biocframe import BiocFrame
import numpy as np
import h5py
import os

from . import _cpphelpers as lib
from .acquire_file import acquire_file
from .acquire_metadata import acquire_metadata
from .load_object import load_object
from ._utils import _fragment_string_contents, _mask_strings

class _LoadedCsvHolder:
    def __init__(self, ptr):
        self.ptr = ptr
        self._numf = None
        self._numr = None

    def __del__(self):
        lib.free_csv(self.ptr)

    def num_fields(self):
        if self._numf is None:
            self._numf = lib.get_csv_num_fields(self.ptr)
        return self._numf

    def num_records(self):
        if self._numr is None:
            self._numr = lib.get_csv_num_records(self.ptr)
        return self._numr

    def column(self, i: int):
        col_type = ct.c_int32(0)
        col_size = ct.c_int32(0)
        col_loaded = ct.c_int32(0)
        lib.get_csv_column_stats(self.ptr, i, ct.byref(col_type), ct.byref(col_size), ct.byref(col_loaded))

        if col_loaded.value == 0:
            return None
        N = col_size.value
        if N != self.num_records():
            raise ValueError("column size exceeds the number of records in the CSV")

        if col_type.value == 0:
            strlengths = np.ndarray(N, dtype=np.int32)
            mask = np.zeros(N, dtype=np.uint8)
            lib.get_csv_string_stats(self.ptr, i, strlengths, mask)

            total_len = int(strlengths.sum())
            concatenated = ct.create_string_buffer(total_len)
            lib.fetch_csv_strings(self.ptr, i, concatenated)

            collected = _fragment_string_contents(strlengths, concatenated.raw)
            _mask_strings(collected, mask)
            return collected

        elif col_type.value == 1:
            values = np.ndarray(N, dtype=np.float64)
            mask = np.zeros(N, dtype=np.uint8)
            masked = lib.fetch_csv_numbers(self.ptr, i, values, mask)
            if masked:
                return np.ma.array(values, mask=mask)
            else:
                return values

        elif col_type.value == 3:
            values = np.ndarray(N, dtype=np.uint8)
            masked = lib.fetch_csv_booleans(self.ptr, i, values)
            if masked:
                mask = values == 2
                return np.ma.array(values.astype(dtype=np.bool_), mask=mask)
            else:
                return values.astype(dtype=np.bool_)

        elif col_type.value == -1:
            return None

        else:
            return NotImplementedError("not-yet-supported type for column " + str(i) + " of the CSV (" + str(col_type.value) + ")")


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
    handle = _LoadedCsvHolder(lib.load_csv(full_path.encode("UTF8")))

    has_row_names = "row_names" in meta["data_frame"] and meta["data_frame"]["row_names"]
    columns = meta["data_frame"]["columns"]

    expected_cols = len(columns) + has_row_names 
    observed_cols = handle.num_fields()
    if expected_cols != observed_cols:
        raise ValueError("difference between the observed and expected number of CSV columns (" + str(observed_cols) + " to " + str(expected_cols) + ")")

    expected_rows = meta["data_frame"]["dimensions"][0]
    observed_rows = handle.num_records()
    if expected_rows != observed_rows:
        raise ValueError("difference between the observed and expected number of CSV rows (" + str(observed_rows) + " to " + str(expected_rows) + ")")

    contents = []
    row_names = None
    for f in range(observed_cols):
        current = handle.column(f)
        if f == 0 and has_row_names:
            row_names = current
        else:
            contents.append(current)

    del handle
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

        elif curval["type"] == "boolean":
            c = c.astype(np.bool_, copy=False)

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

        for i in range(len(columns)):
            name = str(i)
            if name not in ghandle:
                continue
            dhandle = ghandle[name]
            values = dhandle[:]

            is_str = columns[i]["type"] == "string"
            if is_str:
                values = [v.decode('UTF8') for v in values]

            if "missing-value-placeholder" in dhandle.attrs:
                placeholder = dhandle.attrs["missing-value-placeholder"]
                if is_str:
                    for j, y in enumerate(values):
                        if y == placeholder:
                            values[j] = None
                else:
                    if isinstance(placeholder, float) and np.isnan(placeholder): # need to handle NaNs with weird payloads.
                        mask = np.ndarray(len(values), dtype=np.uint8)
                        buffered = np.array(placeholder, dtype=values.dtype)
                        lib.fill_nan_mask(values.ctypes.data, len(values), buffered.ctypes.data, values.dtype.itemsize, mask)
                    else:
                        mask = (values == placeholder)
                    values = np.ma.array(values, mask=mask)

            contents[i] = values 

    expected_rows = meta["data_frame"]["dimensions"][0]
    return _create_BiocFrame(expected_rows, row_names, columns, contents, project, **kwargs)
