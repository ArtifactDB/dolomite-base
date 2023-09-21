import ctypes as ct
from typing import Any
from biocframe import BiocFrame
import numpy as np
import os

from . import _cpphelpers as lib

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

    def column(self, i: int, pop: bool = False):
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
            lib.fetch_csv_strings(self.ptr, i, concatenated, pop)

            sofar = 0
            collected = []
            buffer = concatenated.raw
            for i, x in enumerate(strlengths):
                endpoint = sofar + x 
                collected.append(buffer[sofar:endpoint].decode("ASCII"))
                sofar = endpoint

            for i, x in enumerate(mask):
                if x:
                    collected[i] = None
            return collected

        elif col_type.value == 1:
            values = np.ndarray(N, dtype=np.float64)
            mask = np.zeros(N, dtype=np.uint8)
            masked = lib.fetch_csv_numbers(self.ptr, i, values, mask, pop)
            if masked:
                return np.ma.array(values, mask=mask)
            else:
                return values

        elif col_type.value == 3:
            values = np.ndarray(N, dtype=np.uint8)
            masked = lib.fetch_csv_booleans(self.ptr, i, values, pop)
            if masked:
                mask = values == 2
                return np.ma.array(values.astype(dtype=np.bool_), mask=mask)
            else:
                return values.astype(dtype=np.bool_)

        elif col_type.value == -1:
            return None

        else:
            return NotImplementedError("not-yet-supported type for column " + str(i) + " of the CSV (" + str(col_type.value) + ")")


def load_csv_data_frame(meta: dict[str, Any], dir: str, **kwargs):
    full_path = os.path.join(dir, meta["path"])
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
        # Always popping columns from the C++ representation as soon as we're
        # done, so as to free up memory. Not really sure whether this has much
        # of an effect as the C++/Python heaps aren't the same.
        current = handle.column(f, pop=True)
        if f == 0 and has_row_names:
            row_names = current
        else:
            contents.append(current)

    output = BiocFrame({}, number_of_rows=expected_rows, row_names=row_names)
    for i, c in enumerate(contents):
        curval = columns[i]
        if curval["type"] == "other":
            raise NotImplementedError("oops, can't load complex data frame columns yet") 
        if curval["type"] == "integer":
            c = c.astype(np.int32)
        output[curval["name"]] = c
   
    return output
