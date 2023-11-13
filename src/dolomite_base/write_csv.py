from typing import Optional, Any, Literal, Tuple
from functools import singledispatch
import numpy as np
from biocframe import BiocFrame
from biocutils import Factor, StringList
import gzip

from . import _utils as ut
from . import lib_dolomite_base as lib


def _list_element_to_string(s):
    if s is None:
        return "NA"
    return str(s)


def _numpy_element_to_string(s):
    if np.ma.is_masked(s):
        return "NA"
    return str(s)


def _quotify_string(s):
    if '"' in s:
        s = s.replace('"', '""')
    return '"' + s + '"'


def _quotify_string_or_none(s):
    if s is None:
        return "NA"
    return _quotify_string(s)


@singledispatch
def _choose_operation(x: Any):
    final_type, has_none = ut._determine_list_type(x)
    if final_type is None:
        raise ValueError("failed to determine element type for '" + type(x).__name + "' column") 
    if final_type == str:
        if has_none:
            return _quotify_string_or_none
        else:
            return _quotify_string
    else:
        if has_none:
            return _list_element_to_string
        else:
            return str


@_choose_operation.register
def _choose_operation_StringList(x: StringList):
    return _quotify_string_or_none


@_choose_operation.register
def _choose_operation_Factor(x: Factor):
    return _quotify_string_or_none


@_choose_operation.register
def _choose_operation_numpy(x: np.ndarray):
    if np.ma.is_masked(x):
        return _numpy_element_to_string
    else:
        return str


def write_csv(x: BiocFrame, path: str, compression: Literal["none", "gzip"] = "none"):
    """Write a :py:class:`~biocframe.BiocFrame.BiocFrame` to a CSV.
    This is intended for use by developers of dolomite extensions.

    Args:
        x: Data frame.

        path: File path to write the CSV.

        compression: Compression algorithm to use, if any.

    Returns:
        A (compressed) CSV file is created at ``path``.
    """
    columns = []
    operations = []

    for i, col in enumerate(x.get_column_names()):
        current = x.column(col)
        operations.append(_choose_operation(current))
        columns.append(current)

    if compression == "gzip":
        with gzip.open(path, "wb") as handle:
            _dump_csv_contents(handle, x, columns, operations)
    else:
        with open(path, "wb") as handle:
            _dump_csv_contents(handle, x, columns, operations)


def _dump_csv_contents(handle, x: BiocFrame, columns: list, operations: list):
    extracted_row_names = x.get_row_names()
    has_row_names = extracted_row_names is not None

    header_line = ""
    if has_row_names:
        header_line += _quotify_string("row_names")
    for c in x.column_names:
        if header_line:
            header_line += ","
        header_line += _quotify_string(c)
    header_line += "\n"
    handle.write(header_line.encode("UTF8"))

    for r in range(x.shape[0]):
        if has_row_names:
            line = _quotify_string(extracted_row_names[r])
        else:
            line = ""
        for i, trans in enumerate(operations):
            if line:
                line += ","
            line += trans(columns[i][r])
        line += "\n"
        handle.write(line.encode("UTF8"))


def read_csv(path: str, num_rows: int, compression: Literal["none", "gzip"]) -> Tuple[list, list]:
    """Read the contents of a CSV into memory.
    This is intended for use by developers of dolomite extensions.

    Args:
        path: Path to the CSV file.

        num_rows: Number of rows in the CSV file.

        compression: Compression algorithm that was used, if any.

    Returns:
        Tuple containing the (1) a list of strings with the column names and
        (2) a list of lists/arrays with the contents of each column.
    """
    if compression == "gzip":
        compressed = True
    elif compression == "none":
        compressed = False
    else:
        raise NotImplementedError(compression + " decompression is not yet supported for CSVs")

    return lib.load_csv(
        path, 
        num_rows,
        compressed,
        True
    )
