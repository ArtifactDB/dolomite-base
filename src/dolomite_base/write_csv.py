from typing import Optional, Any, Literal
from biocframe import BiocFrame
from ._process_columns import (
    _process_columns_for_csv, 
    _write_csv,
)
import gzip
from . import lib_dolomite_base as lib


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
    columns, otherable, operations = _process_columns_for_csv(x)
    if len(otherable):
        raise ValueError("unsupported types for column " + str(otherable[0]))

    if compression == "gzip":
        with gzip.open(path, "wb") as handle:
            _write_csv(x, handle, operations)
    else:
        with open(path, "wb") as handle:
            _write_csv(x, handle, operations)


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
