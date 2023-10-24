from typing import Optional, Any
from biocframe import BiocFrame
from ._process_columns import (
    _process_columns_for_csv, 
    _write_csv,
)
import gzip
from . import lib_dolomite_base as lib


def write_csv(x: BiocFrame, path: str, compressed: bool = False):
    """Write a :py:class:`~biocframe.BiocFrame.BiocFrame` to a CSV.
    This is intended for use by developers of dolomite extensions.

    Args:
        x: Data frame.

        path: File path to write the CSV.

        compressed: Whether to save it in Gzip-compressed form.

    Returns:
        A (compressed) CSV file is created at ``path``.
    """
    columns, otherable, operations = _process_columns_for_csv(x)
    if len(otherable):
        raise ValueError("unsupported types for column " + str(otherable[0]))

    if compressed:
        with gzip.open(path, "wb") as handle:
            _write_csv(x, handle, operations)
    else:
        with open(path, "wb") as handle:
            _write_csv(x, handle, operations)


def read_csv(path: str, num_rows: int, compressed: bool) -> dict[str, Any]:
    """Read the contents of a CSV into memory.
    This is intended for use by developers of dolomite extensions.

    Args:
        path: Path to the CSV file.

        num_rows: Number of rows in the CSV file.

        compressed: Whether the CSV file is Gzip-compressed.

    Returns:
        Dictionary containing the column ``names`` and the contents of the
        ``fields`` as lists or NumPy arrays.
    """
    return lib.load_csv(
        path, 
        num_rows,
        compressed,
        True
    )
