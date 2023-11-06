from typing import Any, Tuple, Optional, Literal
import os
from biocframe import BiocFrame
import numpy as np
import h5py
import gzip

from . import lib_dolomite_base as lib
from .stage_object import stage_object
from .alt_stage_object import alt_stage_object
from .write_metadata import write_metadata
from . import _utils as ut
from ._process_columns import (
    _process_columns_for_csv, 
    _process_columns_for_hdf5, 
    _save_fixed_length_strings, 
    _write_csv,
)


@stage_object.register
def stage_data_frame(
    x: BiocFrame, 
    dir: str, 
    path: str, 
    is_child: bool = False, 
    mode: Optional[Literal["hdf5", "csv"]] = None,
    **kwargs
) -> dict[str, Any]:
    """Method for saving :py:class:`~biocframe.BiocFrame.BiocFrame`
    objects to the corresponding file representations, see
    :py:meth:`~dolomite_base.stage_object.stage_object` for details.

    Args:
        x: Object to be staged.

        dir: Staging directory.

        path: Relative path inside ``dir`` to save the object.

        is_child: Is ``x`` a child of another object?

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    os.mkdir(os.path.join(dir, path))

    if mode == None:
        mode = choose_data_frame_format()
    if mode == "csv":
        meta, other = _stage_csv_data_frame(x, dir, path, is_child=is_child)
    else:
        meta, other = _stage_hdf5_data_frame(x, dir, path, is_child=is_child)

    for i in other:
        more_meta = alt_stage_object(x.column(i), dir, path + "/child-" + str(i + 1), is_child = True)
        resource_stub = write_metadata(more_meta, dir=dir)
        meta["data_frame"]["columns"][i]["resource"] = resource_stub

    if x.metadata is not None and len(x.metadata):
        mmeta = alt_stage_object(x.metadata, dir, path + "/other", is_child=True)
        meta["data_frame"]["other_data"] = { "resource": write_metadata(mmeta, dir=dir) }

    if x.mcols is not None and x.mcols.shape[1] > 0:
        mmeta = alt_stage_object(x.mcols, dir, path + "/mcols", is_child=True)
        meta["data_frame"]["column_data"] = { "resource": write_metadata(mmeta, dir=dir) }

    return meta


DATA_FRAME_FORMAT = "csv"


def choose_data_frame_format(format: Optional[Literal["hdf5", "csv"]] = None) -> str:
    """Get or set the format to save a simple list.

    Args:
        format: Format to save a simple list, either in HDF5 or as a CSV.

    Return:
        If ``format`` is not provided, the current format choice is returned.
        This defaults to `"csv"` if no other setting has been provided.

        If ``format`` is provided, it is used to define the format choice,
        and the previous choice is returned.
    """
    global DATA_FRAME_FORMAT
    if format is None:
        return DATA_FRAME_FORMAT
    else:
        old = DATA_FRAME_FORMAT
        DATA_FRAME_FORMAT = format
        return old


def _stage_hdf5_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    basename = "simple.h5"
    groupname = "df"

    full = os.path.join(dir, path, basename)
    with h5py.File(full, "w") as handle:
        ghandle = handle.create_group(groupname)
        dhandle = ghandle.create_group("data")
        columns, otherable = _process_columns_for_hdf5(x, dhandle)
        _save_fixed_length_strings(ghandle, "column_names", x.column_names)

        has_row_names = x.row_names is not None
        if has_row_names:
            _save_fixed_length_strings(ghandle, "row_names", x.row_names)

    metadata = {
        "$schema": "hdf5_data_frame/v1.json",
        "path": path + "/" + basename,
        "is_child": is_child,
        "data_frame": {
            "columns": columns,
            "row_names": has_row_names,
            "dimensions": list(x.shape),
            "version": 2,
        },
        "hdf5_data_frame": {
            "group": groupname,
            "version": 2
        }
    }

    # Running some validation.
    all_names, all_types, all_formats = _inspect_columns(columns)
    lib.check_hdf5_df(
        full, 
        groupname, 
        x.shape[0], 
        has_row_names,
        all_names,
        all_types,
        all_formats,
        [[]] * len(columns),
        metadata["data_frame"]["version"],
        metadata["hdf5_data_frame"]["version"]
    )

    return metadata, otherable


def _stage_csv_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    columns, otherable, operations = _process_columns_for_csv(x)
    has_row_names = x.row_names is not None

    # Manual serialization into a Gzip-compressed CSV, because 
    # pandas doesn't quite give me what I want... oh well.
    basename = "simple.csv.gz"
    full = os.path.join(dir, path, basename)
    with gzip.open(full, "wb") as handle:
        _write_csv(x, handle, operations)

    metadata = {
        "$schema": "csv_data_frame/v1.json",
        "path": path + "/" + basename,
        "is_child": is_child,
        "data_frame": {
            "columns": columns,
            "row_names": has_row_names,
            "dimensions": list(x.shape),
            "version": 2,
        },
        "csv_data_frame": {
            "compression": "gzip",
        }
    }

    # Running some validation.
    all_names, all_types, all_formats = _inspect_columns(columns)
    lib.check_csv_df(
        full, 
        x.shape[0], 
        has_row_names,
        all_names,
        all_types,
        all_formats,
        [[]] * len(columns),
        metadata["data_frame"]["version"],
        metadata["csv_data_frame"]["compression"] == 'gzip',
        True,
    )

    return metadata, otherable


def _inspect_columns(columns: list) -> Tuple:
    all_names = [None] * len(columns)
    all_types = np.zeros(len(columns), dtype=np.int32)
    all_formats = np.zeros(len(columns), dtype=np.int32)

    type_mapping = { "integer": 0, "number": 1, "string": 2, "boolean": 3, "factor": 4, "other": 5 }
    format_mapping = { "date": 1, "date-time": 2 }
    for i, x in enumerate(columns):
        all_names[i] = x["name"]
        all_types[i] = type_mapping[x["type"]]
        if "format" in x:
            all_formats[i] = format_mapping[x["format"]]

    return all_names, all_types, all_formats
