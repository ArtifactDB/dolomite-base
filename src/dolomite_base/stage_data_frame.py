from typing import Any, Tuple, Optional, Literal
from collections import namedtuple
from functools import singledispatch
import os
from biocframe import BiocFrame
from biocutils import Factor, StringList, get_height
import numpy as np
import h5py
import gzip

from . import lib_dolomite_base as lib
from .stage_object import stage_object
from .alt_stage_object import alt_stage_object
from .write_metadata import write_metadata
from .acquire_metadata import acquire_metadata
from .alt_load_object import alt_load_object
from .write_csv import write_csv
from . import _utils as ut


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
        more_meta = alt_stage_object(x.get_column(i), dir, path + "/child-" + str(i + 1), is_child = True)
        resource_stub = write_metadata(more_meta, dir=dir)
        meta["data_frame"]["columns"][i]["resource"] = resource_stub

    md = x.get_metadata()
    if md is not None and len(md):
        mmeta = alt_stage_object(x.metadata, dir, path + "/other", is_child=True)
        meta["data_frame"]["other_data"] = { "resource": write_metadata(mmeta, dir=dir) }

    cd = x.get_column_data(with_names=False) 
    if cd is not None and cd.shape[1] > 0:
        mmeta = alt_stage_object(cd, dir, path + "/column_data", is_child=True)
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


########################################################


def _select_hdf5_placeholder(current, dtype) -> Tuple:
    if dtype == float:
        copy, placeholder = ut._choose_missing_float_placeholder(current)
    elif dtype == int:
        # If there's no valid missing placeholder, we just save it as floating-point.
        copy, placeholder = ut._choose_missing_integer_placeholder(current)
        if copy is None: 
            copy, placeholder = ut._choose_missing_float_placeholder(current)
            dtype = float
    elif dtype == str:
        copy, placeholder = ut._choose_missing_string_placeholder(current)
    elif dtype == bool:
        copy, placeholder = ut._choose_missing_boolean_placeholder(current)
    else:
        raise NotImplementedError("saving a list of " + str(dtype) + " is not supported yet")
    return copy, placeholder, dtype


Hdf5ColumnOutput = namedtuple('Hdf5ColumnOutput', ['handle', 'columns', 'otherable'])


def _dump_column_to_hdf5(contents, data_type, placeholder, name: str, index: str, output: Hdf5ColumnOutput):
    if data_type == int:
        col_meta = { "type": "integer", "name": name }
        savetype = 'i4'
    elif data_type == float:
        col_meta = { "type": "number", "name": name }
        savetype = 'f8'
    elif data_type == str:
        col_meta = { "type": "string", "name": name }
        savetype = None
    elif data_type == bool:
        col_meta = { "type": "boolean", "name": name }
        savetype = 'i1'
    else:
        raise NotImplementedError("saving a list of " + str(data_type) + " is not supported yet")

    output.columns.append(col_meta)

    if savetype: 
        dhandle = output.handle.create_dataset(str(index), data=contents, dtype=savetype, compression="gzip", chunks=True)
    else:
        dhandle = ut._save_fixed_length_strings(output.handle, str(index), contents)

    dhandle.attrs.create("type", data=col_meta["type"])
    if placeholder:
        dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype=savetype)


@singledispatch
def _process_column_for_hdf5(x: Any, name: str, index: int, output: Hdf5ColumnOutput):
    output.columns.append({ "type": "other", "name": name })
    output.otherable.append(index)


@_process_column_for_hdf5.register
def _process_list_column_for_hdf5(x: list, name: str, index: int, output: Hdf5ColumnOutput):
    final_type, has_none = ut._determine_list_type(x)
    if final_type is None:
        _process_column_for_hdf5.registry[object](x, name, index, output)
    else:
        placeholder = None
        if has_none:
            x, placeholder, final_type = _select_hdf5_placeholder(x, final_type)
        _dump_column_to_hdf5(x, final_type, placeholder, name, index, output)


@_process_column_for_hdf5.register
def _process_StringList_column_for_hdf5(x: StringList, name: str, index: int, output: Hdf5ColumnOutput):
    placeholder = None
    if any(y is None for y in x):
        x, placeholder = ut._choose_missing_string_placeholder(current)
    _dump_column_to_hdf5(x, str, placeholder, name, index, output)


@_process_column_for_hdf5.register
def _process_numpy_column_for_hdf5(x: np.ndarray, name: str, index: int, output: Hdf5ColumnOutput):
    final_type = ut._determine_numpy_type(x)
    placeholder = None
    if np.ma.is_masked(x):
        x, placeholder, final_type = _select_hdf5_placeholder(x, final_type)
    _dump_column_to_hdf5(x, final_type, placeholder, name, index, output)


@_process_column_for_hdf5.register
def _process_factor_column_for_hdf5(x: Factor, name: str, index: int, output: Hdf5ColumnOutput):
    coltype = "factor"
    ordered = x.get_ordered()
    output.columns.append({ "type": coltype, "name": name, "ordered": ordered })

    ghandle = output.handle.create_group(str(index))
    ghandle.attrs.create("type", data=coltype)
    ghandle.attrs.create("ordered", data=ordered, dtype="i1")
    ut._save_fixed_length_strings(ghandle, "levels", x.get_levels())

    curcodes = x.get_codes()
    dhandle = ghandle.create_dataset("codes", data=curcodes, dtype='i4', compression="gzip", chunks=True)
    if (curcodes == -1).any():
        dhandle.attrs.create("missing-value-placeholder", data=-1, dtype='i4')


def _stage_hdf5_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    basename = "simple.h5"
    groupname = "df"

    columns = []
    otherable = []
    full = os.path.join(dir, path, basename)
    with h5py.File(full, "w") as handle:
        ghandle = handle.create_group(groupname)
        ghandle.attrs.create("row-count", data=x.shape[0], dtype="u8")
        ghandle.attrs.create("version", data="1.0")

        dhandle = ghandle.create_group("data")
        output = Hdf5ColumnOutput(handle=dhandle, columns=columns, otherable=otherable)
        for i, col in enumerate(x.get_column_names()):
            _process_column_for_hdf5(x.get_column(col), col, i, output)

        ut._save_fixed_length_strings(ghandle, "column_names", x.get_column_names())
        rn = x.get_row_names()
        has_row_names = rn is not None
        if has_row_names:
            ut._save_fixed_length_strings(ghandle, "row_names", rn)

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
            "group": groupname
        }
    }

    # Running some validation.
    inspected = _inspect_columns(columns, dir)
    lib.check_hdf5_df(
        full, 
        groupname, 
        x.shape[0], 
        has_row_names,
        inspected.name,
        inspected.type,
        inspected.string_format,
        inspected.factor_ordered,
        inspected.factor_levels,
        2,
        2
    )

    return metadata, otherable


########################################################


def _create_csv_column_metadata(name, data_type):
    if data_type == int:
        return { "type": "integer", "name": name }
    elif data_type == float:
        return { "type": "number", "name": name }
    elif data_type == str:
        return { "type": "string", "name": name }
    elif data_type == bool:
        return { "type": "boolean", "name": name }
    else:
        raise NotImplementedError("saving a list of " + str(data_type) + " is not supported yet")


CsvColumnOutput = namedtuple('CsvColumnOutput', ['contents', 'metadata', 'otherable', 'dir', 'path'])


@singledispatch
def _format_column_for_csv(x: Any, name: str, index: int, output: CsvColumnOutput):
    output.metadata.append({ "type": "other", "name": name })
    output.contents[name] = np.zeros(get_height(x), np.int8)
    output.otherable.append(index)


@_format_column_for_csv.register
def _format_list_column_for_csv(x: list, name: str, index: int, output: CsvColumnOutput):
    final_type, has_none = ut._determine_list_type(x)
    if final_type is None:
        _format_column_for_csv.registry[object](x, name, index, output)
    else:
        output.contents[name] = x
        output.metadata.append(_create_csv_column_metadata(name, final_type))


@_format_column_for_csv.register
def _format_StringList_column_for_csv(x: StringList, name: str, index: int, output: CsvColumnOutput):
    output.contents[name] = x
    output.metadata.append({ "type": "string", "name": name });


@_format_column_for_csv.register
def _format_NumPy_column_for_csv(x: np.ndarray, name: str, index: int, output: CsvColumnOutput):
    final_type = ut._determine_numpy_type(x)
    output.contents[name] = x
    output.metadata.append(_create_csv_column_metadata(name, final_type))


@_format_column_for_csv.register
def _format_Factor_column_for_csv(x: Factor, name: str, index: int, output: CsvColumnOutput):
    col_meta = { "type": "factor", "name": name, "ordered": x.get_ordered() }
    lmeta = stage_object(x.get_levels(), output.dir, output.path + "/levels-" + str(index), is_child = True)
    col_meta["levels"] = { "resource": write_metadata(lmeta, output.dir) }
    output.metadata.append(col_meta)

    codes = x.get_codes()
    is_missing = (codes == -1)
    if is_missing.any():
        codes = np.ma.array(codes, mask=is_missing)
    output.contents[name] = codes


def _stage_csv_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    contents = {}
    col_metadata = []
    otherable = []

    output = CsvColumnOutput(contents=contents, metadata=col_metadata, otherable=otherable, dir=dir, path=path)
    for i, col in enumerate(x.get_column_names()):
        _format_column_for_csv(x.get_column(col), col, i, output)

    basename = "simple.csv.gz"
    full = os.path.join(dir, path, basename)

    new_df = BiocFrame(
        contents, 
        number_of_rows = x.shape[0],
        column_names = x.get_column_names(),
        row_names = x.get_row_names(),
    )
    write_csv(new_df, full, compression="gzip")
    has_row_names = (x.get_row_names() is not None)

    metadata = {
        "$schema": "csv_data_frame/v1.json",
        "path": path + "/" + basename,
        "is_child": is_child,
        "data_frame": {
            "columns": col_metadata,
            "row_names": has_row_names,
            "dimensions": list(x.shape),
            "version": 2,
        },
        "csv_data_frame": {
            "compression": "gzip",
        }
    }

    # Running some validation.
    inspected = _inspect_columns(col_metadata, dir)
    lib.check_csv_df(
        full, 
        x.shape[0], 
        has_row_names,
        inspected.name,
        inspected.type,
        inspected.string_format,
        inspected.factor_ordered,
        inspected.factor_levels,
        metadata["data_frame"]["version"],
        metadata["csv_data_frame"]["compression"] == 'gzip',
        True,
    )

    return metadata, otherable


########################################################


ColumnDetails = namedtuple('ColumnDetails', ['name', 'type', 'string_format', 'factor_ordered', 'factor_levels'])


def _inspect_columns(columns: list, dir: str) -> Tuple:
    all_names = [""] * len(columns)
    all_types = [""] * len(columns)
    all_formats = [""] * len(columns)
    all_ordered = np.zeros(len(columns), dtype=np.bool_)
    all_levels = [None] * len(columns)

    for i, x in enumerate(columns):
        all_names[i] = x["name"]
        all_types[i] = x["type"]

        if "format" in x:
            all_formats[i] = x["format"]

        if "ordered" in x:
            all_ordered[i] = x["ordered"]

        all_levels[i] = []
        if "levels" in x:
            meta = acquire_metadata(dir, x["levels"]["resource"]["path"])
            all_levels[i] = alt_load_object(meta, dir)

    return ColumnDetails(all_names, all_types, all_formats, all_ordered, all_levels)
