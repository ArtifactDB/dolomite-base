from typing import Any, Tuple, Optional, Literal
from collections import namedtuple
import os
from biocframe import BiocFrame
from biocutils import Factor
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


def _process_columns_for_hdf5(x: BiocFrame, handle) -> Tuple:
    columns = []
    otherable = []

    for i, col in enumerate(x.get_column_names()):
        current = x.get_column(col)
        placeholder = None
        final_type = None
        is_other = False

        if isinstance(current, np.ndarray):
            final_type = ut._determine_numpy_type(current)
            if np.ma.is_masked(current):
                current, placeholder, final_type = _select_hdf5_placeholder(current, final_type)
        elif isinstance(current, list):
            final_type, has_none = ut._determine_list_type(current)
            if final_type is None:
                is_other = True
            elif has_none:
                current, placeholder, final_type = _select_hdf5_placeholder(current, final_type)
        else:
            is_other = True

        if is_other:
            if isinstance(current, Factor):
                coltype = "factor"
                ordered = current.get_ordered()
                columns.append({ "type": coltype, "name": col, "ordered": ordered })

                ghandle = handle.create_group(str(i))
                ghandle.attrs.create("type", data=coltype)
                ghandle.attrs.create("ordered", data=ordered, dtype="i1")
                ut._save_fixed_length_strings(ghandle, "levels", current.get_levels())

                curcodes = current.get_codes()
                dhandle = ghandle.create_dataset("codes", data=curcodes, dtype='i4', compression="gzip", chunks=True)
                if (curcodes == -1).any():
                    dhandle.attrs.create("missing-value-placeholder", data=-1, dtype='i4')
            else:
                columns.append({ "type": "other", "name": col })
                otherable.append(i)
        else:
            if final_type == int:
                columns.append({ "type": "integer", "name": col })
                savetype = 'i4'
            elif final_type == float:
                columns.append({ "type": "number", "name": col })
                savetype = 'f8'
            elif final_type == str:
                columns.append({ "type": "string", "name": col })
                savetype = None
            elif final_type == bool:
                columns.append({ "type": "boolean", "name": col })
                savetype = 'i1'
            else:
                raise NotImplementedError("saving a list of " + str(final_type) + " is not supported yet")

            if savetype: 
                dhandle = handle.create_dataset(str(i), data=current, dtype=savetype, compression="gzip", chunks=True)
            else:
                dhandle = ut._save_fixed_length_strings(handle, str(i), current)
            dhandle.attrs.create("type", data=columns[-1]["type"])
            if placeholder:
                dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype=savetype)

    return columns, otherable


def _stage_hdf5_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    basename = "simple.h5"
    groupname = "df"

    full = os.path.join(dir, path, basename)
    with h5py.File(full, "w") as handle:
        ghandle = handle.create_group(groupname)
        ghandle.attrs.create("row-count", data=x.shape[0], dtype="u8")
        ghandle.attrs.create("version", data="1.0")
        dhandle = ghandle.create_group("data")
        columns, otherable = _process_columns_for_hdf5(x, dhandle)

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


def _stage_csv_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    columns = []
    contents = {}
    otherable = []

    # TODO: handle dates, date-times, pandas' Categorical factors.
    for i, col in enumerate(x.get_column_names()):
        current = x.get_column(col)
        final_type = bool
        is_other = False

        if isinstance(current, np.ndarray):
            final_type = ut._determine_numpy_type(current)
        elif isinstance(current, list):
            final_type, has_none = ut._determine_list_type(current)
            if final_type is None:
                is_other = True
        else:
            is_other = True

        if is_other:
            if isinstance(current, Factor):
                columns.append({ "type": "factor", "name": col, "ordered": current.ordered })
                lmeta = stage_object(current.get_levels(), dir, path + "/levels-" + str(i), is_child = True)
                columns[-1]["levels"] = { "resource": write_metadata(lmeta, dir) }

                codes = current.get_codes()
                is_missing = (codes == -1)
                if is_missing.any():
                    codes = np.ma.array(codes, mask=is_missing)
                x = x.set_column(col, codes)
            else:
                columns.append({ "type": "other", "name": col })
                x = x.set_column(col, np.zeros(len(current), np.int8))
                otherable.append(i)
        else:
            if final_type == int:
                columns.append({ "type": "integer", "name": col })
            elif final_type == float:
                columns.append({ "type": "number", "name": col })
            elif final_type == str:
                columns.append({ "type": "string", "name": col })
            elif final_type == bool:
                columns.append({ "type": "boolean", "name": col })
            else:
                raise NotImplementedError("saving a list of " + str(final_type) + " is not supported yet")

    basename = "simple.csv.gz"
    full = os.path.join(dir, path, basename)
    write_csv(x, full, compression="gzip")
    has_row_names = (x.get_row_names() is not None)

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
    inspected = _inspect_columns(columns, dir)
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
