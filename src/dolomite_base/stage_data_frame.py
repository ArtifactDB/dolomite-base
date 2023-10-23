from typing import Any, Tuple, Optional, Literal
import os
from biocframe import BiocFrame
import numpy as np
import h5py
import gzip

from . import lib_dolomite_base as lib
from .stage_object import stage_object
from .write_metadata import write_metadata
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
        more_meta = stage_object(x.column(i), dir, path + "/child-" + str(i + 1), is_child = True)
        resource_stub = write_metadata(more_meta, dir=dir)
        meta["data_frame"]["columns"][i]["resource"] = resource_stub

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


#######################################################


def _save_fixed_length_strings(handle, name: str, strings: list[str]):
    tmp = [ y.encode("UTF8") for y in strings ]
    maxed = 1
    for b in tmp:
        if len(b) > maxed:
            maxed = len(b)
    return handle.create_dataset(name, data=tmp, dtype="S" + str(maxed), compression="gzip", chunks=True)


def _process_columns(x: BiocFrame, handle) -> Tuple:
    columns = []
    otherable = []
    operations = []
    as_csv = handle is None

    # TODO: handle date-times, pandas' Categorical factors.
    for i, col in enumerate(x.column_names):
        current = x.column(col)
        redirect = False
        final_type = bool

        if isinstance(current, list):
            all_types = set([type(y) for y in current])
            has_none = False
            if type(None) in all_types:
                has_none = True
                all_types.remove(type(None))

            if len(all_types) == 1:
                final_type = list(all_types)[0]
            elif len(all_types) == 2 and int in all_types and float in all_types:
                final_type = float
            else:
                columns.append({ "type": "other", "name": col })
                operations.append(lambda x : 0)
                otherable.append(i)
                continue

            if final_type == int:
                if not ut._is_integer_vector_within_limit(current):
                    final_type = float    

            if not as_csv:
                if has_none:
                    placeholder, final_type = _select_hdf5_placeholder(current, final_type)
                    copy = current[:]
                    for j, y in enumerate(copy):
                        if y is None:
                            copy[j] = placeholder 
                    current = copy

            if final_type == int:
                columns.append({ "type": "integer", "name": col })
                if as_csv:
                    operations.append(_list_element_to_string)
                else:
                    savetype = 'i4'
            elif final_type == float:
                columns.append({ "type": "number", "name": col })
                if as_csv:
                    operations.append(_list_element_to_string)
                else:
                    savetype = 'f8'
            elif final_type == str:
                columns.append({ "type": "string", "name": col })
                if as_csv:
                    operations.append(_quotify_string_or_none)
                else:
                    savetype = None
            elif final_type == bool:
                columns.append({ "type": "boolean", "name": col })
                if as_csv:
                    operations.append(_list_element_to_string)
                else:
                    savetype = 'i1'
            else:
                raise NotImplementedError("saving a list of " + str(list(all_types)[0]) + " is not supported yet")

            if not as_csv:
                if savetype: 
                    dhandle = handle.create_dataset(str(i), data=current, dtype=savetype, compression="gzip", chunks=True)
                else:
                    dhandle = _save_fixed_length_strings(handle, str(i), current)
                if has_none:
                    dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype=savetype)

        elif isinstance(current, np.ndarray):
            dt = current.dtype.type
            final_type = None

            if issubclass(dt, np.integer):
                if ut._is_integer_vector_within_limit(current):
                    final_type = int
                else:
                    final_type = float
            elif issubclass(dt, np.floating):
                final_type = float
            elif issubclass(dt, np.bool_):
                final_type = bool
            else:
                raise NotImplementedError("saving a NumPy array of " + str(current.dtype) + " is not supported yet")

            placeholder = None
            if np.ma.is_masked(current):
                if as_csv:
                    operations.append(_numpy_element_to_string)
                else:
                    placeholder, final_type = _select_hdf5_placeholder(current, final_type)
                    if final_type == int:
                        current = ut._fill_integer_missing_placeholder(current, placeholder)
                    elif final_type == float:
                        current = ut._fill_float_missing_placeholder(current, placeholder)
                    elif final_type == bool:
                        current = ut._fill_boolean_missing_placeholder(current, placeholder)
                    else:
                        raise NotImplementedError("saving a masked NumPy array as " + str(final_type) + " to HDF5 is not supported yet")
            else:
                if as_csv:
                    operations.append(str)

            if final_type == int:
                columns.append({ "type": "integer", "name": col })
            elif final_type == float:
                columns.append({ "type": "number", "name": col })
            elif final_type == bool:
                columns.append({ "type": "boolean", "name": col })
            else:
                raise NotImplementedError("saving a NumPy array as " + str(final_type) + " is not supported yet")

            if not as_csv:
                if final_type == int:
                    savetype = 'i4'
                elif final_type == float:
                    savetype = 'f8'
                elif final_type == bool:
                    savetype = 'i1'
                else:
                    raise NotImplementedError("saving a NumPy array of " + str(final_type) + " to HDF5 is not supported yet")
                dhandle = handle.create_dataset(str(i), data=current, dtype=savetype, compression="gzip", chunks=True)
                if placeholder:
                    dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype=savetype)

        else:
            columns.append({ "type": "other", "name": col })
            otherable.append(i)
            if as_csv:
                operations.append(lambda x : "0")

    return columns, otherable, operations


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


#######################################################


def _select_hdf5_placeholder(current, dtype) -> Tuple:
    if dtype == float:
        placeholder = ut._choose_float_missing_placeholder()
    elif dtype == int:
        placeholder = ut._choose_integer_missing_placeholder(current)
        if placeholder is None: # fallback if no placeholder can be picked.
            return ut._choose_float_missing_placeholder(), float
    elif dtype == str:
        placeholder = ut._choose_string_missing_placeholder(current)
    elif dtype == bool:
        placeholder = ut._choose_boolean_missing_placeholder()
    else:
        raise NotImplementedError("saving a list of " + str(dtype) + " is not supported yet")
    return placeholder, dtype


def _stage_hdf5_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    basename = "simple.h5"
    groupname = "df"

    full = os.path.join(dir, path, basename)
    with h5py.File(full, "w") as handle:
        ghandle = handle.create_group(groupname)
        dhandle = ghandle.create_group("data")
        columns, otherable, operations = _process_columns(x, dhandle)
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


#######################################################


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


def _stage_csv_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    columns, otherable, operations = _process_columns(x, None)
    nr = x.shape[0]

    extracted_row_names = x.row_names
    has_row_names = extracted_row_names is not None
    extracted_columns = []
    for y in x.column_names:
        extracted_columns.append(x.column(y))

    # Manual serialization into a Gzip-compressed CSV, because 
    # pandas doesn't quite give me what I want... oh well.
    basename = "simple.csv.gz"
    full = os.path.join(dir, path, basename)

    with gzip.open(full, "wb") as handle:
        header_line = ""
        if has_row_names:
            header_line += _quotify_string("row_names")
        for c in x.column_names:
            if header_line:
                header_line += ","
            header_line += _quotify_string(c)
        header_line += "\n"
        handle.write(header_line.encode("ASCII"))

        for r in range(nr):
            if has_row_names:
                line = _quotify_string(extracted_row_names[r])
            else:
                line = ""
            for i, trans in enumerate(operations):
                if line:
                    line += ","
                line += trans(extracted_columns[i][r])
            line += "\n"
            handle.write(line.encode("ASCII"))

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
