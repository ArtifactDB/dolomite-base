from biocframe import BiocFrame
import numpy as np
from . import _utils as ut
from typing import Tuple


def _determine_list_type(x: list):
    all_types = set([type(y) for y in x])
    has_none = False
    if type(None) in all_types:
        has_none = True
        all_types.remove(type(None))

    final_type = None
    if len(all_types) == 1:
        final_type = list(all_types)[0]
        if final_type == int:
            if not ut._is_integer_vector_within_limit(x):
                final_type = float 
        elif final_type != str and final_type != bool and final_type != float:
            raise NotImplementedError("don't know how to save a list of " + str(final_type) + " objects")
    elif len(all_types) == 2 and int in all_types and float in all_types:
        final_type = float
    elif len(all_types) == 0: # if all None, this is the fallback.
        final_type = str

    return final_type, has_none


def _determine_numpy_type(x: np.ndarray):
    dt = x.dtype.type
    final_type = None

    if issubclass(dt, np.integer):
        if ut._is_integer_vector_within_limit(x):
            return int
        else:
            return float
    elif issubclass(dt, np.floating):
        return float
    elif issubclass(dt, np.bool_):
        return bool
    else:
        raise NotImplementedError("saving a NumPy array of " + str(x.dtype) + " is not supported yet")


########################################################


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


def _process_columns_for_csv(x: BiocFrame) -> Tuple:
    columns = []
    otherable = []
    operations = []

    # TODO: handle date-times, pandas' Categorical factors.
    for i, col in enumerate(x.column_names):
        current = x.column(col)
        redirect = False
        final_type = bool

        if isinstance(current, list):
            final_type, has_none = _determine_list_type(current)

            if final_type is None:
                columns.append({ "type": "other", "name": col })
                operations.append(lambda x : 0)
                otherable.append(i)
                continue

            if final_type == int:
                columns.append({ "type": "integer", "name": col })
                operations.append(_list_element_to_string)
            elif final_type == float:
                columns.append({ "type": "number", "name": col })
                operations.append(_list_element_to_string)
            elif final_type == str:
                columns.append({ "type": "string", "name": col })
                operations.append(_quotify_string_or_none)
            elif final_type == bool:
                columns.append({ "type": "boolean", "name": col })
                operations.append(_list_element_to_string)
            else:
                raise NotImplementedError("saving a list of " + str(list(all_types)[0]) + " is not supported yet")

        elif isinstance(current, np.ndarray):
            final_type = _determine_numpy_type(current)

            placeholder = None
            if np.ma.is_masked(current):
                operations.append(_numpy_element_to_string)
            else:
                operations.append(str)

            if final_type == int:
                columns.append({ "type": "integer", "name": col })
            elif final_type == float:
                columns.append({ "type": "number", "name": col })
            elif final_type == bool:
                columns.append({ "type": "boolean", "name": col })
            else:
                raise NotImplementedError("saving a NumPy array as " + str(final_type) + " is not supported yet")

        else:
            columns.append({ "type": "other", "name": col })
            otherable.append(i)
            operations.append(lambda x : "0")

    return columns, otherable, operations


def _write_csv(x: BiocFrame, handle, operations: list):
    nr = x.shape[0]

    extracted_row_names = x.row_names
    has_row_names = extracted_row_names is not None
    extracted_columns = []
    for y in x.column_names:
        extracted_columns.append(x.column(y))

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


########################################################


def _save_fixed_length_strings(handle, name: str, strings: list[str]):
    tmp = [ y.encode("UTF8") for y in strings ]
    maxed = 1
    for b in tmp:
        if len(b) > maxed:
            maxed = len(b)
    return handle.create_dataset(name, data=tmp, dtype="S" + str(maxed), compression="gzip", chunks=True)


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


def _process_columns_for_hdf5(x: BiocFrame, handle) -> Tuple:
    columns = []
    otherable = []

    # TODO: handle date-times, pandas' Categorical factors.
    for i, col in enumerate(x.column_names):
        current = x.column(col)
        redirect = False
        final_type = bool

        if isinstance(current, list):
            final_type, has_none = _determine_list_type(current)

            if final_type is None:
                columns.append({ "type": "other", "name": col })
                otherable.append(i)
                continue

            if has_none:
                placeholder, final_type = _select_hdf5_placeholder(current, final_type)
                copy = current[:]
                for j, y in enumerate(copy):
                    if y is None:
                        copy[j] = placeholder 
                current = copy

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
                raise NotImplementedError("saving a list of " + str(list(all_types)[0]) + " is not supported yet")

            if savetype: 
                dhandle = handle.create_dataset(str(i), data=current, dtype=savetype, compression="gzip", chunks=True)
            else:
                dhandle = _save_fixed_length_strings(handle, str(i), current)
            dhandle.attrs.create("type", data=columns[-1]["type"])
            if has_none:
                dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype=savetype)

        elif isinstance(current, np.ndarray):
            final_type = _determine_numpy_type(current)

            placeholder = None
            if np.ma.is_masked(current):
                placeholder, final_type = _select_hdf5_placeholder(current, final_type)
                if final_type == int:
                    current = ut._fill_integer_missing_placeholder(current, placeholder)
                elif final_type == float:
                    current = ut._fill_float_missing_placeholder(current, placeholder)
                elif final_type == bool:
                    current = ut._fill_boolean_missing_placeholder(current, placeholder)
                else:
                    raise NotImplementedError("saving a masked NumPy array as " + str(final_type) + " to HDF5 is not supported yet")

            if final_type == int:
                columns.append({ "type": "integer", "name": col })
                savetype = 'i4'
            elif final_type == float:
                columns.append({ "type": "number", "name": col })
                savetype = 'f8'
            elif final_type == bool:
                columns.append({ "type": "boolean", "name": col })
                savetype = 'i1'
            else:
                raise NotImplementedError("saving a NumPy array as " + str(final_type) + " is not supported yet")

            dhandle = handle.create_dataset(str(i), data=current, dtype=savetype, compression="gzip", chunks=True)
            dhandle.attrs.create("type", data=columns[-1]["type"])
            if placeholder:
                dhandle.attrs.create("missing-value-placeholder", data=placeholder, dtype=savetype)

        else:
            columns.append({ "type": "other", "name": col })
            otherable.append(i)

    return columns, otherable
