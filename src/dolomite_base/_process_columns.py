from biocframe import BiocFrame
import numpy as np
from . import _utils as ut
from typing import Tuple


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
        final_type = bool
        is_other = False

        if isinstance(current, np.ndarray):
            final_type = ut._determine_numpy_type(current)
            if np.ma.is_masked(current):
                operations.append(_numpy_element_to_string)
            else:
                operations.append(str)
        elif isinstance(current, list):
            final_type, has_none = ut._determine_list_type(current)
            if final_type is None:
                is_other = True
            elif final_type == str:
                operations.append(_quotify_string_or_none)
            else:
                operations.append(_list_element_to_string)
        else:
            is_other = True

        if is_other:
            if isinstance(current, Factor):
                columns.append({ "type": "factor", "name": col, "ordered": current.ordered })
                meta = ???? ## HERE!! ###
                columns[-1]["levels"] = { "resource": write_metadata(meta, dir) }
            else:
                columns.append({ "type": "other", "name": col })
                operations.append(lambda x : "0")
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

    for i, col in enumerate(x.column_names):
        current = x.column(col)
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
                columns.append({ "type": "factor", "name": col, "ordered": current.ordered })
                ghandle = handle.create_group(str(i))
                ghandle.attrs.create("type", data=columns[-1]["type"])
                ghandle.attrs.create("ordered", data=int(columns[-1]["ordered"]), dtype="i1")
                ut._save_fixed_length_strings(ghandle, "levels", current.levels)

                codes = current.codes
                has_none = any(y is None for y in codes)
                if has_none:
                    codes = codes[:]
                    for i, y in enumerate(codes):
                        if y is None:
                            codes[i] = -1
                dhandle = ghandle.create_dataset("codes", data=codes, dtype='i4', compression="gzip", chunks=True)
                if has_none:
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
