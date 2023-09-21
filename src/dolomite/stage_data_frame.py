from typing import Any, Tuple
from biocframe import BiocFrame
import numpy as np
import os
import gzip

from .stage_object import stage_object
from .write_metadata import write_metadata


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


def _process_columns(x: BiocFrame) -> Tuple:
    columns = []
    otherable = []
    operations = []

    # TODO: handle date-times, pandas' Categorical factors.
    for i, col in enumerate(x.column_names):
        current = x.column(col)
        redirect = False
        final_type = bool

        if isinstance(current, list):
            all_types = set([type(y) for y in current])
            if type(None) in all_types:
                all_types.remove(type(None))

            if len(all_types) == 1:
                final_type = list(all_types)[0]
            elif len(all_types) == 2 and int in all_types and float in all_types:
                final_type = float
            else:
                redirect = True

            if redirect:
                columns.append({ "type": "other", "name": col })
                operations.append(lambda x : 0)
                otherable.append(i)
            elif final_type == int:
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
            dt = current.dtype.type
            if issubclass(dt, np.integer):
                columns.append({ "type": "integer", "name": col })
            elif issubclass(dt, np.floating):
                columns.append({ "type": "number", "name": col })
            elif issubclass(dt, np.bool_):
                columns.append({ "type": "boolean", "name": col })
            else:
                raise NotImplementedError("saving a NumPy array of " + str(current.dtype) + " is not supported yet")

            if np.ma.is_masked(current):
                operations.append(_numpy_element_to_string)
            else:
                operations.append(str)

        else:
            columns.append({ "type": "other", "name": col })
            operations.append(lambda x : 0)
            otherable.append(i)

    return columns, otherable, operations


def _stage_data_frame_csv(x: BiocFrame, dir: str, path: str, is_child: bool) -> Tuple:
    columns, otherable, operations = _process_columns(x)
    nr = x.shape[0]

    extracted_row_names = x.row_names
    has_row_names = extracted_row_names is not None
    extracted_columns = []
    for y in x.column_names:
        extracted_columns.append(x.column(y))

    # Manual serialization into a Gzip-compressed CSV, because 
    # pandas doesn't quite give me what I want... oh well.
    newpath = os.path.join(path, "simple.csv.gz")
    full = os.path.join(dir, newpath)

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
        "path": newpath,
        "is_child": is_child,
        "data_frame": {
            "columns": columns,
            "row_names": x.row_names is not None,
            "dimensions": list(x.shape),
        },
        "csv_data_frame": {
            "compression": "gzip",
        }
    }

    return metadata, otherable


@stage_object.register
def stage_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool = False, **kwargs) -> dict[str, Any]:
    os.mkdir(os.path.join(dir, path))
    meta, other = _stage_data_frame_csv(x, dir, path, is_child=is_child)

    for i in other:
        more_meta = stage_object(x.column(i), dir, path, is_child = True)
        resource_stub = write_metadata(more_meta, dir=dir)
        meta["data_frame"]["columns"][i]["resource"] = resource_stub
    return meta
