from typing import Any
from numpy import ndarray
import os
import json
import gzip

from .stage_object import stage_object
from ._stage_csv_data_frame import _stage_csv_data_frame
from .write_metadata import write_metadata
from . import _cpphelpers as lib


@stage_object.register
def stage_simple_list(x: list, dir: str, path: str, is_child: bool = False, mode: Literal["hdf5", "json"] = "json", **kwargs) -> dict[str, Any]:
    externals = []
    os.mkdir(os.path.join(dir, path))
    components = {}

    if mode == "json":
        transformed = _stage_simple_list_json(x, externals)
        transformed["version"] = "1.1"
        newpath = path + "/list.json.gz"
        opath = os.path.join(dir, newpath)
        with gzip.open(opath, "wb") as handle:
            json.dump(transformed, handle)
        lib.validate_list_json(opath)

        components["$schema"] = "json_simple_list/v1.json"
        components["path"] = newpath
        components["json_simple_list"] = { "compression": "gzip" }
    else:
        raise NotImplementedError("staging into HDF5 lists is not supported yet")

    children = []
    for ex in externals:
        children.append({ "resource": stage_object(ex) })
    components["simple_list"] = { "children": children }
    components["is_child"] = is_child
    return components


def _stage_simple_list_json(x, externals):
    if isinstance(x, list):
        typecheck = set([ type(y) for y in x ])
        curtype = "list"
        if len(typecheck) == 1:
            if str in typecheck:
                curtype = "string"

        vals = []
        collected = { "type": curtype, "values": vals }
        for i, y in enumerate(x):
            vals.append(_stage_simple_list_json(y, transformed, externals))
        return collected

    elif isinstance(x, dict):
        vals = []
        names = []
        collected = { "type": "list", "values": vals, "names": names }
        for k, v in x.items():
            names.append(k)
            vals.append(v)
        return collected

    elif isinstance(x, int):
        return { "type": "integer", "values": x }

    elif isinstance(x, str):
        return { "type": "string", "values": x }

    elif isinstance(x, float):
        return { "type": "number", "values": x }

    elif isinstance(x, bool):
        return { "type": "boolean", "values": x }

    elif isinstance(x, ndarray) and len(x.shape) == 1:
        if issubdtype(x.dtype, integer):
            return { "type": "integer", "values": list(x) }
        elif issubdtype(x.dtype, floating):
            return { "type": "number", "values": list(x) }
        elif x.dtype == bool_:
            return { "type": "boolean", "values": list(x) }
        else:
            raise NotImplementedError("no staging method for NumPy arrays of " + str(x.dtype))

    elif x == None:
        return { "type": "nothing" }

    externals.append(x)
    return { "type": "other", "index": len(externals) - 1 }
