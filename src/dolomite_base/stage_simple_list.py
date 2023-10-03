from typing import Any, Union, Optional, Literal
import numpy as np
from numpy import ndarray, issubdtype, integer, floating, bool_
import os
import json
import gzip

from .stage_object import stage_object
from ._stage_csv_data_frame import _stage_csv_data_frame
from .write_metadata import write_metadata
from . import _cpphelpers as lib


@stage_object.register
def stage_simple_dict(
    x: dict,
    dir: str, 
    path: str, 
    is_child: bool = False, 
    mode: Optional[Literal["hdf5", "json"]] = None,
    **kwargs
) -> dict[str, Any]:
    """Method for saving dictionaries (Python analogues to R-style named lists)
    to the corresponding file representations, see
    :py:meth:`~dolomite_base.stage_object.stage_object` for details.

    Args:
        x: Object to be staged.

        dir: Staging directory.

        path: Relative path inside ``dir`` to save the object.

        is_child: Is ``x`` a child of another object?

        mode: Whether to save in HDF5 or JSON mode.
            If None, defaults to JSON.

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    return _stage_simple_list_internal(x, dir, path, is_child, mode, **kwargs)


@stage_object.register
def stage_simple_list(
    x: list,
    dir: str, 
    path: str, 
    is_child: bool = False, 
    mode: Optional[Literal["hdf5", "json"]] = None,
    **kwargs
) -> dict[str, Any]:
    """Method for saving lists (Python analogues to R-style unnamed lists) to
    the corresponding file representations, see
    :py:meth:`~dolomite_base.stage_object.stage_object` for details.

    Args:
        x: Object to be staged.

        dir: Staging directory.

        path: Relative path inside ``dir`` to save the object.

        is_child: Is ``x`` a child of another object?

        mode: Whether to save in HDF5 or JSON mode.
            If None, defaults to JSON.

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    return _stage_simple_list_internal(x, dir, path, is_child, mode, **kwargs)


def _stage_simple_list_internal(
    x: Union[dict, list],
    dir: str, 
    path: str, 
    is_child: bool = False, 
    mode: Optional[Literal["hdf5", "json"]] = None,
    **kwargs
) -> dict[str, Any]:

    externals = []
    os.mkdir(os.path.join(dir, path))
    components = {}

    if mode is None or mode == "json":
        transformed = _stage_simple_list_json(x, externals)
        transformed["version"] = "1.1"

        newpath = path + "/list.json.gz"
        opath = os.path.join(dir, newpath)
        with gzip.open(opath, "wt") as handle:
            json.dump(transformed, handle)

        lib.validate_list_json(opath.encode("UTF8"), len(externals))

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
        typecheck = set()
        for y in x:
            if y is not None:
                typecheck.add(type(y))

        if len(typecheck) == 1:
            if str in typecheck:
                return { "type": "string", "values": x }

        vals = []
        collected = { "type": "list", "values": vals }
        for i, y in enumerate(x):
            vals.append(_stage_simple_list_json(y, externals))
        return collected

    elif isinstance(x, dict):
        vals = []
        names = []
        collected = { "type": "list", "values": vals, "names": names }
        for k, v in x.items():
            names.append(k)
            vals.append(_stage_simple_list_json(v, externals))
        return collected

    elif isinstance(x, bool): # bools are ints, so put this before the int check.
        return { "type": "boolean", "values": x }

    elif isinstance(x, int):
        return { "type": "integer", "values": x }

    elif isinstance(x, str):
        return { "type": "string", "values": x }

    elif isinstance(x, float):
        return { "type": "number", "values": x }

    elif isinstance(x, ndarray):
        if len(x.shape) == 1:
            if np.ma.is_masked(x):
                if issubdtype(x.dtype, integer):
                    return { "type": "integer", "values": [None if np.ma.is_masked(y) else int(y) for y in x] }
                elif issubdtype(x.dtype, floating):
                    return { "type": "number", "values": [None if np.ma.is_masked(y) else float(y) for y in x] }
                elif x.dtype == bool_:
                    return { "type": "boolean", "values": [None if np.ma.is_masked(y) else bool(y) for y in x] }
                else:
                    raise NotImplementedError("no staging method for 1D NumPy masked arrays of " + str(x.dtype))
            else:
                if issubdtype(x.dtype, integer):
                    return { "type": "integer", "values": [int(y) for y in x] }
                elif issubdtype(x.dtype, floating):
                    return { "type": "number", "values": [float(y) for y in x] }
                elif x.dtype == bool_:
                    return { "type": "boolean", "values": [bool(y) for y in x] }
                else:
                    raise NotImplementedError("no staging method for 1D NumPy arrays of " + str(x.dtype))

        elif len(x.shape) == 0:
            if np.ma.is_masked(x):
                if issubdtype(x.dtype, integer):
                    return { "type": "integer", "values": None if np.ma.is_masked(y) else int(y) }
                elif issubdtype(x.dtype, floating):
                    return { "type": "number", "values": None if np.ma.is_masked(y) else float(y) }
                elif x.dtype == bool_:
                    return { "type": "boolean", "values": None if np.ma.is_masked(y) else bool(y) }
                else:
                    raise NotImplementedError("no staging method for 0-d NumPy arrays of " + str(x.dtype))
            else:
                if issubdtype(x.dtype, integer):
                    return { "type": "integer", "values": int(x) } 
                elif issubdtype(x.dtype, floating):
                    return { "type": "number", "values": float(x) }
                elif x.dtype == bool_:
                    return { "type": "boolean", "values": bool(x) }
                else:
                    raise NotImplementedError("no staging method for 0-d NumPy arrays of " + str(x.dtype))

    elif isinstance(x, np.generic):
        if isinstance(x, integer):
            return { "type": "integer", "values": int(x) }
        elif isinstance(x, floating):
            return { "type": "number", "values": float(x) }
        elif isinstance(x, bool_):
            return { "type": "boolean", "values": bool(x) }
        else:
            raise NotImplementedError("no staging method for NumPy array scalars of " + str(x.dtype))


    elif x == None:
        return { "type": "nothing" }

    externals.append(x)
    return { "type": "other", "index": len(externals) - 1 }
