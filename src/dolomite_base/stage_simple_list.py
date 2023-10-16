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
from ._utils import choose_missing_placeholder


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


##########################################################################


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
        transformed = _stage_simple_list_recursive(x, externals, None)
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
        newpath = path + "/list.h5"
        opath = os.path.join(dir, newpath)
        with h5py.File(newpath, "w") as handle:
            ghandle = handle.create_group("data")
            ghandle.attrs["uzuki_version"] = "1.1"
            _stage_simple_list_recursive(x, externals, ghandle)

        lib.validate_list_hdf5(opath.encode("UTF8"), len(externals))

        components["$schema"] = "hdf5_simple_list/v1.json"
        components["path"] = newpath
        components["hdf5_simple_list"] = { "group": "data" }

    children = []
    for ex in externals:
        children.append({ "resource": stage_object(ex) })
    components["simple_list"] = { "children": children }
    components["is_child"] = is_child
    return components


def _stage_simple_list_recursive(x, externals, handle):
    if isinstance(x, list):
        typecheck = set()
        has_none = False
        for y in x:
            if y is not None:
                has_none = True
                typecheck.add(type(y))

        if len(typecheck) == 1:
            if str in typecheck:
                if handle is None:
                    return { "type": "string", "values": x }
                else:
                    if has_none:
                        missing_placeholder = _choose_missing_placeholder(x)
                        new_x = x[:]
                        for i, y in enumerate(new_x):
                            if y is None:
                                new_x[i] = missing_placeholder
                        x = new_x

                    dset = handle.create_dataset("data", data=x, compression="gzip", chunks=True)
                    dset.attrs["uzuki_object"] = "vector"
                    dset.attrs["uzuki_type"] = "string"

                    if has_none:
                        dset.attrs["missing-value-placeholder"] = missing_placeholder
                    return

        if handle is None:
            vals = []
            collected = { "type": "list", "values": vals }
            for i, y in enumerate(x):
                vals.append(_stage_simple_list_recursive(y, externals, None))
            return collected
        else:
            handle.attrs["uzuki_object"] = "list"
            dhandle = handle.create_group("data")
            for i, y in enumerate(x):
                ghandle = dhandle.create_group(str(i))
                _stage_simple_list_recursive(y, externals, ghandle)

    elif isinstance(x, dict):
        if handle is None:
            vals = []
            names = []
            collected = { "type": "list", "values": vals, "names": names }
            for k, v in x.items():
                names.append(k)
                vals.append(_stage_simple_list_recursive(v, externals, None))
            return collected
        else:
            handle.attrs["uzuki_object"] = "list"
            dhandle = handle.create_group("data")
            names = []
            for k, v in x.items():
                ghandle = dhandle.create_group(str(len(names)))
                _stage_simple_list_recursive(v, externals, ghandle)
                names.append(k)
            handle.create_dataset("names", data=names, compression="gzip", chunks=True)
            return

    elif isinstance(x, bool): # bools are ints, so put this before the int check.
        if handle is None:
            return { "type": "boolean", "values": x }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=bool)
            return

    elif isinstance(x, int):
        if handle is None:
            return { "type": "integer", "values": x }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=int)
            return

    elif isinstance(x, str):
        if handle is None:
            return { "type": "string", "values": x }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=str)
            return

    elif isinstance(x, float):
        if handle is None:
            return { "type": "number", "values": x }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=float)
            return

    elif isinstance(x, ndarray):
        if len(x.shape) == 1:
            if np.ma.is_masked(x):
                as_float = False
                if issubdtype(x.dtype, integer):
                    if _is_integer_vector_within_limit(x):
                        if handle is None:
                            return { "type": "integer", "values": [None if np.ma.is_masked(y) else int(y) for y in x] }
                        else:
                            # If there's no valid missing placeholder, we just save it as floating-point.
                            missing_placeholder = _fetch_integer_missing_placeholder(x)
                            if missing_placeholder is not None:
                                _stage_vector_hdf5(handle, x=_fill_integer_missing_placeholder(x, missing_placeholder), dtype=int, missing_placeholder=missing_placeholder)
                                return
                    as_float = True

                if as_float or issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": [None if np.ma.is_masked(y) else float(y) for y in x] }
                    else:
                        missing_placeholder = _fetch_float_missing_placeholder()
                        x = _fill_float_missing_placeholder(x, missing_placeholder)
                        _stage_vector_hdf5(handle, x=x, dtype=float, missing_placeholder=missing_placeholder)
                        return
                elif x.dtype == bool_:
                    if handle is None:
                        return { "type": "boolean", "values": [None if np.ma.is_masked(y) else bool(y) for y in x] }
                    else:
                        missing_placeholder = _fetch_boolean_missing_placeholder()
                        x = _fill_boolean_missing_placeholder(x, missing_placeholder)
                        _stage_vector_hdf5(handle, x=x, dtype=bool, missing_placeholder=missing_placeholder)
                        return
                else:
                    raise NotImplementedError("no staging method for 1D NumPy masked arrays of " + str(x.dtype))

            else:
                as_float = False
                if issubdtype(x.dtype, integer):
                    if _is_integer_vector_within_limit(x):
                        if handle is None:
                            return { "type": "integer", "values": [int(y) for y in x] }
                        else:
                            _stage_vector_hdf5(handle, x=x, dtype=int)
                            return
                    as_float = True

                if as_float or issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": [float(y) for y in x] }
                    else:
                        _stage_vector_hdf5(handle, x=x, dtype=float)
                        return
                elif x.dtype == bool_:
                    if handle is None:
                        return { "type": "boolean", "values": [bool(y) for y in x] }
                    else:
                        _stage_vector_hdf5(handle, x=x, dtype=bool)
                        return
                else:
                    raise NotImplementedError("no staging method for 1D NumPy arrays of " + str(x.dtype))

        elif len(x.shape) == 0:
            if np.ma.is_masked(x):
                if issubdtype(x.dtype, integer):
                    if handle is None:
                        return { "type": "integer", "values": None if np.ma.is_masked(x) else int(x) }
                    else:
                        missing_placeholder = _fetch_integer_missing_placeholder([])
                        _stage_scalar_hdf5(handle, x=missing_placeholder, dtype=int, missing_placeholder=missing_placeholder)
                        return
                elif issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": None if np.ma.is_masked(x) else float(x) }
                    else:
                        missing_placeholder = _fetch_float_missing_placeholder()
                        _stage_scalar_hdf5(handle, x=missing_placeholder, dtype=float, missing_placeholder=missing_placeholder)
                        return
                elif x.dtype == bool_:
                    if handle is None:
                        return { "type": "boolean", "values": None if np.ma.is_masked(x) else bool(x) }
                    else:
                        missing_placeholder = _fetch_float_missing_placeholder()
                        _stage_scalar_hdf5(handle, x=missing_placeholder, dtype=float, missing_placeholder=missing_placeholder)
                        return
                else:
                    raise NotImplementedError("no staging method for 0-d NumPy arrays of " + str(x.dtype))

            else:
                as_float = False
                if issubdtype(x.dtype, integer):
                    if _is_integer_vector_within_limit(x):
                        if handle is None:
                            return { "type": "integer", "values": int(x) } 
                        else:
                            _stage_scalar_hdf5(handle, x=x, dtype=int)
                            return
                    as_float = True

                if as_float or issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": float(x) }
                    else:
                        _stage_scalar_hdf5(handle, x=x, dtype=float)
                        return
                elif x.dtype == bool_:
                    if handle is None:
                        return { "type": "boolean", "values": bool(x) }
                    else:
                        _stage_scalar_hdf5(handle, x=x, dtype=bool)
                        return
                else:
                    raise NotImplementedError("no staging method for 0-d NumPy arrays of " + str(x.dtype))

    elif isinstance(x, np.generic):
        as_float = False
        if isinstance(x, integer):
            if _is_integer_scalar_within_limit(x):
                if handle is None:
                    return { "type": "integer", "values": int(x) }
                else:
                    _stage_scalar_hdf5(handle, x=x, dtype=int)
                    return
            as_float = True

        if as_float or isinstance(x, floating):
            if handle is None:
                return { "type": "number", "values": float(x) }
            else:
                _stage_scalar_hdf5(handle, x=x, dtype=float)
                return
        elif isinstance(x, bool_):
            if handle is None:
                return { "type": "boolean", "values": bool(x) }
            else:
                _stage_scalar_hdf5(handle, x=x, dtype=bool)
                return 
        else:
            raise NotImplementedError("no staging method for NumPy array scalars of " + str(x.dtype))

    elif x == None:
        if handle is None:
            return { "type": "nothing" }
        else:
            handle["uzuki_type"] = "nothing"
            return

    externals.append(x)
    if handle is None:
        return { "type": "other", "index": len(externals) - 1 }
    else:
        handle["uzuki_type"] = "other"
        handle.create_dataset("index", data=len(externals) - 1)
        return

##########################################################################

def _stage_scalar_hdf5(handle, x, dtype, missing_placeholder = None):
    handle.attrs["uzuki_object"] = "vector"

    dtype = None
    if dtype == bool:
        handle.attrs["uzuki_type"] = "boolean"
        dtype = 'u1'
    elif dtype == int:
        handle.attrs["uzuki_type"] = "integer"
        dtype = 'i4'
    elif dtype == str:
        handle.attrs["uzuki_type"] = "string"
    elif dtype == float:
        handle.attrs["uzuki_type"] = "number"
        dtype = 'f8'
    else:
        raise NotImplementedError("no staging method for scalars of " + str(dtype))

    dhandle = handle.create_dataset("data", data=x, dtype=dtype)
    if missing_placeholder:
        dhandle.attrs.create("missing-value-placeholder", data=missing_placeholder, dtype=dtype)


def _stage_vector_hdf5(handle, x, dtype, missing_placeholder = None):
    handle.attrs["uzuki_object"] = "vector"

    dtype = None
    if dtype == bool:
        handle.attrs["uzuki_type"] = "boolean"
        dtype = "u1"
        if missing_placeholder:
            x = _fill_boolean_missing_placeholder(x, missing_placeholder)
    elif dtype == int:
        handle.attrs["uzuki_type"] = "integer"
        dtype = 'i4'
        if missing_placeholder:
            x = _fill_integer_missing_placeholder(x, missing_placeholder)
    elif dtype == float:
        handle.attrs["uzuki_type"] = "number"
        dtype = "f8"
        if missing_placeholder:
            x = _fill_float_missing_placeholder(x, missing_placeholder)
    else:
        raise NotImplementedError("no staging method for vectors of " + str(dtype))

    dhandle = handle.create_dataset("data", data=x, dtype=dtype, compression="gzip", chunks=True)
    if missing_placeholder:
        dhandle.attrs.create("missing-value-placeholder", data=missing_placeholder, dtype=dtype)
