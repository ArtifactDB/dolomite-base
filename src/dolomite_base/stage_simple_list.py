from typing import Any, Union, Optional, Literal
import numpy as np
from numpy import ndarray, issubdtype, integer, floating, bool_
import os
import json
import gzip
import h5py

from .stage_object import stage_object
from .write_metadata import write_metadata
from . import _cpphelpers as lib
from . import _utils as ut


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
            If None, defaults to :py:meth:`~choose_simple_list_format`.

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
            If None, defaults to :py:meth:`~choose_simple_list_format`.

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    return _stage_simple_list_internal(x, dir, path, is_child, mode, **kwargs)


SAVE_LIST_FORMAT = "json"


def choose_simple_list_format(format: Optional[Literal["hdf5", "json"]] = None) -> str:
    """Get or set the format to save a simple list.

    Args:
        format: Format to save a simple list, either in HDF5 or JSON.

    Return:
        If ``format`` is not provided, the current format choice is returned.
        This defaults to `"json"` if no other setting has been provided.

        If ``format`` is provided, it is used to define the format choice,
        and the previous choice is returned.
    """
    global SAVE_LIST_FORMAT
    if format is None:
        return SAVE_LIST_FORMAT
    else:
        old = SAVE_LIST_FORMAT
        SAVE_LIST_FORMAT = format
        return old


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

    if mode == None:
        mode = choose_simple_list_format()

    if mode == "json":
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
        oname = "uzuki2_list"
        with h5py.File(opath, "w") as handle:
            ghandle = handle.create_group(oname)
            ghandle.attrs["uzuki_version"] = "1.1"
            _stage_simple_list_recursive(x, externals, ghandle)

        lib.validate_list_hdf5(opath.encode("UTF8"), oname.encode("UTF8"), len(externals))

        components["$schema"] = "hdf5_simple_list/v1.json"
        components["path"] = newpath
        components["hdf5_simple_list"] = { "group": oname }

    children = []
    for i, ex in enumerate(externals):
        child_meta = stage_object(ex, dir, path + "/" + str(i))
        children.append({ "resource": write_metadata(child_meta, dir) })
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
                        missing_placeholder = ut._choose_string_missing_placeholder(x)
                        new_x = x[:]
                        for i, y in enumerate(new_x):
                            if y is None:
                                new_x[i] = missing_placeholder
                        x = new_x

                    handle.attrs["uzuki_object"] = "vector"
                    handle.attrs["uzuki_type"] = "string"
                    dset = handle.create_dataset("data", data=x, compression="gzip", chunks=True)

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
            return

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
        if ut._is_integer_scalar_within_limit(x):
            if handle is None:
                return { "type": "integer", "values": x }
            else:
                _stage_scalar_hdf5(handle, x=x, dtype=int)
                return
        else:
            if handle is None:
                return { "type": "number", "values": x }
            else:
                _stage_scalar_hdf5(handle, x=x, dtype=float)
                return

    elif isinstance(x, str):
        if handle is None:
            return { "type": "string", "values": x }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=str)
            return

    elif isinstance(x, float):
        if handle is None:
            return { "type": "number", "values": _sanitize_float_json(x) }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=float)
            return

    elif isinstance(x, ndarray):
        if len(x.shape) == 1:
            if np.ma.is_masked(x):
                as_float = False
                if issubdtype(x.dtype, integer):
                    if ut._is_integer_vector_within_limit(x):
                        if handle is None:
                            return { "type": "integer", "values": [None if np.ma.is_masked(y) else int(y) for y in x] }
                        else:
                            # If there's no valid missing placeholder, we just save it as floating-point.
                            missing_placeholder = ut._choose_integer_missing_placeholder(x)
                            if missing_placeholder is not None:
                                x = ut._fill_integer_missing_placeholder(x, missing_placeholder) 
                                _stage_vector_hdf5(handle, x=x, dtype=int, missing_placeholder=missing_placeholder)
                                return
                    as_float = True

                if as_float or issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": [_sanitize_masked_float_json(y) for y in x] }
                    else:
                        missing_placeholder = ut._choose_float_missing_placeholder()
                        x = ut._fill_float_missing_placeholder(x, missing_placeholder)
                        _stage_vector_hdf5(handle, x=x, dtype=float, missing_placeholder=missing_placeholder)
                        return
                elif x.dtype == bool_:
                    if handle is None:
                        return { "type": "boolean", "values": [None if np.ma.is_masked(y) else bool(y) for y in x] }
                    else:
                        missing_placeholder = ut._choose_boolean_missing_placeholder()
                        x = ut._fill_boolean_missing_placeholder(x, missing_placeholder)
                        _stage_vector_hdf5(handle, x=x, dtype=bool, missing_placeholder=missing_placeholder)
                        return
                else:
                    raise NotImplementedError("no staging method for 1D NumPy masked arrays of " + str(x.dtype))

            else:
                as_float = False
                if issubdtype(x.dtype, integer):
                    if ut._is_integer_vector_within_limit(x):
                        if handle is None:
                            return { "type": "integer", "values": [int(y) for y in x] }
                        else:
                            _stage_vector_hdf5(handle, x=x, dtype=int)
                            return
                    as_float = True

                if as_float or issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": [_sanitize_float_json(y) for y in x] }
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
                        return { "type": "integer", "values": None }
                    else:
                        missing_placeholder = ut._choose_integer_missing_placeholder([])
                        _stage_scalar_hdf5(handle, x=missing_placeholder, dtype=int, missing_placeholder=missing_placeholder)
                        return
                elif issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": None }
                    else:
                        missing_placeholder = ut._choose_float_missing_placeholder()
                        _stage_scalar_hdf5(handle, x=missing_placeholder, dtype=float, missing_placeholder=missing_placeholder)
                        return
                elif x.dtype == bool_:
                    if handle is None:
                        return { "type": "boolean", "values": None }
                    else:
                        missing_placeholder = ut._choose_boolean_missing_placeholder()
                        _stage_scalar_hdf5(handle, x=missing_placeholder, dtype=bool, missing_placeholder=missing_placeholder)
                        return
                else:
                    raise NotImplementedError("no staging method for 0-d NumPy arrays of " + str(x.dtype))

            else:
                as_float = False
                if issubdtype(x.dtype, integer):
                    y = int(x)
                    if ut._is_integer_scalar_within_limit(y):
                        if handle is None:
                            return { "type": "integer", "values": y } 
                        else:
                            _stage_scalar_hdf5(handle, x=y, dtype=int)
                            return
                    as_float = True

                if as_float or issubdtype(x.dtype, floating):
                    if handle is None:
                        return { "type": "number", "values": _sanitize_float_json(x) }
                    else:
                        _stage_scalar_hdf5(handle, x=float(x), dtype=float)
                        return
                elif x.dtype == bool_:
                    y = bool(x)
                    if handle is None:
                        return { "type": "boolean", "values": y }
                    else:
                        _stage_scalar_hdf5(handle, x=y, dtype=bool)
                        return
                else:
                    raise NotImplementedError("no staging method for 0-d NumPy arrays of " + str(x.dtype))

    elif isinstance(x, np.generic):
        as_float = False
        if isinstance(x, integer):
            y = int(x)
            if ut._is_integer_scalar_within_limit(y):
                if handle is None:
                    return { "type": "integer", "values": y }
                else:
                    _stage_scalar_hdf5(handle, x=y, dtype=int)
                    return
            as_float = True

        if as_float or isinstance(x, floating):
            if handle is None:
                return { "type": "number", "values": _sanitize_float_json(x) }
            else:
                _stage_scalar_hdf5(handle, x=float(x), dtype=float)
                return
        elif isinstance(x, bool_):
            y = bool(x)
            if handle is None:
                return { "type": "boolean", "values": y }
            else:
                _stage_scalar_hdf5(handle, x=y, dtype=bool)
                return 
        else:
            raise NotImplementedError("no staging method for NumPy array scalars of " + str(x.dtype))

    elif x == None:
        if handle is None:
            return { "type": "nothing" }
        else:
            handle.attrs["uzuki_object"] = "nothing"
            return

    externals.append(x)
    if handle is None:
        return { "type": "external", "index": len(externals) - 1 }
    else:
        handle.attrs["uzuki_object"] = "external"
        handle.create_dataset("index", data=len(externals) - 1, dtype='i4')
        return


##########################################################################


def _sanitize_float_json(x):
    if np.isnan(x):
        return "NaN"
    elif x == np.Inf:
        return "Inf"
    elif x == -np.Inf:
        return "-Inf"
    return float(x)


def _sanitize_masked_float_json(x):
    if np.ma.is_masked(x):
        return None
    return _sanitize_float_json(x)


##########################################################################


def _stage_scalar_hdf5(handle, x, dtype, missing_placeholder = None):
    handle.attrs["uzuki_object"] = "vector"

    if dtype == bool:
        handle.attrs["uzuki_type"] = "boolean"
        savetype = 'u1'
    elif dtype == int:
        handle.attrs["uzuki_type"] = "integer"
        savetype = 'i4'
    elif dtype == str:
        handle.attrs["uzuki_type"] = "string"
        savetype = None
    elif dtype == float:
        handle.attrs["uzuki_type"] = "number"
        savetype = 'f8'
    else:
        raise NotImplementedError("no staging method for scalars of " + str(dtype))

    dhandle = handle.create_dataset("data", data=x, dtype=savetype)
    if missing_placeholder:
        dhandle.attrs.create("missing-value-placeholder", data=missing_placeholder, dtype=savetype)


def _stage_vector_hdf5(handle, x, dtype, missing_placeholder = None):
    handle.attrs["uzuki_object"] = "vector"

    if dtype == bool:
        handle.attrs["uzuki_type"] = "boolean"
        savetype = "u1"
    elif dtype == int:
        handle.attrs["uzuki_type"] = "integer"
        savetype = 'i4'
    elif dtype == float:
        handle.attrs["uzuki_type"] = "number"
        savetype = "f8"
    else:
        raise NotImplementedError("no staging method for vectors of " + str(dtype))

    dhandle = handle.create_dataset("data", data=x, dtype=savetype, compression="gzip", chunks=True)
    if missing_placeholder:
        dhandle.attrs.create("missing-value-placeholder", data=missing_placeholder, dtype=savetype)
