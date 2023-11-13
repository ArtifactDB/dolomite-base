from typing import Any, Union, Optional, Literal
import numpy as np
import warnings
from numpy import ndarray, issubdtype, integer, floating, bool_
from functools import singledispatch
from biocutils import Factor, StringList
import os
import json
import gzip
import h5py

from .stage_object import stage_object
from .alt_stage_object import alt_stage_object
from .write_metadata import write_metadata
from . import lib_dolomite_base as lib
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
        child_meta = alt_stage_object(ex, dir, path + "/" + str(i))
        children.append({ "resource": write_metadata(child_meta, dir) })
    components["simple_list"] = { "children": children }
    components["is_child"] = is_child
    return components


@singledispatch
def _stage_simple_list_recursive(x: Any, externals: list, handle):
    externals.append(x)
    if handle is None:
        return { "type": "external", "index": len(externals) - 1 }
    else:
        handle.attrs["uzuki_object"] = "external"
        handle.create_dataset("index", data=len(externals) - 1, dtype='i4')
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_stringlist(x: StringList, externals: list, handle):
    if handle is None:
        return { "type": "string", "values": x }
    else:
        has_none = any(y is None for y in x)
        if has_none:
            x, placeholder = ut._choose_missing_string_placeholder(x)

        handle.attrs["uzuki_object"] = "vector"
        handle.attrs["uzuki_type"] = "string"
        dset = ut._save_fixed_length_strings(handle, "data", x)

        if has_none:
           dset.attrs["missing-value-placeholder"] = placeholder
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_list(x: list, externals: list, handle):
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


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_dict(x: dict, externals: list, handle):
    if handle is None:
        vals = []
        names = []
        collected = { "type": "list", "values": vals, "names": names }
        for k, v in x.items():
            if not isinstance(k, str):
                warnings.warn("converting non-string key with value " + str(k) + " to a string", UserWarning)
            names.append(str(k))
            vals.append(_stage_simple_list_recursive(v, externals, None))
        return collected
    else:
        handle.attrs["uzuki_object"] = "list"
        dhandle = handle.create_group("data")
        names = []
        for k, v in x.items():
            ghandle = dhandle.create_group(str(len(names)))
            _stage_simple_list_recursive(v, externals, ghandle)
            if not isinstance(k, str):
                warn("converting non-string key with value " + str(k) + " to a string", UserWarning)
            names.append(str(k))
        handle.create_dataset("names", data=names, compression="gzip", chunks=True)
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_bool(x: bool, externals: list, handle):
    if handle is None:
        return { "type": "boolean", "values": bool(x) }
    else:
        _stage_scalar_hdf5(handle, x=x, dtype=bool)
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_int(x: int, externals: list, handle):
    if ut._is_integer_scalar_within_limit(x):
        if handle is None:
            return { "type": "integer", "values": int(x) }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=int)
            return
    else:
        if handle is None:
            return { "type": "number", "values": x }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=float)
            return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_str(x: str, externals: list, handle):
    if handle is None:
        return { "type": "string", "values": str(x) }
    else:
        _stage_scalar_hdf5(handle, x=x, dtype=str)
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_float(x: float, externals: list, handle):
    if handle is None:
        return { "type": "number", "values": _sanitize_float_json(x) }
    else:
        _stage_scalar_hdf5(handle, x=x, dtype=float)
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_ndarray(x: np.ndarray, externals: list, handle):
    ndims = len(x.shape)
    if ndims == 0:
        if not np.ma.is_masked(x) or not bool(x.mask):
            return _stage_simple_list_recursive(x.dtype.type(x), externals, handle)
        else:
            final_type = ut._determine_numpy_type(x.dtype.type(x))
            if final_type == int:
                if handle is None:
                    return { "type": "integer", "values": None }
                else:
                    _stage_scalar_hdf5(handle, x=-ut.LIMIT32, dtype=int, missing_placeholder=-ut.LIMIT32)
                    return
            elif final_type == float:
                if handle is None:
                    return { "type": "number", "values": None }
                else:
                    _stage_scalar_hdf5(handle, x=np.NaN, dtype=float, missing_placeholder=np.NaN)
                    return
            elif x.dtype == bool_:
                if handle is None:
                    return { "type": "boolean", "values": None }
                else:
                    _stage_scalar_hdf5(handle, x=-1, dtype=bool, missing_placeholder=-1)
                    return
            else:
                raise NotImplementedError("no staging method for 1D NumPy masked arrays of " + str(x.dtype))

    elif ndims != 1:
        return _stage_simple_list_recursive.registry[Any](x, externals, handle)
    else:
        final_type = ut._determine_numpy_type(x)
        if np.ma.is_masked(x):
            if final_type == int:
                if handle is None:
                    return { "type": "integer", "values": [None if np.ma.is_masked(y) else int(y) for y in x] }
                else:
                    # If there's no valid missing placeholder, we just save it as floating-point.
                    x, placeholder = ut._choose_missing_integer_placeholder(x)
                    if placeholder is not None:
                        _stage_vector_hdf5(handle, x=x, dtype=int, missing_placeholder=placeholder)
                    else:
                        x, placeholder = ut._choose_missing_float_placeholder(x)
                        _stage_vector_hdf5(handle, x=x, dtype=float, missing_placeholder=placeholder)
                    return
            elif final_type == float:
                if handle is None:
                    return { "type": "number", "values": [_sanitize_masked_float_json(y) for y in x] }
                else:
                    x, placeholder = ut._choose_missing_float_placeholder(x)
                    _stage_vector_hdf5(handle, x=x, dtype=float, missing_placeholder=placeholder)
                    return
            elif x.dtype == bool_:
                if handle is None:
                    return { "type": "boolean", "values": [None if np.ma.is_masked(y) else bool(y) for y in x] }
                else:
                    x, placeholder = ut._choose_missing_boolean_placeholder(x)
                    _stage_vector_hdf5(handle, x=x, dtype=bool, missing_placeholder=placeholder)
                    return
            else:
                raise NotImplementedError("no staging method for 1D NumPy masked arrays of " + str(x.dtype))
        else:
            if final_type == int:
                if handle is None:
                    return { "type": "integer", "values": [int(y) for y in x] }
                else:
                    _stage_vector_hdf5(handle, x=x, dtype=int)
                    return
            elif final_type == float:
                if handle is None:
                    return { "type": "number", "values": [_sanitize_float_json(y) for y in x] }
                else:
                    _stage_vector_hdf5(handle, x=x, dtype=float)
                    return
            elif final_type == bool:
                if handle is None:
                    return { "type": "boolean", "values": [bool(y) for y in x] }
                else:
                    _stage_vector_hdf5(handle, x=x, dtype=bool)
                    return
            else:
                raise NotImplementedError("no staging method for 1D NumPy arrays of " + str(x.dtype))


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_MaskedConstant(x: np.ma.core.MaskedConstant, externals: list, handle):
    if handle is None:
        return { "type": "number", "values": [None]}
    else:
        _stage_scalar_hdf5(handle, x=np.NaN, dtype=float, missing_placeholder=np.NaN)
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_numpy_generic(x: np.generic, externals: list, handle):
    final_type = ut._determine_numpy_type(x)

    if final_type == int:
        if handle is None:
            return { "type": "integer", "values": int(x) }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=int)
            return
    elif final_type == float:
        if handle is None:
            return { "type": "number", "values": _sanitize_float_json(x) }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=float)
            return
    elif final_type == bool:
        if handle is None:
            return { "type": "boolean", "values": bool(x) }
        else:
            _stage_scalar_hdf5(handle, x=x, dtype=bool)
            return 
    else:
        raise NotImplementedError("no staging method for NumPy array scalars of " + str(x.dtype))


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_factor(x: Factor, externals: list, handle):
    if handle is None:
        return { 
            "type": "factor",
            "values": [(None if y == -1 else int(y)) for y in x.get_codes()],
            "levels": x.get_levels(),
            "ordered": x.get_ordered(),
        }

    else:
        handle.attrs["uzuki_object"] = "vector"
        handle.attrs["uzuki_type"] = "factor"

        dhandle = handle.create_dataset("data", data=x.get_codes(), dtype="i4", compression="gzip", chunks=True)
        if (x.get_codes() == -1).any():
            dhandle.attrs["missing-value-placeholder"] = -1

        ut._save_fixed_length_strings(handle, "levels", x.get_levels())
        if x.get_ordered():
            handle.create_dataset("ordered", data=x.get_ordered(), dtype="i1")
        return


@_stage_simple_list_recursive.register
def _stage_simple_list_recursive_none(x: None, externals: list, handle):
    if handle is None:
        return { "type": "nothing" }
    else:
        handle.attrs["uzuki_object"] = "nothing"
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
        savetype = 'i1'
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
        savetype = "i1"
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
