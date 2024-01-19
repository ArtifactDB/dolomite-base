from typing import Any, Union, Literal
import numpy as np
import warnings
from functools import singledispatch
from biocutils import Factor, StringList
import os
import json
import gzip
import h5py

from .save_object import save_object, validate_saves
from .alt_save_object import alt_save_object
from . import _utils as ut


@save_object.register
@validate_saves
def save_simple_dict(x: dict, path: str, simple_list_mode: Literal["hdf5", "json"] = "json", **kwargs):
    """Method for saving dictionaries (Python analogues to R-style named lists)
    to the corresponding file representations, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to a directory in which to save the object.

        simple_list_mode: Whether to save in HDF5 or JSON mode.

        kwargs: Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    _save_simple_list_internal(x, path, simple_list_mode, **kwargs)
    return


@save_object.register
@validate_saves
def save_simple_list(x: list, path: str, simple_list_mode: Literal["hdf5", "json"] = "json", **kwargs):
    """Method for saving lists (Python analogues to R-style unnamed lists) to
    the corresponding file representations, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to a directory in which to save the object.

        simple_list_mode: Whether to save in HDF5 or JSON mode.

        kwargs: Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    _save_simple_list_internal(x, path, simple_list_mode, **kwargs)
    return


##########################################################################


def _save_simple_list_internal(x: Union[dict, list], path: str, simple_list_mode: Literal["hdf5", "json"] = None, **kwargs):
    os.mkdir(path)
    with open(os.path.join(path, "OBJECT"), 'w', encoding="utf-8") as handle:
        format2 = simple_list_mode 
        if format2 == "json":
            format2 = "json.gz"
        handle.write('{ "type": "simple_list", "simple_list": { "version": "1.0", "format": "' + format2 + '" } }')

    externals = []
    components = {}

    if simple_list_mode == "json":
        transformed = _save_simple_list_recursive(x, externals, None)
        transformed["version"] = "1.2"
        opath = os.path.join(path, "list_contents.json.gz")
        with gzip.open(opath, "wt") as handle:
            json.dump(transformed, handle)

    else:
        opath = os.path.join(path, "list_contents.h5")
        with h5py.File(opath, "w") as handle:
            ghandle = handle.create_group("simple_list")
            ghandle.attrs["uzuki_version"] = "1.3"
            _save_simple_list_recursive(x, externals, ghandle)

    if len(externals):
        exdir = os.path.join(path, "other_contents")
        os.mkdir(exdir)
        for i, ex in enumerate(externals):
            alt_save_object(ex, os.path.join(exdir, str(i)), **kwargs)
    return


@singledispatch
def _save_simple_list_recursive(x: Any, externals: list, handle):
    externals.append(x)
    if handle is None:
        return { "type": "external", "index": len(externals) - 1 }
    else:
        handle.attrs["uzuki_object"] = "external"
        handle.create_dataset("index", data=len(externals) - 1, dtype='i4')
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_stringlist(x: StringList, externals: list, handle):
    nms = x.get_names()

    if handle is None:
        output = { "type": "string", "values": x.as_list() }
        if nms is not None:
            output["names"] = nms.as_list()
        return output

    has_none = any(y is None for y in x)
    if has_none:
        x, placeholder = ut._choose_missing_string_placeholder(x)

    handle.attrs["uzuki_object"] = "vector"
    handle.attrs["uzuki_type"] = "string"
    dset = ut._save_fixed_length_strings(handle, "data", x)

    if has_none:
       dset.attrs["missing-value-placeholder"] = placeholder
    if nms is not None:
        ut._save_fixed_length_strings(handle, "names", nms.as_list())

    return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_list(x: list, externals: list, handle):
    if handle is None:
        vals = []
        collected = { "type": "list", "values": vals }
        for i, y in enumerate(x):
            vals.append(_save_simple_list_recursive(y, externals, None))
        return collected
    else:
        handle.attrs["uzuki_object"] = "list"
        dhandle = handle.create_group("data")
        for i, y in enumerate(x):
            ghandle = dhandle.create_group(str(i))
            _save_simple_list_recursive(y, externals, ghandle)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_dict(x: dict, externals: list, handle):
    if handle is None:
        vals = []
        names = []
        collected = { "type": "list", "values": vals, "names": names }
        for k, v in x.items():
            if not isinstance(k, str):
                warnings.warn("converting non-string key with value " + str(k) + " to a string", UserWarning)
            names.append(str(k))
            vals.append(_save_simple_list_recursive(v, externals, None))
        return collected
    else:
        handle.attrs["uzuki_object"] = "list"
        dhandle = handle.create_group("data")
        names = []
        for k, v in x.items():
            ghandle = dhandle.create_group(str(len(names)))
            _save_simple_list_recursive(v, externals, ghandle)
            if not isinstance(k, str):
                warn("converting non-string key with value " + str(k) + " to a string", UserWarning)
            names.append(str(k))
        handle.create_dataset("names", data=names, compression="gzip", chunks=True)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_bool(x: bool, externals: list, handle):
    if handle is None:
        return { "type": "boolean", "values": bool(x) }
    else:
        _save_scalar_hdf5(handle, x=x, dtype=bool)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_int(x: int, externals: list, handle):
    if ut._is_integer_scalar_within_limit(x):
        if handle is None:
            return { "type": "integer", "values": int(x) }
        else:
            _save_scalar_hdf5(handle, x=x, dtype=int)
            return
    else:
        if handle is None:
            return { "type": "number", "values": x }
        else:
            _save_scalar_hdf5(handle, x=x, dtype=float)
            return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_str(x: str, externals: list, handle):
    if handle is None:
        return { "type": "string", "values": str(x) }
    else:
        _save_scalar_hdf5(handle, x=x, dtype=str)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_float(x: float, externals: list, handle):
    if handle is None:
        return { "type": "number", "values": _sanitize_float_json(x) }
    else:
        _save_scalar_hdf5(handle, x=x, dtype=float)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_ndarray(x: np.ndarray, externals: list, handle):
    ndims = len(x.shape)
    if ndims == 0:
        x_scalar = x[()]
        if not ut._is_actually_masked(x):
            return _save_simple_list_recursive(x_scalar, externals, handle)
        else:
            final_type = ut._determine_save_type(x_scalar)
            if final_type == int:
                if handle is None:
                    return { "type": "integer", "values": None }
                else:
                    _save_scalar_hdf5(handle, x=-ut.LIMIT32, dtype=int, missing_placeholder=-ut.LIMIT32)
                    return
            elif final_type == float:
                if handle is None:
                    return { "type": "number", "values": None }
                else:
                    _save_scalar_hdf5(handle, x=np.NaN, dtype=float, missing_placeholder=np.NaN)
                    return
            elif final_type == bool:
                if handle is None:
                    return { "type": "boolean", "values": None }
                else:
                    _save_scalar_hdf5(handle, x=-1, dtype=bool, missing_placeholder=-1)
                    return
            else:
                raise NotImplementedError("no staging method for NumPy masked scalars of " + str(x.dtype))

    elif ndims == 1:
        final_type = ut._determine_save_type(x)
        if ut._is_actually_masked(x):
            if final_type == int:
                if handle is None:
                    return { "type": "integer", "values": [None if np.ma.is_masked(y) else int(y) for y in x] }
                else:
                    x, placeholder, final_type = ut._choose_missing_integer_placeholder(x)
                    _save_vector_hdf5(handle, x=x, dtype=final_type, missing_placeholder=placeholder)
                    return
            elif final_type == float:
                if handle is None:
                    return { "type": "number", "values": [_sanitize_masked_float_json(y) for y in x] }
                else:
                    x, placeholder = ut._choose_missing_float_placeholder(x)
                    _save_vector_hdf5(handle, x=x, dtype=float, missing_placeholder=placeholder)
                    return
            elif final_type == bool:
                if handle is None:
                    return { "type": "boolean", "values": [None if np.ma.is_masked(y) else bool(y) for y in x] }
                else:
                    x, placeholder = ut._choose_missing_boolean_placeholder(x)
                    _save_vector_hdf5(handle, x=x, dtype=bool, missing_placeholder=placeholder)
                    return
            else:
                raise NotImplementedError("no staging method for 1D NumPy masked arrays of " + str(x.dtype))
        else:
            if final_type == int:
                if handle is None:
                    return { "type": "integer", "values": [int(y) for y in x] }
                else:
                    _save_vector_hdf5(handle, x=x, dtype=int)
                    return
            elif final_type == float:
                if handle is None:
                    return { "type": "number", "values": [_sanitize_float_json(y) for y in x] }
                else:
                    _save_vector_hdf5(handle, x=x, dtype=float)
                    return
            elif final_type == bool:
                if handle is None:
                    return { "type": "boolean", "values": [bool(y) for y in x] }
                else:
                    _save_vector_hdf5(handle, x=x, dtype=bool)
                    return
            else:
                raise NotImplementedError("no staging method for 1D NumPy arrays of " + str(x.dtype))

    else:
        return _save_simple_list_recursive.registry[Any](x, externals, handle)


@_save_simple_list_recursive.register
def _save_simple_list_recursive_MaskedConstant(x: np.ma.core.MaskedConstant, externals: list, handle):
    if handle is None:
        return { "type": "number", "values": [None]}
    else:
        _save_scalar_hdf5(handle, x=np.NaN, dtype=float, missing_placeholder=np.NaN)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_numpy_generic(x: np.generic, externals: list, handle):
    final_type = ut._determine_save_type(x)

    if final_type == int:
        if handle is None:
            return { "type": "integer", "values": int(x) }
        else:
            _save_scalar_hdf5(handle, x=x, dtype=int)
            return
    elif final_type == float:
        if handle is None:
            return { "type": "number", "values": _sanitize_float_json(x) }
        else:
            _save_scalar_hdf5(handle, x=x, dtype=float)
            return
    elif final_type == bool:
        if handle is None:
            return { "type": "boolean", "values": bool(x) }
        else:
            _save_scalar_hdf5(handle, x=x, dtype=bool)
            return 
    else:
        raise NotImplementedError("no staging method for NumPy array scalars of " + str(x.dtype))


@_save_simple_list_recursive.register
def _save_simple_list_recursive_factor(x: Factor, externals: list, handle):
    nms = x.get_names()

    if handle is None:
        output = { 
            "type": "factor",
            "values": [(None if y == -1 else int(y)) for y in x.get_codes()],
            "levels": x.get_levels().as_list(),
            "ordered": x.get_ordered(),
        }
        if not nms is None:
            output["names"] = nms.as_list()
        return output

    else:
        handle.attrs["uzuki_object"] = "vector"
        handle.attrs["uzuki_type"] = "factor"

        dhandle = handle.create_dataset("data", data=x.get_codes(), dtype="i4", compression="gzip", chunks=True)
        if (x.get_codes() == -1).any():
            dhandle.attrs.create("missing-value-placeholder", data=-1, dtype="i4")

        ut._save_fixed_length_strings(handle, "levels", x.get_levels().as_list())
        if x.get_ordered():
            handle.create_dataset("ordered", data=x.get_ordered(), dtype="i1")

        if not nms is None:
            ut._save_fixed_length_strings(handle, "names", nms.as_list())
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_none(x: None, externals: list, handle):
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


def _save_scalar_hdf5(handle, x, dtype, missing_placeholder = None):
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


def _save_vector_hdf5(handle, x, dtype, missing_placeholder = None):
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
