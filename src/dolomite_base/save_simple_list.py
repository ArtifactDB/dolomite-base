from typing import Any, Union, Literal
import numpy as np
from warnings import warn
from functools import singledispatch
from biocutils import Factor, StringList, NamedList, IntegerList, BooleanList, FloatList
import os
import json
import gzip
import h5py

from .save_object import save_object, validate_saves
from .save_object_file import save_object_file
from .alt_save_object import alt_save_object
from . import _utils_misc as misc
from . import _utils_string as strings
from . import write_vector_to_hdf5 as write


@save_object.register
@validate_saves
def save_simple_list_from_dict(x: dict, path: str, simple_list_mode: Literal["hdf5", "json"] = "json", **kwargs):
    """Method for saving dictionaries (Python analogues to R-style named lists)
    to the corresponding file representations, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to a directory in which to save the object.

        simple_list_mode: 
            Whether to save in HDF5 or JSON mode.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    _save_simple_list_internal(x, path, simple_list_mode, **kwargs)
    return


@save_object.register
@validate_saves
def save_simple_list_from_list(x: list, path: str, simple_list_mode: Literal["hdf5", "json"] = "json", **kwargs):
    """Method for saving lists (Python analogues to R-style unnamed lists) to
    the corresponding file representations, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to a directory in which to save the object.

        simple_list_mode: 
            Whether to save in HDF5 or JSON mode.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    _save_simple_list_internal(x, path, simple_list_mode, **kwargs)
    return


@save_object.register
@validate_saves
def save_simple_list_from_NamedList(x: NamedList, path: str, simple_list_mode: Literal["hdf5", "json"] = "json", **kwargs):
    """Method for saving a NamedList to its corresponding file representation,
    see :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to a directory in which to save the object.

        simple_list_mode: 
            Whether to save in HDF5 or JSON mode.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    _save_simple_list_internal(x, path, simple_list_mode, **kwargs)
    return


##########################################################################


def _save_simple_list_internal(x: Union[dict, list, NamedList], path: str, simple_list_mode: Literal["hdf5", "json"] = None, **kwargs):
    os.mkdir(path)

    format2 = simple_list_mode 
    if format2 == "json":
        format2 = "json.gz"
    save_object_file(path, "simple_list", { "simple_list": { "version": "1.0", "format": format2 } })

    externals = []

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
    return _save_simple_list_recursive_Any(x, externals, handle)


def _save_simple_list_recursive_Any(x: Any, externals: list, handle):
    externals.append(x)
    if handle is None:
        return { "type": "external", "index": len(externals) - 1 }
    else:
        handle.attrs["uzuki_object"] = "external"
        handle.create_dataset("index", data=len(externals) - 1, dtype='i4')
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_StringList(x: StringList, externals: list, handle):
    nms = x.get_names()

    if handle is None:
        output = { "type": "string", "values": x.as_list() }
        if nms is not None:
            output["names"] = nms.as_list()
        return output

    handle.attrs["uzuki_object"] = "vector"
    handle.attrs["uzuki_type"] = "string"
    write.write_string_vector_to_hdf5(handle, "data", x.as_list())
    if nms is not None:
        strings.save_fixed_length_strings(handle, "names", nms.as_list())
    return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_IntegerList(x: IntegerList, externals: list, handle):
    nms = x.get_names()

    if handle is None:
        final_type = "integer"
        if misc.sequence_exceeds_int32(x):
            final_type = "number"
        output = { "type": final_type, "values": x.as_list() }
        if nms is not None:
            output["names"] = nms.as_list()
        return output

    handle.attrs["uzuki_object"] = "vector"
    dset = write.write_integer_vector_to_hdf5(handle, "data", x.as_list(), allow_float_promotion=True)
    if np.issubdtype(dset, np.floating):
        handle.attrs["uzuki_type"] = "number"
    else:
        handle.attrs["uzuki_type"] = "integer"
    if nms is not None:
        strings.save_fixed_length_strings(handle, "names", nms.as_list())
    return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_FloatList(x: FloatList, externals: list, handle):
    nms = x.get_names()

    if handle is None:
        xcopy = [ _sanitize_masked_float_json(y) for y in x.as_list() ]
        output = { "type": "number", "values": xcopy }
        if nms is not None:
            output["names"] = nms.as_list()
        return output

    handle.attrs["uzuki_object"] = "vector"
    handle.attrs["uzuki_type"] = "number"
    write.write_float_vector_to_hdf5(handle, "data", x.as_list())
    if nms is not None:
        strings.save_fixed_length_strings(handle, "names", nms.as_list())
    return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_BooleanList(x: BooleanList, externals: list, handle):
    nms = x.get_names()

    if handle is None:
        output = { "type": "boolean", "values": x.as_list() }
        if nms is not None:
            output["names"] = nms.as_list()
        return output

    handle.attrs["uzuki_object"] = "vector"
    handle.attrs["uzuki_type"] = "boolean"
    write.write_boolean_vector_to_hdf5(handle, "data", x.as_list())
    if nms is not None:
        strings.save_fixed_length_strings(handle, "names", nms.as_list())
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
                warn("converting non-string key with value " + str(k) + " to a string", UserWarning)
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
        strings.save_fixed_length_strings(handle, "names", names)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_NamedList(x: NamedList, externals: list, handle):
    if x.get_names() is None:
        return _save_simple_list_recursive_list(x.as_list(), externals, handle)

    if handle is None:
        vals = []
        collected = { "type": "list", "values": vals, "names": x.get_names().as_list() }
        for v in x.as_list():
            vals.append(_save_simple_list_recursive(v, externals, None))
        return collected
    else:
        handle.attrs["uzuki_object"] = "list"
        dhandle = handle.create_group("data")
        for i, v in enumerate(x.as_list()):
            ghandle = dhandle.create_group(str(i))
            _save_simple_list_recursive(v, externals, ghandle)
        strings.save_fixed_length_strings(handle, "names", x.get_names().as_list())
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
    if not misc.scalar_exceeds_int32(x):
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
def _save_simple_list_recursive_MaskedConstant(x: np.ma.core.MaskedConstant, externals: list, handle):
    if handle is None:
        return { "type": "number", "values": None}
    else:
        _save_scalar_hdf5(handle, x=np.nan, dtype=float, missing_placeholder=np.nan)
        return


@_save_simple_list_recursive.register
def _save_simple_list_recursive_MaskedConstant(x: np.ndarray, externals: list, handle):
    if len(x.shape) == 0:
        return _save_simple_list_recursive(x[()], externals, handle)
    else:
        return _save_simple_list_recursive_Any(x, externals, handle)


@_save_simple_list_recursive.register
def _save_simple_list_recursive_numpy_generic(x: np.generic, externals: list, handle):
    final_type = None
    if np.issubdtype(x.dtype, np.integer):
        if not misc.scalar_exceeds_int32(x):
            final_type = int
        else:
            final_type = float
    elif np.issubdtype(x.dtype, np.floating):
        final_type = float
    elif x.dtype == np.bool_:
        final_type = bool

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
        if nms is not None:
            output["names"] = nms.as_list()
        return output

    else:
        handle.attrs["uzuki_object"] = "vector"
        handle.attrs["uzuki_type"] = "factor"

        dhandle = handle.create_dataset("data", data=x.get_codes(), dtype="i4", compression="gzip", chunks=True)
        if (x.get_codes() == -1).any():
            dhandle.attrs.create("missing-value-placeholder", data=-1, dtype="i4")

        strings.save_fixed_length_strings(handle, "levels", x.get_levels().as_list())
        if x.get_ordered():
            handle.create_dataset("ordered", data=x.get_ordered(), dtype="i1")

        if nms is not None:
            strings.save_fixed_length_strings(handle, "names", nms.as_list())
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
    elif x == np.inf:
        return "Inf"
    elif x == -np.inf:
        return "-Inf"
    return float(x)


def _sanitize_masked_float_json(x):
    if x is None:
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
