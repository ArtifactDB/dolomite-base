from typing import Any, Union
import numpy as np
import ctypes as ct

from .acquire_file import acquire_file
from .acquire_metadata import acquire_metadata
from .alt_load_object import alt_load_object
from . import lib_dolomite_base as lib
from ._utils import _is_gzip_compressed


def load_json_simple_list(meta: dict[str, Any], project: Any, **kwargs) -> Union[list, dict]:
    """Load an R-style list from a (possibly Gzip-compressed) JSON file in the
    **uzuki2** format. In general, this function should not be called directly
    but instead via :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this JSON list.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A list or dictionary.
    """
    _is_gzip_compressed(meta, "json_simple_list") # check it's not crazy.
    full_path = acquire_file(project, meta["path"])
    children = _load_all_children(meta, project)
    return lib.load_list_json(full_path, children)


def load_hdf5_simple_list(meta: dict[str, Any], project: Any, **kwargs) -> Union[list, dict]:
    """Load an R-style list from a HDF5 file in the **uzuki2** format. In
    general, this function should not be called directly but instead via
    :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this HDF5 list.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A list or dictionary.
    """
    full_path = acquire_file(project, meta["path"])
    children = _load_all_children(meta, project)
    group_name = meta["hdf5_simple_list"]["group"]
    return lib.load_list_hdf5(full_path, group_name, children)


def _load_all_children(meta: dict[str, Any], project: Any) -> list:
    collected = []

    if "simple_list" in meta:
        smeta = meta["simple_list"]
        if "children" in smeta:
            for cinfo in smeta["children"]:
                cmeta = acquire_metadata(project, cinfo["resource"]["path"])
                collected.append(alt_load_object(cmeta, project))

    return collected
