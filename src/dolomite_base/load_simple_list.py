from typing import Any, Union
import numpy as np
import ctypes as ct

from .acquire_file import acquire_file
from .acquire_metadata import acquire_metadata
from .load_object import load_object
from . import _cpphelpers as lib
from ._utils import _fragment_string_contents, _mask_strings


class _ParsedList:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        lib.uzuki2_free_list(self.ptr)

    @property
    def entrypoint(self):
        return lib.uzuki2_get_parent_node(self.ptr)


def extract_list_contents(ptr, externals):
    index = lib.uzuki2_get_node_type(ptr)

    if index == 0: # integer.
        n = lib.uzuki2_get_integer_vector_length(ptr)
        actual = max(n, 1)
        output = np.ndarray(actual, dtype=np.int32)
        masked = lib.uzuki2_get_integer_vector_values(ptr, output)

        if masked:
            mask = np.zeros(actual, dtype=np.uint8)
            lib.uzuki2_get_integer_vector_mask(ptr, mask)
            output = np.ma.array(output, mask=mask)
            if n == -1 and np.ma.is_masked(output[0]):
                return output[0]

        if n == -1:
            return int(output[0])
        else:
            return output

    elif index == 1: # number
        n = lib.uzuki2_get_number_vector_length(ptr)
        actual = max(n, 1)
        output = np.ndarray(actual, dtype=np.float64)
        masked = lib.uzuki2_get_number_vector_values(ptr, output)

        if masked:
            mask = np.zeros(actual, dtype=np.uint8)
            lib.uzuki2_get_number_vector_mask(ptr, mask)
            output = np.ma.array(output, mask=mask)
            if n == -1 and np.ma.is_masked(output[0]):
                return output[0]

        if n == -1:
            return float(output[0])
        else:
            return output

    elif index == 2: # boolean
        n = lib.uzuki2_get_boolean_vector_length(ptr)
        actual = max(n, 1)
        output = np.ndarray(actual, dtype=np.uint8)
        masked = lib.uzuki2_get_boolean_vector_values(ptr, output)

        if masked:
            mask = np.zeros(actual, dtype=np.uint8)
            lib.uzuki2_get_boolean_vector_mask(ptr, mask)
            output = np.ma.array(output, mask=mask)
            if n == -1 and np.ma.is_masked(output[0]):
                return output[0]

        if n == -1:
            return bool(output[0])
        else:
            return output.astype(np.bool_)

    elif index == 3: # string
        n = lib.uzuki2_get_string_vector_length(ptr)
        actual = max(n, 1)
        strlengths = np.ndarray(actual, dtype=np.int32)
        total = lib.uzuki2_get_string_vector_lengths(ptr, strlengths)

        concatenated = ct.create_string_buffer(total)
        masked = lib.uzuki2_get_string_vector_contents(ptr, concatenated)
        collected = _fragment_string_contents(strlengths, concatenated.raw)

        if masked:
            mask = np.zeros(actual, dtype=np.uint8)
            lib.uzuki2_get_string_vector_mask(ptr, mask=mask)
            _mask_strings(collected, mask)

        if n == -1:
            return collected[0]
        else:
            return collected

    elif index == 4: # list
        collected = []
        n = lib.uzuki2_get_list_length(ptr)
        for i in range(n):
            collected.append(extract_list_contents(lib.uzuki2_get_list_element(ptr, i), externals))
        if not lib.uzuki2_get_list_named(ptr):
            return collected

        strlengths = np.ndarray(n, dtype=np.int32)
        total = lib.uzuki2_get_list_names_lengths(ptr, strlengths)
        concatenated = ct.create_string_buffer(total)
        lib.uzuki2_get_list_names_contents(ptr, concatenated)
        names = _fragment_string_contents(strlengths, concatenated.raw)

        output = {}
        for i, x in enumerate(collected):
            output[names[i]] = x
        return output

    elif index == 5:
        return None

    elif index == 6:
        return externals[lib.get_external_index(ptr)]

    else:
        raise NotImplementedError("unknown uzuki2 code " + str(index))


def _load_all_children(meta: dict[str, Any], project: Any) -> list:
    collected = []

    if "simple_list" in meta:
        smeta = meta["simple_list"]
        if "children" in smeta:
            for cinfo in smeta["children"]:
                cmeta = acquire_metadata(project, cmeta["resource"]["path"])
                collected.append(load_object(cmeta, project))

    return collected


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
        A data frame.
    """
    full_path = acquire_file(project, meta["path"])
    children = _load_all_children(meta, project)
    handle = _ParsedList(lib.load_list_json(full_path.encode("UTF8"), len(children)))
    return extract_list_contents(handle.entrypoint, children)

