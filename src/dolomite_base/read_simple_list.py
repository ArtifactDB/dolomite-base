from typing import Union
import os

from .alt_read_object import alt_read_object
from . import lib_dolomite_base as lib


def read_simple_list(path: str, metadata: dict, **kwargs) -> Union[dict, list]:
    """Read an R-style list from its on-disk representation in the **uzuki2**
    format.  In general, this function should not be called directly but
    instead via :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: 
            Path to the directory containing the object.

        metadata: 
            Metadata for the object.
    
        kwargs: 
            Further arguments, passed to nested objects.

    Returns:
        A list or dictionary.
    """

    other_dir = os.path.join(path, "other_contents")
    children = []
    if os.path.exists(other_dir):
        files = os.listdir(other_dir)
        collected = []
        for f in files:
            if f.isdigit():
                collected.append(f)
        children = [None] * len(collected)
        for f in collected:
            children[int(f)] = alt_read_object(os.path.join(other_dir, f))

    if metadata["simple_list"]["format"] == "hdf5":
        full_path = os.path.join(path, "list_contents.h5")
        return lib.load_list_hdf5(full_path, "simple_list", children)
    else:
        full_path = os.path.join(path, "list_contents.json.gz")
        return lib.load_list_json(full_path, children)
