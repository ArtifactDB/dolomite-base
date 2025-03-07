from typing import Dict
import biocframe
import os

from .read_object_file import read_object_file


def list_objects(dir: str, include_children: bool = False) -> biocframe.BiocFrame:
    """List all objects in a directory, along with their types.

    Args:
        dir:
            Path to a directory in which one or more objects were saved, typically via :py:func:`~dolomite_base.save_object.save_object`.

        include_children:
            Whether to include child objects (i.e., objects that are components of other objects) in the listing.

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row corresponds to an object in ``dir``.
        It contains the following columns:

        - ``path``, the relative path to the object's subdirectory inside ``dir``.
        - ``type``, the type of the object.
        - ``child``, whether the object is a child of another object.

        If ``include_children=False``, the listing will only contain non-child objects. 
    """
    return biocframe.BiocFrame(_traverse_directory_listing(dir, ".", include_children=include_children))


def _traverse_directory_listing(root: str, dir: str, already_child: bool = False, include_children: bool = False) -> Dict:
    full = os.path.join(root, dir)
    is_obj = os.path.exists(os.path.join(full, "OBJECT"))

    paths = []
    types = []
    childs = []
    if is_obj:
        paths.append(dir)
        types.append(read_object_file(full)["type"])
        childs.append(already_child)

    if include_children or not is_obj:
        for k in next(os.walk(full))[1]:
            if dir != ".":
                subdir = os.path.join(dir, k)
            else:
                subdir = k
            sub = _traverse_directory_listing(root, subdir, already_child=(already_child or is_obj), include_children=include_children)
            paths += sub["path"]
            types += sub["type"]
            childs += sub["child"]

    return { "path": paths, "type": types, "child": childs }
