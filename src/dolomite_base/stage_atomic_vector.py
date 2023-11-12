from typing import Any
from biocutils import StringList
from biocframe import BiocFrame
import os

from .stage_object import stage_object
from .write_csv import write_csv
from ._utils import _determine_numpy_type


def _stage_atomic_vector(x, dir: str, path: str, is_child: bool = False):
    os.mkdir(os.path.join(dir, path))
    ofpath = path + "/simple.csv.gz"
    write_csv(BiocFrame({ "values": x }), os.path.join(dir, ofpath), compression="gzip")

    if isinstance(x, StringList):
        vectype = "string"
    else:
        ntype = _determine_numpy_type(x)
        if ntype == float:
            vectype = "number"
        elif ntype == int:
            vectype = "integer"
        elif ntype == bool:
            vectype = "boolean"

    return {
        "path": ofpath,
        "is_child": is_child,
        "$schema": "atomic_vector/v1.json",
        "atomic_vector": {
            "length": len(x),
            "type": vectype,
            "compression": "gzip"
        }
    }


@stage_object.register
def stage_string_list(
    x: StringList, 
    dir: str, 
    path: str, 
    is_child: bool = False, 
    **kwargs
) -> dict[str, Any]:
    """Method for saving :py:class:`~biocutils.StringList.StringList` objects
    to their corresponding file representation, see
    :py:meth:`~dolomite_base.stage_object.stage_object` for details.

    Args:
        x: Object to be staged.

        dir: Staging directory.

        path: Relative path inside ``dir`` to save the object.

        is_child: Is ``x`` a child of another object?

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    return _stage_atomic_vector(x, dir=dir, path=path, is_child=is_child, **kwargs)
