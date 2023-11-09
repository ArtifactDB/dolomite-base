from typing import Any
import os
from biocutils import Factor
from biocframe import BiocFrame
import numpy

from .stage_object import stage_object
from .alt_stage_object import alt_stage_object
from .write_metadata import write_metadata
from .write_csv import write_csv


@stage_object.register
def stage_string_factor(x: Factor, dir: str, path: str, is_child: bool = False, **kwargs) -> dict[str, Any]:
    """Method for saving :py:class:`~biocutils.Factor.Factor` objects to their
    corresponding file representation, see
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
    os.mkdir(os.path.join(dir, path))
    ofpath = path + "/codes.csv.gz"

    codes = x.get_codes()
    missing = codes == -1
    if missing.any():
        codes = numpy.ma.array(codes, mask=missing)
    write_csv(BiocFrame({ "values": codes }), os.path.join(dir, ofpath), compression="gzip")

    levmeta = alt_stage_object(x.get_levels(), dir, path + "/levels", is_child = True)
    levres = write_metadata(levmeta, dir)

    fmeta = {
        "length": len(x),
        "compression": "gzip",
    }
#    if x.get_ordered():
#        fmeta["ordered"] = True

    return {
        "$schema": "string_factor/v1.json",
        "path": ofpath,
        "is_child": is_child,
        "factor": fmeta,
        "string_factor": {
            "levels": { "resource": levres },
        }
    }
