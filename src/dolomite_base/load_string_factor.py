from typing import Any
from biocutils import Factor

from .acquire_file import acquire_file
from .acquire_metadata import acquire_metadata
from .alt_load_object import alt_load_object
from .write_csv import read_csv
from ._utils import _is_gzip_compressed


def load_string_factor(meta: dict[str, Any], project: Any, **kwargs) -> Factor:
    """
    Load a string factor from a CSV file. In general, this function should not
    be called directly but instead via
    :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this string factor.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        A :py:class:`~biocutils.Factor.Factor` object.
    """
    fpath = acquire_file(project, meta["path"])
    gzip = _is_gzip_compressed(meta, "factor")
    dump_names, dump_fields = read_csv(fpath, meta["factor"]["length"], compression=("gzip" if gzip else "none"))

    # Better hope it's not named, we don't have a good way to deal with that right now.
    # Probably should just emit a warning about lost names.
    codes = dump_fields[-1]

    lmeta = acquire_metadata(project, meta["string_factor"]["levels"]["resource"]["path"])
    levels = alt_load_object(lmeta, project)

    ordered = False
    if "ordered" in meta["factor"]:
        ordered = meta["factor"]["ordered"]

    return Factor(codes, levels, ordered=ordered)
