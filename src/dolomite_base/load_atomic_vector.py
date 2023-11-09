from typing import Any, Union
from biocutils import StringList
from numpy import ndarray

from .acquire_file import acquire_file
from .write_csv import read_csv
from ._utils import _is_gzip_compressed

def load_atomic_vector(meta: dict[str, Any], project: Any, **kwargs) -> Union[StringList, ndarray]:
    """
    Load an atomic vector from a CSV file. In general, this function should not
    be called directly but instead via
    :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this string list.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, passed to nested objects.

    Returns:
        An atomic vector. For string types, this will be a ``StringList``
        object, otherwise it will be a 1-dimensional NumPy array.
    """
    fpath = acquire_file(project, meta["path"])
    gzip = _is_gzip_compressed(meta, "atomic_vector") # check it's not crazy.
    dump_names, dump_fields = read_csv(fpath, meta["atomic_vector"]["length"], compression="gzip" if gzip else "none") 

    # Better hope it's not named, we don't have a good way to deal with that right now.
    # Probably should just emit a warning about lost names.
    vec = dump_fields[-1]

    vectype = meta["atomic_vector"]["type"]
    if vectype == "string":
        return StringList(vec)
    elif vectype == "integer":
        return vec.astype(numpy.int32)
    else:
        return vec
