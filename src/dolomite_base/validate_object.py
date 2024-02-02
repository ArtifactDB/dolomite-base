from typing import Optional, Dict, Callable, Literal

from . import lib_dolomite_base as lib


validate_object_registry = {}


def validate_object(path: str, metadata: Optional[Dict] = None):
    """
    Validate an on-disk representation of an object, typically using validators
    based on the **takane** specifications. 

    Applications may also register their own validators by adding entries to
    ``validate_object_registry``. Each key should be the object type and each
    value should be a function that accepts a path to a directory (string) and
    JSON-derived metadata (dictionary). The function should raise an error if
    the object in the directory is not valid for the specified object type.

    Args:
        path: 
            Path to the directory containing the object's representation.

        metadata:
            Metadata for the object. If None, this is read from the ``OBJECT``
            file in the ``path``.

    Raise:
        Error if the validation fails.
    """
    lib.validate(path, metadata, validate_object_registry)
    return

