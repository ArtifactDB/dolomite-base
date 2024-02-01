from typing import Optional, Dict, Callable, Literal

from . import lib_dolomite_base as lib


def validate_object(path: str, metadata: Optional[Dict] = None):
    """
    Validate an on-disk representation of an object, typically using validators
    based on the **takane** specifications. Applications may also register their
    own validators via :py:func:`~register_validate_object_function`. 

    Args:
        path: 
            Path to the directory containing the object's representation.

        metadata:
            Metadata for the object. If None, this is read from the ``OBJECT``
            file in the ``path``.

    Raise:
        Error if the validation fails.
    """
    lib.validate(path, metadata)
    return


def register_validate_object_function(type: str, fun: Optional[Callable] = None, existing: Literal["old", "new", "error"] = "old"): 
    """
    Register a validator function for use by :py:func:`~validate_object`.

    Args:
        type:
            Object type, as specified in the ``type`` property of the
            ``OBJECT`` file.

        fun:
            Function that accepts a path to a directory (string) and
            JSON-derived metadata (dictionary) and raises an error if the
            object in the directory is not valid for the specified object type.

            Alternatively None, in which case any previously registered
            validator for ``type`` is removed.

        existing:
            What to do when a function is already registered for ``type`` -
            keep the ``old`` function, use the ``new`` function, or 
            raise an ``error``.
    """
    if fun is None:
        lib.deregister_validate_function(type)
    else:
        lib.register_validate_function(type, fun, existing)
