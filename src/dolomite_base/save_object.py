from typing import Any
from functools import singledispatch, wraps
from .validate_object import validate_object


@singledispatch
def save_object(x: Any, path: str, **kwargs):
    """Save an object to its on-disk representation. **dolomite** extensions
    should define methods for this generic to stage different object classes.

    Arguments:
        x: Object to be saved.

        path: Path to the output directory.

        kwargs: Further arguments to be passed to individual methods.

    Returns:
        `x` is saved to `path`.
    """
    raise NotImplementedError("'save_object' is not implemented for " + str(type(x)))


def validate_saves(fn):
    @wraps(fn)
    def wrapper(x, path, **kwargs):
        out = fn(x, path, **kwargs)
        validate_object(path)
        return out
    return wrapper
