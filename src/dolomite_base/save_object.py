from typing import Any
from functools import singledispatch, wraps
from .validate_object import validate_object
from importlib import import_module


_save_object_implementations = {
    "GenomicRanges": "dolomite_ranges",
    "GenomicRangesList": "dolomite_ranges",
    "SeqInfo": "dolomite_ranges",

    "ndarray": "dolomite_matrix",
    "csc_matrix": "dolomite_matrix",
    "csr_matrix": "dolomite_matrix",
    "coo_matrix": "dolomite_matrix",
    "DelayedArray": "dolomite_matrix",

    "SummarizedExperiment": "dolomite_se",
    "RangedSummarizedExperiment": "dolomite_se",

    "SingleCellExperiment": "dolomite_sce",
    "MultiAssayExperiment": "dolomite_mae"
}


@singledispatch
def save_object(x: Any, path: str, **kwargs):
    """
    Save an object to its on-disk representation. **dolomite** extensions
    should define methods for this generic to stage different object classes.

    Saver methods may accept additional arguments in the ``kwargs``; these
    should be prefixed by the object type to avoid conflicts (see
    :py:func:`~dolomite_base.save_data_frame.save_data_frame` for examples).

    Saver methods should also use the :py:func:`~validate_saves` decorator
    to ensure that the generated output in ``path`` is valid.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to the output directory.

        kwargs: 
            Further arguments to be passed to individual methods.

    Returns:
        `x` is saved to `path`.
    """

    if hasattr(type(x), "mro"):
        import sys
        hierarchy = type(x).mro()
        for y in hierarchy:
            nm = y.__name__
            if nm not in _save_object_implementations:
                continue

            pkg = _save_object_implementations[nm]
            try:
                import_module(pkg) # this should hopefully register the methods to the dispatcher.
            except:
                raise ModuleNotFoundError("cannot find '" + pkg + "', which contains a 'save_object' method for type '" + type(x).__name__ + "'")
            return save_object(x, path, **kwargs)

    raise NotImplementedError("'save_object' is not implemented for type '" + type(x).__name__ + "'")


def validate_saves(fn):
    """
    Decorator to validate the output of :py:func:`~save_object`.

    Args:
        fn: Function that implements a method for ``save_object``.

    Returns:
        A wrapped version of the function that validates the directory
        containing the on-disk representation of the saved object.
    """
    @wraps(fn)
    def wrapper(x, path, **kwargs):
        out = fn(x, path, **kwargs)
        validate_object(path)
        return out
    return wrapper
