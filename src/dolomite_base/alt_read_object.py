from typing import Any, Optional, Callable, Dict
from .read_object import read_object


ALT_READ_OBJECT_FUNCTION = read_object


def alt_read_object_function(fun: Optional[Callable] = None) -> Callable:
    """
    Get or set the alternative reading function for use by
    :py:meth:`~alt_read_object`. Typically set by applications prior to reading
    for customization, e.g., to attach more metadata to the loaded object.

    Args:
        fun:
            The alternative reading function. This should accept the same
            arguments and return the same value as
            :py:meth:`~dolomite_base.read_object.read_object`.

    Returns:
        If ``fun = None``, the current setting of the alternative reading
        function is returned. 

        Otherwise, the alternative reading function is set to ``fun``,
        and the previous function is returned.
    """
    global ALT_READ_OBJECT_FUNCTION
    if fun is None:
        return ALT_READ_OBJECT_FUNCTION
    else:
        old = ALT_READ_OBJECT_FUNCTION
        ALT_READ_OBJECT_FUNCTION = fun
        return old


def alt_read_object(path: str, metadata: Optional[Dict] = None, **kwargs) -> Any:
    """Wrapper around :py:meth:`~dolomite_base.read_object.read_object` 
    that respects application-defined overrides from
    :py:meth:`~alt_read_object_function`.  This allows applications to
    customize the reading process for some or all of the object classes,
    assuming that developers of dolomite extensions (and the associated
    functions called by ``read_object``) use ``alt_read_object`` internally for
    staging child objects instead of ``read_object``.

    Args:
        path: 
            Directory containing the object to load.

        metadata: 
            Metadata for the object. If None, this should be read from the
            ``OBJECT`` file inside ``path``.

        kwargs: 
            Further arguments, passed to individual methods.

    Returns:
        Some kind of object.
    """
    fun = alt_read_object_function()
    return fun(path, metadata=metadata, **kwargs)
