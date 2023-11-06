from typing import Any, Optional, Callable, Union
from .load_object import load_object


ALT_LOAD_OBJECT_FUNCTION = load_object


def alt_load_object_function(fun: Optional[Callable] = None) -> Union[Callable, None]:
    """
    Get or set the alternative loading function for use by
    :py:meth:`~alt_load_object`. Typically set by applications prior to loading
    for customization, e.g., to attach more metadata to the loaded object.

    Arguments:
        fun:
            The alternative loading function. This should accept the same
            arguments and return the same value as
            :py:meth:`~dolomite_base.load_object.load_object`.

    Returns:
        If ``fun = None``, the current setting of the alternative loading
        function is returned. 

        Otherwise, the alternative loading function is set to ``fun``,
        and the previous function is returned.
    """
    global ALT_LOAD_OBJECT_FUNCTION
    if fun is None:
        return ALT_LOAD_OBJECT_FUNCTION
    else:
        old = ALT_LOAD_OBJECT_FUNCTION
        ALT_LOAD_OBJECT_FUNCTION = fun
        return old


def alt_load_object(meta: dict, project: Any, **kwargs) -> Any:
    """
    Wrapper around :py:meth:`~dolomite_base.load_object.load_object` that
    respects application-defined overrides from
    :py:meth:`~alt_load_object_function`.  This allows applications to
    customize the loading process for some or all of the object classes,
    assuming that developers of dolomite extensions (and the associated
    functions called by ``load_object``) use ``alt_load_object`` internally for
    staging child objects instead of ``load_object``.

    Arguments:
        meta: Metadata for this object.

        project: 
            Value specifying the project of interest, see
            :py:meth:`~dolomite_base.load_object.load_object` for details.

        kwargs: Further arguments, passed to individual methods.

    Returns:
        Some kind of object.
    """
    fun = alt_load_object_function()
    return fun(meta, project, **kwargs)

