from typing import Any, Callable, Dict, Optional

from .save_object import save_object

ALT_SAVE_OBJECT_FUNCTION = save_object


def alt_save_object_function(fun: Optional[Callable] = None) -> Callable:
    """Get or set the alternative saving function for use by
    :py:meth:`~alt_save_object`. Typically set by applications prior to
    saving for customization, e.g., to save extra metadata.

    Args:
        fun:
            The alternative saving function. This should accept the same
            arguments and return the same value as
            :py:meth:`~dolomite_base.save_object.save_object`.

    Returns:
        If ``fun = None``, the current setting of the alternative saving
        function is returned. 

        Otherwise, the alternative saving function is set to ``fun``,
        and the previous function is returned.
    """
    global ALT_SAVE_OBJECT_FUNCTION
    if fun is None:
        return ALT_SAVE_OBJECT_FUNCTION
    else:
        old = ALT_SAVE_OBJECT_FUNCTION
        ALT_SAVE_OBJECT_FUNCTION = fun
        return old


def alt_save_object(x: Any, path: str, **kwargs) -> Dict[str, Any]:
    """Wrapper around :py:meth:`~dolomite_base.save_object.save_object` that
    respects application-defined overrides from
    :py:meth:`~alt_save_object_function`.  
    
    This allows applications to
    customize the saving process for some or all of the object classes,
    assuming that developers of dolomite extensions (and the associated
    ``save_object`` methods) use ``alt_save_object`` internally for saving
    child objects instead of ``save_object``.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to a directory to save `x`.

        kwargs: 
            Further arguments to be passed to individual methods.

    Returns:
        `x` is saved to `path`.
    """
    fun = alt_save_object_function()
    return fun(x, path, **kwargs)
