from typing import Any, Optional, Callable, Union
from .stage_object import stage_object


ALT_STAGE_OBJECT_FUNCTION = stage_object


def alt_stage_object_function(fun: Optional[Callable] = None) -> Union[Callable, None]:
    """
    Get or set the alternative staging function for use by
    :py:meth:`~alt_stage_object`. Typically set by applications prior to
    staging for customization, e.g., to save extra metadata.

    Arguments:
        fun:
            The alternative staging function. This should accept the same
            arguments and return the same value as
            :py:meth:`~dolomite_base.stage_object.stage_object`.

    Returns:
        If ``fun = None``, the current setting of the alternative staging
        function is returned. 

        Otherwise, the alternative staging function is set to ``fun``,
        and the previous function is returned.
    """
    global ALT_STAGE_OBJECT_FUNCTION
    if fun is None:
        return ALT_STAGE_OBJECT_FUNCTION
    else:
        old = ALT_STAGE_OBJECT_FUNCTION
        ALT_STAGE_OBJECT_FUNCTION = fun
        return old


def alt_stage_object(x: Any, dir: str, path: str, **kwargs) -> dict[str, Any]:
    """
    Wrapper around :py:meth:`~dolomite_base.stage_object.stage_object` that
    respects application-defined overrides from
    :py:meth:`~alt_stage_object_function`.  This allows applications to
    customize the staging process for some or all of the object classes,
    assuming that developers of dolomite extensions (and the associated
    ``stage_object`` methods) use ``alt_stage_object`` internally for staging
    child objects instead of ``stage_object``.

    Arguments:
        x: Object to be staged.

        dir: Path to a staging directory.

        path: Relative path inside ``dir`` where ``x`` is to be saved.
            This will be used to create a subdirectory inside ``dir``.

        kwargs: Further arguments to be passed to individual methods.

    Returns:
        Metadata for ``x``, see
        :py:meth:`~dolomite_base.stage_object.stage_object` for details.
    """
    fun = alt_stage_object_function()
    return fun(x, dir, path, **kwargs)
