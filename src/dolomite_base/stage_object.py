from typing import Any
from functools import singledispatch


@singledispatch
def stage_object(x: Any, dir: str, path: str, **kwargs) -> dict[str, Any]:
    """Save an object to file in a staging directory. **dolomite** extensions
    should define methods for this generic to stage different object classes.

    Arguments:
        x: Object to be staged.

        dir: Path to a staging directory.

        path: Relative path inside ``dir`` where ``x`` is to be saved.
            This will be used to create a subdirectory inside ``dir``.

        kwargs: Further arguments to be passed to individual methods.

    Returns:
        Metadata for ``x``, which can be edited by calling methods and then
        saved with :py:meth:`~dolomite_base.write_metadata.write_metadata`.
        This should contain at least a ``$schema`` property, specifying the
        schema to use for validation of the metadata; and a ``path`` property,
        specifying the relative path to the file representation of ``x``.
    """
    raise NotImplementedError("'stage_object' is not implemented for " + str(type(x)))
