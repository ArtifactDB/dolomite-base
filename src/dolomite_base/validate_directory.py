from typing import List
import os

from .list_objects import list_objects
from .validate_object import validate_object


def validate_directory(dir: str) -> List[str]:
    """Check whether each object in a directory is valid by calling :py:func:`~dolomite_base.validate_object.validate_object` on each non-child object.

    Args:
        dir:
            Path to a directory with subdirectories populated by :py:func:`~dolomite_base.save_object.save_object`.
            ``dir`` itself may also correspond to an object.

    Returns:
        List of the paths inside ``dir`` that were validated.
        This contains only ``None`` if ``dir`` itself corresponds to an object.
    """
    objects = list_objects(dir)
    paths = objects.get_column("path")
    for x in paths:
        try:
            validate_object(os.path.join(dir, x))
        except Exception as e:
            raise ValueError("failed to validate '" + x + "'; " + str(e))
    return paths
