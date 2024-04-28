from typing import Dict, Any
import os
import json


def save_object_file(path: str, object_type: str, extra: Dict[str, Any] = {}):
    """
    Saves object-specific metadata into the ``OBJECT`` file inside each
    directory, to be used by, e.g., :py:func:`~.read_object_file`.

    Args:
        path:
            Path to the directory representing an object. An ``OBJECT`` file
            will be created inside this directory.

        object_type: 
            Type of the object.

       extra:
            Extra metadata to be written to the ``OBJECT`` file in ``path``.
            Any entry named ``type`` will be overwritten by ``object_type``.
    """
    to_save = { **extra }
    to_save["type"] = object_type
    with open(os.path.join(path, "OBJECT"), 'w', encoding="utf-8") as handle:
        json.dump(to_save, handle)
