from typing import Dict, Any
import os
import json


def read_object_file(path: str) -> Dict[str, Any]:
    """
    Read the ``OBJECT`` file in each directory, which provides some high-level
    metadata of the object represented by that directory. It is guaranteed to
    have a ‘type’ property that specifies the object type; individual objects
    may add their own information to this file. 

    Args:
        path: 
            Path to a directory containing the object.

    Returns:
        Dictionary containing the object metadata.
    """
    with open(os.path.join(path, "OBJECT"), "rb") as handle:
        metadata = json.load(handle)
    return metadata
