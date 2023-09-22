from typing import Any
from functools import singledispatch
import os
import json


@singledispatch
def acquire_metadata(project: Any, path: str) -> dict[str, Any]:
    """Acquire metadata for a resource inside a project. Applications should
    define methods for this generic to acquire metadata from different places.

    Args:
        project: 
            Any value that specifies the project of interest.  By default, this
            should be a path to a staging directory, but other applications may
            provide their own classes, e.g., to identify projects on remotes. 

        path:
            Relative path to the resource of interest inside the project.

    Returns:
        Dictionary representing the JSON metadata.
    """
    raise NotImplementedError("'acquire_metadata' for " + str(type(project)) + " has not been implemented")


@acquire_metadata.register
def acquire_metadata_from_dir(project: str, path: str) -> str:
    """Acquire metadata for a resource inside a local staging directory.

    Args:
        project: 
            Path to a staging directory.

        path:
            Relative path to the resource of interest inside the project.

    Returns:
        Dictionary representing the JSON metadata.
    """
    if not path.endswith(".json"):
        path += ".json"
    fpath = os.path.join(project, path)
    with open(fpath, "r") as handle:
        return json.load(handle)
