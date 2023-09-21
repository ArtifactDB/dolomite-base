from typing import Any
from functools import singledispatch
import os
import json


@singledispatch
def acquire_metadata(project: Any, path: str) -> dict[str, Any]:
    """Acquire metadata for a resource.

    Arguments:
        project: 
            Any value that specifies the project of interest.
            For example, this may be a path to a staging directory,
            but other applications may provide their own classes,
            e.g., identify projects on remotes. 

        path:
            Relative path to the resource of interest inside the project.

    Returns:
        Dictionary representing the JSON metadata.
    """
    raise NotImplementedError("'acquire_metadata' for " + str(type(project)) + " has not been implemented")


@acquire_metadata.register
def acquire_metadata_from_dir(project: str, path: str) -> str:
    if not path.endswith(".json"):
        path += ".json"
    fpath = os.path.join(project, path)
    with open(fpath, "r") as handle:
        return json.load(handle)
