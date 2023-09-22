from typing import Any
from functools import singledispatch
import os


@singledispatch
def acquire_file(project: Any, path : str) -> str:
    """Acquire the file path for a resource. Applications should define methods
    for this generic to acquire metadata from other places.

    Arguments:
        project: 
            Any value that specifies the project of interest.  By default, this
            should be a path to a staging directory, but other applications may
            provide their own classes, e.g., identify projects on remotes. 

        path:
            Relative path to the resource of interest inside the project.

    Returns:
        File path to the resource, possibly after downloading and caching.
    """
    raise NotImplementedError("'acquire_file' for " + str(type(project)) + " has not been implemented")


@acquire_file.register
def acquire_file_from_dir(project: str, path: str) -> str:
    """Acquire the file path for a resource in a local staging directory.

    Arguments:
        project: 
            Path to a staging directory.

        path:
            Relative path to the resource of interest inside the project.

    Returns:
        File path to the resource, possibly after downloading and caching.
    """
    return os.path.join(project, path)
