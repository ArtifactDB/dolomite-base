from typing import Any
from functools import singledispatch
import os


@singledispatch
def acquire_file(project: Any, path : str) -> str:
    """Acquire the path to a resource's file.

    Arguments:
        project: 
            Any value that specifies the project of interest.
            For example, this may be a path to a staging directory,
            but other applications may provide their own classes,
            e.g., identify projects on remotes. 

        path:
            Relative path to the resource of interest inside the project.

    Returns:
        File path to the resource, possibly after downloading and caching.
    """
    raise NotImplementedError("'acquire_file' for " + str(type(project)) + " has not been implemented")


@acquire_file.register
def acquire_file_from_dir(project: str, path: str) -> str:
    return os.path.join(project, path)
