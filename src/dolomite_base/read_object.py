from typing import Any, Optional
from importlib import import_module
import os
import json

registry = {
    "atomic_vector": "dolomite_base.read_atomic_vector",
    "string_factor": "dolomite_base.read_string_factor",
    "simple_list": "dolomite_base.read_simple_list",
    "data_frame": "dolomite_base.read_data_frame",
    "dense_array": "dolomite_matrix.read_dense_array",
    "compressed_sparse_matrix": "dolomite_matrix.read_compressed_sparse_matrix",
}


def read_object(path: str, metadata: Optional[dict] = None, **kwargs) -> Any:
    """Read an object from its on-disk representation.

    Args:
        path: 
            Path to a directory containing the object.

        metadata: 
            Metadata for the object. This is read from the `OBJECT` file if None.

        kwargs: 
            Further arguments, passed to individual methods.

    Returns:
        Some kind of object.
    """
    if metadata is None:
        with open(os.path.join(path, "OBJECT"), "rb") as handle:
            metadata = json.load(handle)

    tt = metadata["type"]
    if tt not in registry:
        raise NotImplementedError("could not find a Python command to read '" + tt + "'")

    command = registry[tt]
    if isinstance(command, str): 
        first_period = command.find(".")
        mod = import_module(command[:first_period])
        command = getattr(mod, command[first_period + 1:])
        registry[tt] = command

    return command(path, metadata, **kwargs)
