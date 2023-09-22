from typing import Any
from importlib import import_module
import os

from ._schemas import _hunt_for_schemas


def custom_load_object_helper(meta: dict, project: Any, locations: list, memory: dict[str, Any], **kwargs) -> Any:
    """Helper function to create application-specific variants of
    :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this object.

        project: 
            Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        locations: 
            List of names of packages to be searched for object schemas. It is
            expected that the schema for each object specifies the function
            required to create that object from its file representations.

        memory:
            Cache of restoration functions for each object schema. This avoids
            redundant look-ups if the same ``memory`` is recycled across calls.

        kwargs: Further arguments, passed to individual methods.

    Returns:
        Some kind of object.
    """
    schema = meta["$schema"]

    if schema not in memory:
        schema_details = _hunt_for_schemas(locations, schema)
        command = None
        if "_attributes" in schema_details:
            attr_meta = schema_details["_attributes"]
            if "restore" in attr_meta:
                res_meta = attr_meta["restore"]
                if "python" in res_meta:
                    command = res_meta["python"]

        if command is None:
            # TODO: replace these in the schemas themselves, once everything has settled down.
            if schema == "csv_data_frame/v1.json":
                command = "dolomite_base.load_csv_data_frame"
            else:
                raise NotImplementedError("could not find a Python context to restore '" + schema + "'")

        first_period = command.find(".")
        mod = import_module(command[:first_period])
        memory[schema] = getattr(mod, command[first_period + 1:])

    return memory[schema](meta, project, **kwargs)


"""Default locations (as package names) in which to search for schemas.
Each package should have a `schemas/` subdirectory containing all schemas."""
DEFAULT_SCHEMA_LOCATIONS = [ "dolomite_schemas" ]


_schema_restoration = {}


def load_object(meta: dict, project: Any, **kwargs) -> Any:
    """Load an object from a resource inside a project. This uses the schemas
    in `DEFAULT_LOCATIONS` to identify the restoration functions.

    Args:
        meta: Metadata for this object.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, passed to individual methods.

    Returns:
        Some kind of object.
    """
    return custom_load_object_helper(
        meta, 
        project, 
        locations = DEFAULT_SCHEMA_LOCATIONS, 
        memory = _schema_restoration, 
        **kwargs
    ) 
