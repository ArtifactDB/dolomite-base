from ._schema_cache import _hunt_for_schemas
import os


def custom_load_object_helper(meta: dict, project: Any, locations: list, memory: dict[str, Any], **kwargs) -> Any:
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
            raise NotImplementedError("could not find a Python context to restore '" + schema + "'")

        first_period = command.find(".")
        mod = importlib.import_module(command[:first_period])
        memory[schema] = getattr(mod, command[first_period + 1:])

    return memory[schema](meta, project, **kwargs)


default_locations = [ "dolomite.schemas" ]


_schema_restoration = {}


def load_object(meta: dict, project: Any, **kwargs) -> Any:
    return custom_load_object_helper(
        meta, 
        project, 
        locations = default_locations, 
        memory = _schema_restoration, 
        **kwargs
    ) 
