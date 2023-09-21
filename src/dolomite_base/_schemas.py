import json
from importlib import import_module 
import os

_schema_cache = {}

def _fetch_schema(pkg, schema):
    if pkg not in _schema_cache:
        _schema_cache[pkg] = {}

    current_cache = _schema_cache[pkg]
    if schema not in current_cache:
        schema_pkg = import_module(pkg)
        schema_path = os.path.join(os.path.dirname(schema_pkg.__file__), "schemas", schema)
        if os.path.exists(schema_path):
            with open(schema_path, "r") as handle:
                current_cache[schema] = json.load(handle)
        else:
            current_cache[schema] = None

    return current_cache[schema]


def _hunt_for_schemas(locations, schema, fallback = None):
    schema_details = None
    for pkg in locations:
        schema_details = _fetch_schemas(pkg, schema)
        if schema_details is not None:
            break

    if schema_details is None:
        if fallback is None:
            raise FileNotFoundError("could not find schema '" + schema + "' in any package location")
        schema_details = fallback(schema)

    return schema_details
