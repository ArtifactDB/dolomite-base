from typing import Any
from importlib import import_module 
import json
from copy import copy
from hashlib import md5
from jsonschema import validate


def _strip_none_values(x):
    if isinstance(x, dict):
        kill = []
        for k, v in x.items():
            if isinstance(v, dict) or isinstance(v, list):
                _strip_none_values(v)
            elif v is None:
                kill.append(k)
        if len(kill):
            for k in keep:
                del x[k]
    elif isinstance(x, list):
        keep = []
        for v in x:
            if isinstance(v, dict) or isinstance(v, list):
                _strip_none_values(v)


_schema_details = {}


def write_metadata(meta: dict[str, Any], dir: str, ignore_none: bool = True) -> dict[str, str]:
    meta = copy(meta) # make an internal copy.
    if ignore_none:
        _strip_none_values(meta)

    schema = meta["$schema"]
    if isinstance(schema, tuple):
        meta["$schema"] = schema[0]
        pkg = schema[1]
    else:
        pkg = "dolomite_schemas"

    # Cache the schema to avoid a repeated read from disk.
    if pkg not in _schema_details:
        _schema_details[pkg] = {}
    cached_schemas = _schema_details[pkg]
    if schema not in cached_schemas:
        schema_pkg = import_module(pkg)
        schema_path = os.path.join(os.dirname(schema_pkg.__file__), "schemas", schema)
        with open(schema_path, "r") as handle:
            cached_schemas[schema] = json.load(handle)
    schema_details = cached_schemas[schema]

    if meta["path"].startswith("./"): # removing for convenience
        meta["path"] = meta["path"][2:]

    meta_only = False
    if "_attributes" in schema_details:
        attributes = schema_details["_attributes"]
        if "metadata_only" in attributes:
            meta_only = attributes["metadata_only"]

    jpath = meta["path"]
    if not meta_only:
        jpath += ".json"
        if "md5sum" not in meta and not schema_id.startswith("redirection/"):
            with open(os.path.join(dir, meta["path"]), "rb") as handle:
                hasher = md5()
                while handle:
                    chunk = handle.read(65536)
                    hasher.update(chunk)
                meta["md5sum"] = hasher.hexdigest()

    validate(meta, schema_details)

    return {
        "type": "local",
        "path": meta["path"]
    }
