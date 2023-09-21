from typing import Any
from hashlib import md5
from jsonschema import validate
from ._schemas import _fetch_schema


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


def write_metadata(meta: dict[str, Any], dir: str, ignore_none: bool = True) -> dict[str, str]:
    meta = copy(meta) # make an internal copy.
    if ignore_none:
        _strip_none_values(meta)

    schema = meta["$schema"]
    if isinstance(schema, tuple):
        meta["$schema"] = schema[0]
        pkg = schema[1]
    else:
        pkg = "dolomite.schemas"
    schema_details = _fetch_schema(pkg, schema)

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
