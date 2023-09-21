from typing import Any
from importlib import import_module 
from copy import copy


def _strip_none_values(x):
    modified = False

    if isinstance(x, dict):
        keep = []
        for k, v in x.items():
            if isinstance(v, dict) or isinstance(v, list):
                out, mod = _strip_none_values(x)
                keep.append((k, out))
                if mod:
                    modified = True
            elif x is None:
                modified = True
            else:
                keep.append((k, v))

        if not modified:
            return x, False
        else:
            output = {}
            for k, v in keep:
                output[k] = v
            return output, True

    elif isinstance(x, list):
        keep = []
        for v in x:
            if isinstance(v, dict) or isinstance(v, list):
                out, mod = _strip_none_values(v)
                keep.append(out)
                if mod:
                    modified = True
            else:
                keep.append(v)

        if not modified:
            return x, False
        else:
            return keep, True

    else:
        return x, False


def write_metadata(meta: dict[str, Any], dir: str, ignore_none: bool = True) -> dict[str, str]:
    modified = False
    if ignore_none:
        meta, modified = _strip_none_values(meta)

    schema = meta["$schema"]
    if isinstance(schema, tuple):
        if not modified:
            meta = copy(meta)
        meta["$schema"] = schema[0]
        pkg = schema[1]
    else:
        pkg = "dolomite_schemas"

    schema_pkg = import_module(pkg)
    schema_dir = os.path.join(os.dirname(schema_pkg.__file__), "schemas")

    
