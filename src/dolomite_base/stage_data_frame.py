from typing import Any
import os
from biocframe import BiocFrame

from .stage_object import stage_object
from ._stage_csv_data_frame import _stage_csv_data_frame
from .write_metadata import write_metadata


@stage_object.register
def stage_data_frame(x: BiocFrame, dir: str, path: str, is_child: bool = False, **kwargs) -> dict[str, Any]:
    os.mkdir(os.path.join(dir, path))
    meta, other = _stage_csv_data_frame(x, dir, path, is_child=is_child)

    for i in other:
        more_meta = stage_object(x.column(i), dir, path, is_child = True)
        resource_stub = write_metadata(more_meta, dir=dir)
        meta["data_frame"]["columns"][i]["resource"] = resource_stub
    return meta
