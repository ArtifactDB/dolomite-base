import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "dolomite-base"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .stage_object import stage_object
from .stage_data_frame import stage_data_frame, choose_data_frame_format
from .write_metadata import write_metadata
from .load_object import *
from .stage_simple_list import stage_simple_list, choose_simple_list_format
from .load_data_frame import load_csv_data_frame, load_hdf5_data_frame
from .load_simple_list import load_json_simple_list, load_hdf5_simple_list
from .acquire_metadata import acquire_metadata
from .acquire_file import acquire_file
