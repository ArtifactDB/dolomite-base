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
from .alt_stage_object import alt_stage_object, alt_stage_object_function
from .stage_data_frame import stage_data_frame, choose_data_frame_format, convert_data_frame_list_to_vector
from .write_metadata import write_metadata
from .load_object import *
from .alt_load_object import alt_load_object, alt_load_object_function
from .stage_simple_list import stage_simple_list, choose_simple_list_format
from .load_data_frame import load_csv_data_frame, load_hdf5_data_frame
from .load_simple_list import load_json_simple_list, load_hdf5_simple_list
from .acquire_metadata import acquire_metadata
from .acquire_file import acquire_file
from .write_csv import read_csv, write_csv
from .stage_atomic_vector import stage_string_list
from .load_atomic_vector import load_atomic_vector
from .stage_string_factor import stage_string_factor
from .load_string_factor import load_string_factor
