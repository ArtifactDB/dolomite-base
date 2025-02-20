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

from .save_object import save_object, validate_saves
from .alt_save_object import alt_save_object, alt_save_object_function
from .save_object_file import save_object_file

from .save_atomic_vector import save_atomic_vector_from_string_list, save_atomic_vector_from_integer_list, save_atomic_vector_from_float_list, save_atomic_vector_from_boolean_list
from .save_string_factor import save_string_factor
from .save_simple_list import save_simple_list_from_list, save_simple_list_from_dict, save_simple_list_from_NamedList
from .save_data_frame import save_data_frame

from .read_object import read_object, read_object_registry
from .alt_read_object import alt_read_object, alt_read_object_function
from .read_object_file import read_object_file

from .read_atomic_vector import read_atomic_vector
from .read_string_factor import read_string_factor
from .read_simple_list import read_simple_list
from .read_data_frame import read_data_frame

from .validate_object import validate_object, validate_object_registry
from .list_objects import list_objects
from .validate_directory import validate_directory

from .choose_missing_placeholder import choose_missing_integer_placeholder, choose_missing_float_placeholder, choose_missing_string_placeholder
from .write_vector_to_hdf5 import write_string_vector_to_hdf5, write_float_vector_to_hdf5, write_integer_vector_to_hdf5, write_boolean_vector_to_hdf5
from .load_vector_from_hdf5 import load_vector_from_hdf5
