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

from .save_object import save_object
from .validate_object import validate_object
from .save_atomic_vector import save_atomic_vector_from_string_list, save_atomic_vector_from_integer_list, save_atomic_vector_from_float_list, save_atomic_vector_from_boolean_list
from .save_string_factor import save_string_factor
from .save_simple_list import save_simple_list_from_list, save_simple_list_from_dict, save_simple_list_from_NamedList
from .save_data_frame import save_data_frame
from .read_atomic_vector import read_atomic_vector
from .read_string_factor import read_string_factor
from .read_simple_list import read_simple_list
from .read_data_frame import read_data_frame
from .read_object import read_object
