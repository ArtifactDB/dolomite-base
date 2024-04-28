from biocutils import StringList, IntegerList, FloatList, BooleanList
import os
import h5py
import numpy

from .save_object import save_object, validate_saves
from .save_object_file import save_object_file
from . import _utils_string as strings
from . import write_vector_to_hdf5 as write


@save_object.register
@validate_saves
def save_atomic_vector_from_string_list(x: StringList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.StringList.StringList` objects
    to their corresponding file representation, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to save the object.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)
    save_object_file(path, "atomic_vector", { "atomic_vector": { "version": "1.0" } })

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "string"
        write.write_string_vector_to_hdf5(ghandle, "values", x.as_list())
        nms = x.get_names()
        if nms is not None:
            strings.save_fixed_length_strings(ghandle, "names", nms.as_list())

    return


@save_object.register
@validate_saves
def save_atomic_vector_from_integer_list(x: IntegerList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.IntegerList.IntegerList` objects
    to their corresponding file representation, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to save the object.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)
    save_object_file(path, "atomic_vector", { "atomic_vector": { "version": "1.0" } })

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        dset = write.write_integer_vector_to_hdf5(ghandle, "values", x.as_list(), allow_float_promotion=True)

        if numpy.issubdtype(dset, numpy.floating):
            ghandle.attrs["type"] = "number"
            dset.attrs.create("_python_original_type", "biocutils.IntegerList")
        else:
            ghandle.attrs["type"] = "integer"

        nms = x.get_names()
        if nms is not None:
            strings.save_fixed_length_strings(ghandle, "names", nms.as_list())

    return


@save_object.register
@validate_saves
def save_atomic_vector_from_float_list(x: FloatList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.FloatList.FloatList` objects
    to their corresponding file representation, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to save the object.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)
    save_object_file(path, "atomic_vector", { "atomic_vector": { "version": "1.0" } })

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "number"
        write.write_float_vector_to_hdf5(ghandle, "values", x.as_list())
        nms = x.get_names()
        if nms is not None:
            strings.save_fixed_length_strings(ghandle, "names", nms.as_list())

    return


@save_object.register
@validate_saves
def save_atomic_vector_from_boolean_list(x: BooleanList, path: str, **kwargs): 
    """Method for saving :py:class:`~biocutils.BooleanList.BooleanList` objects
    to their corresponding file representation, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to save the object.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)
    save_object_file(path, "atomic_vector", { "atomic_vector": { "version": "1.0" } })

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "boolean"
        write.write_boolean_vector_to_hdf5(ghandle, "values", x.as_list())
        nms = x.get_names()
        if nms is not None:
            strings.save_fixed_length_strings(ghandle, "names", nms.as_list())

    return
