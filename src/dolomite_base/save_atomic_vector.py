from typing import Optional
from biocutils import StringList, IntegerList, FloatList, BooleanList
import os
import h5py
import numpy

from .save_object import save_object, validate_saves
from .save_object_file import save_object_file
from . import _utils_string as strings
from . import write_vector_to_hdf5 as write
from . import choose_missing_placeholder as ch


@save_object.register
@validate_saves
def save_atomic_vector_from_string_list(x: StringList, path: str, string_list_vls: Optional[bool] = False, **kwargs): 
    """Method for saving :py:class:`~biocutils.StringList.StringList` objects to their corresponding file representation,
    see :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: 
            Object to be saved.

        path: 
            Path to save the object.

        string_list_vls:
            Whether to save variable-length strings into a custom VLS array format.
            If ``None``, this is automatically determined by comparing the required storage with that of fixed-length strings.

        kwargs: 
            Further arguments, ignored.

    Returns:
        `x` is saved to `path`.
    """
    os.mkdir(path)
    save_object_file(path, "atomic_vector", { "atomic_vector": { "version": "1.1" } })

    # Gathering some preliminary statistics.
    has_missing = False
    for val in x:
        if val is None:
            has_missing = True
    if has_missing:
        placeholder = ch.choose_missing_string_placeholder(x)
        placeholder_encoded = placeholder.encode("UTF-8")

    x_encoded = [None] * len(x)
    if has_missing:
        for i, val in enumerate(x):
            if val is None:
                x_encoded[i] = placeholder_encoded
            else:
                x_encoded[i] = val.encode("UTF-8")
    else:
        for i, val in enumerate(x):
            x_encoded[i] = val.encode("UTF-8")

    maxed = 1
    total = 0
    for b in x_encoded:
        bn = len(b)
        total += bn
        if bn > maxed:
            maxed = bn

    with h5py.File(os.path.join(path, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")

        # Deciding whether to use the custom VLS layout. Note that we use 2
        # uint64's to store the pointer for each string, hence the 16.
        nstr = len(x_encoded)
        if string_list_vls is None:
            string_list_vls = (maxed * nstr > total + nstr * 16)

        if string_list_vls:
            ghandle.attrs["type"] = "vls"
            dtype = numpy.dtype([('offset', 'u8'), ('length', 'u8')])

            pointers = [None] * nstr
            cumulative = 0
            for i, b in enumerate(x_encoded):
                bn = len(b)
                pointers[i] = (cumulative, bn)
                cumulative += bn

            pset = ghandle.create_dataset("pointers", data=pointers, dtype=dtype, compression="gzip", chunks=True)
            if has_missing:
                pset.attrs["missing-value-placeholder"] = placeholder
            
            heap = numpy.ndarray(cumulative, dtype=numpy.dtype("u1"))
            cumulative = 0
            for i, b in enumerate(x_encoded):
                start = cumulative
                cumulative += len(b)
                heap[start:cumulative] = list(b)
            ghandle.create_dataset("heap", data=heap, dtype='u1', compression="gzip", chunks=True)

        else:
            # No VLS is a lot simpler as it's handled by h5py.
            ghandle.attrs["type"] = "string"
            dset = ghandle.create_dataset("values", data=x_encoded, dtype="S" + str(maxed), compression="gzip", chunks=True)
            if has_missing:
                dset.attrs["missing-value-placeholder"] = placeholder

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
