from typing import Sequence, Union
import numpy
import h5py

from . import choose_missing_placeholder as ch
from . import _utils_misc as misc
from . import _utils_string as strings


def _is_missing_scalar(x): 
    return x is None or numpy.ma.is_masked(x)


def _has_missing(x: Sequence):
    if isinstance(x, numpy.ndarray):
        if isinstance(x, numpy.ma.MaskedArray):
            return x.mask.any()
        return False
    return any(_is_missing_scalar(y) for y in x)


def _fill_with_placeholder(x, dtype, placeholder):
    copy = numpy.ndarray(len(x), dtype=dtype)
    for i, y in enumerate(x):
        if _is_missing_scalar(y):
            copy[i] = placeholder
        else:
            copy[i] = y
    return copy


###################################################
###################################################


def write_string_vector_to_hdf5(
    handle: h5py.Group, 
    name: str, 
    x: Sequence[str], 
    placeholder_name: str = "missing-value-placeholder"
) -> h5py.Dataset:
    """
    Write a string vector to a HDF5 file as a 1-dimensional dataset with a
    fixed-length string datatype. If ``x`` contains missing values, a suitable
    placeholder value is selected using
    :py:func:`~dolomite_base.choose_missing_placeholder.choose_missing_string_placeholder`.
    and used to replace all missing values in the dataset. The placeholder
    itself is stored as an attribute of the dataset.

    Args:
        handle: A handle to a HDF5 group.

        name: Name of the dataset in which to save the string vector.

        x: Sequence containing strings, Nones, and/or masked NumPy values.
            
        placeholder_name: 
            Name of the attribute in which to store the missing value
            placeholder, if ``x`` contains None or masked values.

    Returns:
        Handle for the newly created dataset.
    """
    missed = _has_missing(x)

    if missed:
        placeholder = ch.choose_missing_string_placeholder(x)
        x = list(x)
        for i, y in enumerate(x):
            if _is_missing_scalar(y):
                x[i] = placeholder

    dset = strings.save_fixed_length_strings(handle, name, x)
    if missed:
        dset.attrs[placeholder_name] = placeholder
    return dset


###################################################
###################################################


def write_integer_vector_to_hdf5(
    handle: h5py.Group, 
    name: str, 
    x: Sequence[int], 
    h5type: str = "i4",
    placeholder_name: str = "missing-value-placeholder", 
    allow_float_promotion: bool = False 
) -> h5py.Dataset:
    """
    Write an integer vector to a HDF5 file as a 1-dimensional dataset. If
    ``x`` contains missing values, a placeholder value is selected by
    :py:func:`~dolomite_base.choose_missing_placeholder.choose_missing_integer_placeholder`
    and used to replace all of the missing values in the dataset. The
    placeholder value itself is stored as an attribute of the dataset.

    Args:
        handle: A handle to a HDF5 group.

        name: Name of the dataset in which to save the integer vector.

        x: Sequence containing integers, Nones, and/or masked NumPy values.

        h5type: Integer type of the HDF5 dataset to create.
            
        placeholder_name: 
            Name of the attribute in which to store the missing value
            placeholder, if ``x`` contains None or masked values.

        allow_float_promotion:
            Whether to save ``x`` into a 64-bit floating-point dataset if any
            values in ``x`` exceeds the range of values that can be represented
            by ``h5type``, or if no missing value placeholder can be found
            within the acceptable range of integer values. If ``False``, an
            error is raised if ``x`` cannot be saved without promotion.

    Returns:
        Handle for the newly created dataset.
    """
    missed = _has_missing(x)

    max_dtype = numpy.dtype(h5type).type
    limits = numpy.iinfo(max_dtype)
    exceeds = False
    for y in x:
        if not _is_missing_scalar(y):
            if y < limits.min or y > limits.max:
                exceeds = True
                break

    if exceeds:
        if not allow_float_promotion:
            raise ValueError("cannot save out-of-range integers without type promotion")
        if missed:
            placeholder = numpy.nan
            x = _fill_with_placeholder(x, numpy.float64, placeholder)
    else:
        if missed:
            placeholder = ch.choose_missing_integer_placeholder(x, max_dtype=max_dtype)
            if placeholder is None:
                exceeds = True
                if not allow_float_promotion:
                    raise ValueError("cannot find a suitable missing value placeholder without type promotion")
                placeholder = numpy.nan
                x = _fill_with_placeholder(x, numpy.float64, placeholder)
            else:
                x = _fill_with_placeholder(x, placeholder.dtype.type, placeholder)

    if exceeds:
        h5type = "f8"

    dset = handle.create_dataset(name, data=x, dtype=h5type, compression="gzip", chunks=True)
    if missed:
       dset.attrs.create(placeholder_name, placeholder, dtype=h5type)
    return dset


###################################################
###################################################


def write_float_vector_to_hdf5(
    handle: h5py.Group, 
    name: str, 
    x: Sequence[float], 
    h5type: str = "f8",
    placeholder_name: str = "missing-value-placeholder"
) -> h5py.Dataset:
    """
    Write a floating-point vector to a HDF5 file as a 1-dimensional dataset.
    If ``x`` contains missing values, a placeholder value is selected by
    :py:func:`~dolomite_base.choose_missing_placeholder.choose_missing_float_placeholder`.
    and used to replace all of the missing values in the dataset. The
    placeholder value itself is stored as an attribute of the dataset.

    Args:
        handle: A handle to a HDF5 group.

        name: Name of the dataset in which to save the integer vector.

        x: Sequence containing floats, Nones, and/or masked NumPy values.

        h5type: Floating-point type of the HDF5 dataset to create.
            
        placeholder_name: 
            Name of the attribute in which to store the missing value
            placeholder, if ``x`` contains None or masked values.

    Returns:
        Handle for the newly created dataset.
    """
    missed = _has_missing(x)
    if missed:
        dtype = numpy.dtype(h5type).type
        placeholder = ch.choose_missing_float_placeholder(x, dtype=dtype)
        x = _fill_with_placeholder(x, dtype, placeholder)

    dset = handle.create_dataset(name, data=x, dtype=h5type, compression="gzip", chunks=True)
    if missed:
       dset.attrs.create(placeholder_name, placeholder, dtype=h5type)
    return dset


###################################################
###################################################


def write_boolean_vector_to_hdf5(
    handle: h5py.Group, 
    name: str, 
    x: Sequence[bool],
    placeholder_name: str = "missing-value-placeholder"
) -> h5py.Dataset:
    """
    Write a boolean vector to a HDF5 file as a 1-dimensional dataset with
    a 8-bit signed integer datatype. If ``x`` contains missing values, they
    are replaced with a placeholder value of -1.

    Args:
        handle: A handle to a HDF5 group.

        name: Name of the dataset in which to save the integer vector.

        x: Sequence containing booleans, Nones, and/or masked NumPy values.
            
        placeholder_name: 
            Name of the attribute in which to store the missing value
            placeholder, if ``x`` contains None or masked values.

    Returns:
        Handle for the newly created dataset.
    """

    missed = _has_missing(x)
    if missed:
        placeholder = -1
        x = _fill_with_placeholder(x, numpy.int8, placeholder)

    h5type = "i1"
    dset = handle.create_dataset(name, data=x, dtype=h5type, compression="gzip", chunks=True)
    if missed:
       dset.attrs.create(placeholder_name, placeholder, dtype=h5type)
    return dset

