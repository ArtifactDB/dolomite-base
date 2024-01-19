from numpy import ndarray
import numpy
from typing import Union, Sequence, Tuple
from . import lib_dolomite_base as lib
import h5py


def _is_missing_scalar(x) -> bool:
    return numpy.ma.is_masked(x) or x is None


LIMIT32 = 2**31


def _is_integer_scalar_within_limit(x) -> bool:
    if _is_missing_scalar(x):
        return True
    return x >= -LIMIT32 and x < LIMIT32


def _is_integer_vector_within_limit(x: Sequence[int]) -> bool:
    for y in x:
        if not _is_integer_scalar_within_limit(y):
            return False
    return True


def _determine_save_type(x: Union[numpy.ndarray, numpy.generic]):
    dt = x.dtype.type
    if issubclass(dt, numpy.integer):
        okay = False
        if isinstance(x, numpy.generic): 
            okay = _is_integer_scalar_within_limit(x)
        else:
            okay = _is_integer_vector_within_limit(x)
        if okay:
            return int
        else:
            return float
    elif issubclass(dt, numpy.floating):
        return float
    elif issubclass(dt, numpy.bool_):
        return bool
    else:
        raise NotImplementedError("saving a NumPy array of " + str(x.dtype) + " is not supported yet")


def _is_actually_masked(x: numpy.ndarray) -> bool:
    if not numpy.ma.is_masked(x):
        return False
    if isinstance(x.mask, bool):
        return x.mask
    if isinstance(x.mask, numpy.ndarray):
        return x.mask.any()
    return True


def list_to_numpy_with_mask(x: Sequence, x_dtype, mask_dtype = numpy.uint8) -> numpy.ndarray:
    """
    Convert a list of numbers or None into NumPy arrays.

    Args:
        x: List of numbers.
        x_dtype: Data type to use for the output array.
        mask_dtype: Data type to use for the mask array.

    Returns:
        Tuple containing the contents of ``x`` in a NumPy array, plus another
        array indicating whether each element of ``x`` was None or masked.
        (Masked or None values are set to zero in the first array.)
    """
    mask = numpy.ndarray(len(x), dtype=mask_dtype)
    arr = numpy.ndarray(len(x), dtype=x_dtype)
    for i, y in enumerate(x):
        if _is_missing_scalar(y):
            arr[i] = 0
            mask[i] = 1
        else:
            arr[i] = y
            mask[i] = 0
    return arr, mask


def choose_missing_integer_placeholder(x: numpy.ndarray, mask: numpy.ndarray, copy: bool = True) -> Tuple:
    """
    Choose a missing placeholder for integer arrays.

    Args:
        x: An integer array.
        mask: An array of the same shape as ``x``, indicating which elements are masked.
        copy: Whether to make a copy of ``x``. If ``False``, this function may mutate it in-place.

    Returns:
        A tuple containing an int32 array with the contents of ``x``, where all
        masked values are replaced by a placeholder, plus the placeholder value
        itself. Note that the output array may be of a floating-point type.
    """
    xcopy = x.astype(numpy.int32, copy = copy) # make a copy as we'll be mutating it in C++.
    mask = mask.astype(numpy.uint8, copy = False) # use uint8 to avoid problems with ambiguous bool typing.

    okay, placeholder = lib.choose_missing_integer_placeholder(xcopy, mask)
    if okay:
        return xcopy, placeholder

    # In the rare case that it's not okay, we just convert it to a float, which
    # gives us some more room to save placeholders.
    xcopy, placeholder = choose_missing_float_placeholder(x, mask, copy = copy)
    return xcopy, placeholder


def choose_missing_float_placeholder(x: numpy.ndarray, mask: numpy.ndarray, copy: bool = True) -> Tuple:
    """
    Choose a missing placeholder for float arrays.

    Args:
        x: A floating-point array.
        mask: An array of the same shape as ``x``, indicating which elements are masked.
        copy: Whether to make a copy of ``x``. If ``False``, this function may mutate it in-place.

    Returns:
        A tuple containing a float64 array with the contents of ``x`` where all
        masked values are replaced by a placeholder, plus the placeholder value.
    """
    xcopy = x.astype(numpy.float64, copy = copy) # make a copy as we'll be mutating it in C++.
    mask = mask.astype(numpy.uint8, copy = False) # use uint8 to avoid problems with ambiguous bool typing.
    okay, placeholder = lib.choose_missing_float_placeholder(xcopy, mask)
    if not okay:
        raise ValueError("failed to find an appropriate floating-point missing value placeholder")
    return xcopy, placeholder


def choose_missing_boolean_placeholder(x: numpy.ndarray, mask: numpy.ndarray, copy: bool = True):
    """
    Choose a missing placeholder for boolean arrays.

    Args:
        x: A boolean array (or any numeric array to be interpreted as boolean).
        mask: An array of the same shape as ``x``, indicating which elements are masked.
        copy: Whether to make a copy of ``x``. If ``False``, this function may mutate it in-place.

    Returns:
        A tuple containing an int8 array with the contents of ``x``, where all
        masked values are replaced by a placeholder, plus the placeholder value.
    """
    xcopy = x.astype(numpy.int8, copy = copy) 
    placeholder = numpy.int8(-1)
    if mask.dtype == numpy.bool_:
        xcopy[mask] = placeholder
    else:
        xcopy[mask != 0] = placeholder
    return xcopy, placeholder


def choose_missing_string_placeholder(x: Sequence, copy: bool = True) -> Tuple:
    """
    Choose a missing placeholder for string sequences.

    Args:
        x: A sequence of strings or Nones.
        copy: Whether to make a copy of ``x``. If ``False``, this function may mutate it in-place.

    Returns:
        A tuple containing a list of strings with the contents of ``x`` where
        all masked values are replaced by a placeholder, plus the placeholder.
    """
    present = set(x)
    placeholder = "NA"
    while placeholder in present:
        placeholder += "_"

    if copy:
        xcopy = x[:]
    else:
        xcopy = x

    for j, y in enumerate(xcopy):
        if y is None:
            xcopy[j] = placeholder 

    return xcopy, placeholder


def save_fixed_length_strings(handle: h5py.Group, name: str, strings: list[str]):
    """
    Save a list of strings into a fixed-length string dataset.

    Args:
        handle: Handle to a HDF5 Group.
        name: Name of the dataset to create in ``handle``.
        strings: List of strings to save.

    Returns:
        ``strings`` is saved into the group as a fixed-length string dataset.
    """
    tmp = [ y.encode("UTF8") for y in strings ]
    maxed = 1
    for b in tmp:
        if len(b) > maxed:
            maxed = len(b)
    return handle.create_dataset(name, data=tmp, dtype="S" + str(maxed), compression="gzip", chunks=True)
