from numpy import ndarray
import numpy
from typing import Union, Sequence, Tuple
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
