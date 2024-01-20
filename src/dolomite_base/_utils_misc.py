from numpy import ndarray
import numpy
from typing import Union, Sequence, Tuple
from biocutils import StringList, IntegerList, FloatList, BooleanList, NamedList
import h5py


LIMIT32 = 2**31


def scalar_exceeds_int32(x: int) -> bool:
    return x < -LIMIT32 or x >= LIMIT32


def sequence_exceeds_int32(x: int, check_none: bool = True) -> bool:
    if check_none:
        for y in x:
            if y is not None and scalar_exceeds_int32(y):
                return True
    else:
        for y in x:
            if scalar_exceeds_int32(y):
                return True
    return False


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
