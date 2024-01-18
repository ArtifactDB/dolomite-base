from numpy import ndarray
import numpy
from typing import Union, Sequence, Tuple
from biocutils import StringList
from . import lib_dolomite_base as lib


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


def _is_actually_masked(x: numpy.ndarray):
    if not numpy.ma.is_masked(x):
        return False
    if isinstance(x.mask, bool):
        return x.mask
    if isinstance(x.mask, numpy.ndarray):
        return x.mask.any()
    return True


def _choose_missing_integer_placeholder(x: numpy.ma.MaskedArray) -> Tuple:
    copy = x.data.astype(numpy.int32) # make a copy as we'll be mutating it in C++.
    mask = x.mask.astype(numpy.uint8) # use uint8 to avoid problems with ambiguous bool typing.

    okay, placeholder = lib.choose_missing_integer_placeholder(copy, mask)
    if okay:
        return copy, placeholder, int

    # In the rare case that it's not okay, we just convert it to a float, which
    # gives us some more room to save placeholders.
    copy, placeholder = _choose_missing_float_placeholder(x)
    return copy, placeholder, float


def _choose_missing_float_placeholder(x: numpy.ma.MaskedArray) -> Tuple:
    copy = x.data.astype(numpy.float64) # make a copy as we'll be mutating it in C++.
    mask = x.mask.astype(numpy.uint8) # use uint8 to avoid problems with ambiguous bool typing.

    okay, placeholder = lib.choose_missing_float_placeholder(copy, mask)
    if not okay:
        raise ValueError("failed to find an appropriate floating-point missing value placeholder")
    return copy, placeholder


def _choose_missing_boolean_placeholder(x: numpy.ma.MaskedArray) -> Tuple:
    copy = x.data.astype(numpy.int8) 
    placeholder = numpy.int8(-1)
    copy[x.mask] = placeholder
    return copy, placeholder


def _choose_missing_string_placeholder(x: StringList) -> Tuple:
    present = set(x)
    placeholder = "NA"
    while placeholder in present:
        placeholder += "_"

    copy = x[:]
    for j, y in enumerate(copy):
        if y is None:
            copy[j] = placeholder 

    return copy, placeholder


def _save_fixed_length_strings(handle, name: str, strings: list[str]):
    tmp = [ y.encode("UTF8") for y in strings ]
    maxed = 1
    for b in tmp:
        if len(b) > maxed:
            maxed = len(b)
    return handle.create_dataset(name, data=tmp, dtype="S" + str(maxed), compression="gzip", chunks=True)
