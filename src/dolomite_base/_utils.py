from numpy import ndarray
import numpy
from typing import Union, Sequence
from . import _cpphelpers as lib


def _fragment_string_contents(strlengths: ndarray, buffer: ndarray) -> list[str]:
    sofar = 0
    collected = []
    for i, x in enumerate(strlengths):
        endpoint = sofar + x 
        collected.append(buffer[sofar:endpoint].decode("ASCII"))
        sofar = endpoint
    return collected


def _mask_strings(collected: list, mask: ndarray):
    for i, x in enumerate(mask):
        if x:
            collected[i] = None


def _choose_string_missing_placeholder(x: Sequence[str]) -> str:
    present = set(x)
    base = "NA"
    while base in present:
        base += "_"
    return base


LIMIT32 = 2**31


def _is_integer_scalar_within_limit(x) -> bool:
    return x >= -LIMIT32 and x < LIMIT32


def _is_integer_vector_within_limit(x: Sequence[int]) -> bool:
    for y in x:
        if not _is_integer_scalar_within_limit(y):
            return False
    return True


def _choose_integer_missing_placeholder(x: Sequence) -> Union[numpy.int32, None]:
    in_use = set(x)
    candidate = -2**31
    maxval = 2**31
    while candidate in in_use and candidate < maxval:
        candidate += 1
    if candidate == maxval:
        return None
    return numpy.int32(candidate)


def _fill_integer_missing_placeholder(x : numpy.ma.array, placeholder: numpy.int32) -> numpy.ndarray:
    copy = y.astype(np.int32)
    copy.fill_value = placeholder
    return copy.data


def _choose_float_missing_placeholder(x: Sequence) -> numpy.float64:
    store = numpy.ndarray(1, dtypes=numpy.float64)
    lib.extract_r_missing(store)
    return store[0]


def _fill_float_missing_placeholder(x: numpy.ma.array, placeholder: numpy.float64) -> numpy.ndarray:
    copy = y.astype(np.float64)
    copy.fill_value = placeholder
    return copy.data


def _choose_boolean_missing_placeholder() -> numpy.int8:
    return numpy.int8(-1)


def _fill_boolean_missing_placeholder(x : numpy.ma.array, placeholder: numpy.int8) -> numpy.ndarray:
    copy = y.astype(np.int8)
    copy.fill_value = _choose_boolean_missing_placeholder()
    return copy.data
