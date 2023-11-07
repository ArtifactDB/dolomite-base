from numpy import ndarray
import numpy
from typing import Union, Sequence, Tuple
from biocframe import Factor
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


def _determine_list_type(x: list) -> Tuple:
    all_types = set()
    has_none = False
    for y in x:
        if isinstance(y, numpy.generic):
            if numpy.issubdtype(y.dtype.type, numpy.integer):
                all_types.add(int)
            elif numpy.issubdtype(y.dtype.type, numpy.floating):
                all_types.add(float)
            else:
                all_types.add(type(y))
        elif _is_missing_scalar(y):
            has_none = True
        else:
            all_types.add(type(y))

    final_type = None
    if len(all_types) == 1:
        final_type = list(all_types)[0]
        if final_type == int:
            if not _is_integer_vector_within_limit(x):
                final_type = float 
        elif final_type != str and final_type != bool and final_type != float:
            final_type = None
    elif len(all_types) == 2 and int in all_types and float in all_types:
        final_type = float
    elif len(all_types) == 0: # if all None, this is the fallback.
        final_type = str

    return final_type, has_none


def _determine_numpy_type(x: Union[numpy.ndarray, numpy.generic]):
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


def _funnel_to_mask(x, dtype) -> Tuple:
    if isinstance(x, numpy.ma.MaskedArray):
        copy = x.data.astype(dtype) # must be a copy, as we'll be mutating it.
        mask = x.mask.astype(numpy.uint8) # use uint8 to avoid problems with ambiguous bool typing.
    else:
        copy = numpy.zeros(len(x), dtype=dtype)
        mask = numpy.zeros(len(x), dtype=numpy.uint8)
        for i, y in enumerate(x):
            if _is_missing_scalar(y):
                mask[i] = 1
            else:
                copy[i] = y

    return copy, mask


def _choose_missing_integer_placeholder(x: Sequence) -> Tuple:
    copy, mask = _funnel_to_mask(x, dtype=numpy.int32)
    okay, placeholder = lib.choose_missing_integer_placeholder(copy, mask)
    if okay:
        return copy, placeholder
    else:
        return None, None


def _choose_missing_float_placeholder(x: Sequence) -> Tuple:
    copy, mask = _funnel_to_mask(x, dtype=numpy.float64)
    okay, placeholder = lib.choose_missing_float_placeholder(copy, mask)
    if not okay:
        raise ValueError("failed to find an appropriate floating-point missing value placeholder")
    return copy, placeholder


def _choose_missing_boolean_placeholder(x: Sequence) -> Tuple:
    copy, mask = _funnel_to_mask(x, dtype=numpy.int8)
    placeholder = numpy.int8(-1)
    for i, m in enumerate(mask):
        if m:
            copy[i] = placeholder
    return copy, placeholder


def _choose_missing_string_placeholder(x: Sequence) -> Tuple:
    present = set(x)
    placeholder = "NA"
    while placeholder in present:
        placeholder += "_"

    copy = x[:]
    for j, y in enumerate(copy):
        if y is None:
            copy[j] = placeholder 

    return copy, placeholder


def _choose_missing_factor_placeholder(x: Factor) -> Tuple:
    copy = x.codes[:]
    for i, y in enumerate(copy):
        if y is None:
            copy[i] = -1
    return copy, -1


def _is_gzip_compressed(meta, parent_name):
    if parent_name in meta:
        x = meta[parent_name]
        if "compression" in x:
            method = x["compression"]
            if method == "gzip":
                return True
            elif method != "none":
                raise NotImplementedError(method + " decompression is not yet supported for '" + parent_name + "'")
    return False


def _save_fixed_length_strings(handle, name: str, strings: list[str]):
    tmp = [ y.encode("UTF8") for y in strings ]
    maxed = 1
    for b in tmp:
        if len(b) > maxed:
            maxed = len(b)
    return handle.create_dataset(name, data=tmp, dtype="S" + str(maxed), compression="gzip", chunks=True)
