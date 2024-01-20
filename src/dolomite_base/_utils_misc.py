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
