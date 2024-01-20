from typing import Tuple, Sequence
import numpy
from . import lib_dolomite_base as lib


def choose_missing_integer_placeholder(x: numpy.ndarray, mask: numpy.ndarray, copy: bool = True) -> Tuple:
    """Choose a missing placeholder for integer arrays.

    Args:
        x: 
            An integer array.
        
        mask: 
            An array of the same shape as ``x``, indicating 
            which elements are masked.
        
        copy: 
            Whether to make a copy of ``x``. 
            If ``False``, this function may mutate it in-place.

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
    """Choose a missing placeholder for float arrays.

    Args:
        x: 
            A floating-point array.
        
        mask: 
            An array of the same shape as ``x``, 
            indicating which elements are masked.
        
        copy: 
            Whether to make a copy of ``x``. 
            If ``False``, this function may mutate it in-place.

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
    """Choose a missing placeholder for boolean arrays.

    Args:
        x: 
            A boolean array (or any numeric array to be interpreted as boolean).
        
        mask: 
            An array of the same shape as ``x``, 
            indicating which elements are masked.
        
        copy: 
            Whether to make a copy of ``x``. 
            If ``False``, this function may mutate it in-place.

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
    """Choose a missing placeholder for string sequences.

    Args:
        x: 
            A sequence of strings or Nones.
        
        copy: 
            Whether to make a copy of ``x``. 
            If ``False``, this function may mutate it in-place.

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
