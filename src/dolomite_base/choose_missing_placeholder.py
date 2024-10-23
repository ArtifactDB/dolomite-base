from typing import Tuple, Sequence, Optional
import numpy


def _scan_for_integer_placeholder(x: Sequence[int], dtype: type) -> Optional[numpy.generic]:
    stats = numpy.iinfo(dtype)
    if not stats.min in x:
        return dtype(stats.min)
    if not stats.max in x:
        return dtype(stats.max)
    if stats.min != 0 and not 0 in x:
        return dtype(0)

    if not isinstance(x, set):
        alt = set()
        for y in x:
            if y is not None and not numpy.ma.is_masked(y):
                alt.add(y)
        x = alt

    for i in range(stats.min + 1, stats.max):
        if not i in x:
            return dtype(i)
    return None


def choose_missing_integer_placeholder(x: Sequence[int], max_dtype: type = numpy.int32) -> Optional[numpy.generic]:
    """Choose a missing placeholder for integer sequences.

    Args:
        x:
            Sequence of integers, possibly containing masked or None values.

        max_dtype: 
            Integer NumPy type that is guaranteed to faithfully represent all
            (non-None, non-masked) values of ``x``.

    Returns:
        Value of the placeholder. This is guaranteed to be of a type that can
        fit into ``max_dtype``. It also may not be of the same type as
        ``x.dtype`` if ``x`` is a NumPy array, so some casting may be required
        when replacing missing values with the placeholder.

        If no suitable placeholder can be found, None is returned instead.
    """
    if isinstance(x, numpy.ndarray) and (x.itemsize < max_dtype().itemsize):
        candidate = _scan_for_integer_placeholder(x, x.dtype.type)
        if not candidate is None:
            return candidate
    return _scan_for_integer_placeholder(x, max_dtype)


def choose_missing_float_placeholder(x: Sequence[float], dtype: type = numpy.float64) -> Optional[numpy.generic]:
    """Choose a missing placeholder for float sequences.

    Args:
        x: 
            Sequence of floats, possibly containing masked or None values.

        dtype:
            Floating-point NumPy type to use for the placeholder. Ignored if
            ``x`` is already a NumPy floating-point array, in which case the
            ``dtype`` is just set to the ``x.dtype``.

    Returns:
        Value of the placeholder. If ``x`` is a NumPy floating-point array,
        this is guaranteed to be of the same type as ``x.dtype``.

        If no suitable placeholder can be found, None is returned instead.
    """
    if isinstance(x, numpy.ndarray) and numpy.issubdtype(x.dtype, numpy.floating):
        dtype = x.dtype.type

    can_nan = True
    for y in x:
        if y is not None and not numpy.ma.is_masked(y) and numpy.isnan(y):
            can_nan = False
            break
    if can_nan:
        return dtype(numpy.nan)

    if not numpy.inf in x:
        return dtype(numpy.inf)
    if not -numpy.inf in x:
        return dtype(-numpy.inf)

    stats = numpy.finfo(dtype)
    if not stats.min in x:
        return dtype(stats.min)
    if not stats.max in x:
        return dtype(stats.max)
    if not 0 in x:
        return dtype(0)

    if not isinstance(x, set):
        alt = set()
        for y in x:
            if y is not None and not numpy.ma.is_masked(y):
                alt.add(y)
        x = alt

    accumulated = []
    for y in x:
        if y is not None and numpy.isfinite(y):
            accumulated.append(y)
    accumulated.sort()
    for i in range(1, len(accumulated)):
        previous = accumulated[i-1]
        current = accumulated[i]
        mid = previous + (current - previous) / dtype(2)
        if mid != previous and mid != current:
            return mid

    # Highly unlikely that we'll get to this point.
    return None


def choose_missing_string_placeholder(x: Sequence[str]) -> str:
    """Choose a missing placeholder for string sequences.

    Args:
        x: 
            Sequence of strings, possibly containing missing or None values.

    Returns:
        String to use as the placeholder. This may be longer than the maximum
        string length in ``x`` (for fixed-length-string arrays), so some
        casting may be required.
    """
    placeholder = "NA"
    while placeholder in x:
        placeholder += "_"
    return placeholder
