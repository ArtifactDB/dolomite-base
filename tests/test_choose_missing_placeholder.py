import dolomite_base as dl
import numpy


def test_choose_missing_integer_placeholder():
    out = dl.choose_missing_integer_placeholder([None, 1, 2, 3]) 
    assert out.dtype == numpy.int32
    assert out == -2**31

    out = dl.choose_missing_integer_placeholder([None, 1, 2, -2**31]) 
    assert out.dtype == numpy.int32
    assert out == 2**31 - 1

    out = dl.choose_missing_integer_placeholder([None, 1, 2, 2**31-1, -2**31]) 
    assert out.dtype == numpy.int32
    assert out == 0

    out = dl.choose_missing_integer_placeholder([None, 1, 2, 2**31-1, 0, -2**31]) 
    assert out.dtype == numpy.int32
    assert out == -2**31 + 1

    # Attempts to respect the provided NumPy type.
    out = dl.choose_missing_integer_placeholder(numpy.array([1, 2], dtype=numpy.uint8))
    assert out.dtype == numpy.uint8
    assert out == 0

    out = dl.choose_missing_integer_placeholder(numpy.array([1, 2, 0], dtype=numpy.uint8))
    assert out.dtype == numpy.uint8
    assert out == 255

    out = dl.choose_missing_integer_placeholder(numpy.array(range(256), dtype=numpy.uint8))
    assert out.dtype == numpy.int32
    assert out == -2**31

    out = dl.choose_missing_integer_placeholder(numpy.array(range(256), dtype=numpy.uint8), max_dtype=numpy.uint8)
    assert out is None

    # Works with a set.
    out = dl.choose_missing_integer_placeholder(set([1, 2, 2**31-1, 0, -2**31, -2**31 + 1]))
    assert out.dtype == numpy.int32
    assert out == -2**31 + 2

    # Works with a Numpy masked arrays.
    out = dl.choose_missing_integer_placeholder(numpy.ma.MaskedArray([1, 2, 2**31-1, 0, -2**31], mask=[True] + [False] * 4))
    assert out.dtype == numpy.int32
    assert out == -2**31 + 1 


def test_choose_missing_float_placeholder():
    out = dl.choose_missing_float_placeholder([None, 1, 2, 3]) 
    assert out.dtype == numpy.float64
    assert numpy.isnan(out)

    out = dl.choose_missing_float_placeholder([None, 1, 2, 3, numpy.nan]) 
    assert out.dtype == numpy.float64
    assert out == numpy.inf

    out = dl.choose_missing_float_placeholder([None, 1, 2, 3, numpy.nan, numpy.inf]) 
    assert out.dtype == numpy.float64
    assert out == -numpy.inf

    fstats = numpy.finfo(numpy.float64)
    out = dl.choose_missing_float_placeholder([None, 1, 2, 3, numpy.nan, numpy.inf, -numpy.inf]) 
    assert out.dtype == numpy.float64
    assert out == fstats.min

    out = dl.choose_missing_float_placeholder([None, 1, 2, 3, numpy.nan, numpy.inf, -numpy.inf, fstats.min]) 
    assert out.dtype == numpy.float64
    assert out == fstats.max

    out = dl.choose_missing_float_placeholder([None, 1, 2, 3, numpy.nan, numpy.inf, -numpy.inf, fstats.min, fstats.max]) 
    assert out.dtype == numpy.float64
    assert out == 0

    out = dl.choose_missing_float_placeholder([None, 1, 2, 3, numpy.nan, numpy.inf, -numpy.inf, fstats.min, fstats.max, 0]) 
    assert out.dtype == numpy.float64
    assert out == fstats.min / 2

    # Respects original type. 
    out = dl.choose_missing_float_placeholder(numpy.array([1, 2, 3], dtype=numpy.float32)) 
    assert out.dtype == numpy.float32
    assert numpy.isnan(out)

    fstats = numpy.finfo(numpy.float32)
    out = dl.choose_missing_float_placeholder(numpy.array([1, 2, 3, numpy.nan, numpy.inf, -numpy.inf, fstats.min, fstats.max, 0], dtype=numpy.float32)) 
    assert out.dtype == numpy.float32
    assert out == fstats.min / 2

    # Works with NumPy masked arrays.
    out = dl.choose_missing_float_placeholder(numpy.ma.MaskedArray(numpy.array([1, 2, 3, numpy.nan]), mask=[True] + [False] * 3))
    assert out.dtype == numpy.float64
    assert out == numpy.inf

    fstats = numpy.finfo(numpy.float64)
    out = dl.choose_missing_float_placeholder(numpy.ma.MaskedArray(numpy.array([1, 2, 3, numpy.nan, numpy.inf, -numpy.inf, fstats.min, fstats.max, 0]), mask=[True] + [False] * 8))
    assert out.dtype == numpy.float64
    assert out == fstats.min/2

    # Works with a set.
    out = dl.choose_missing_float_placeholder(set([None, 1, 2, 3]))
    assert out.dtype == numpy.float64
    assert numpy.isnan(out)

    fstats = numpy.finfo(numpy.float64)
    out = dl.choose_missing_float_placeholder(set([None, 1, 2, 3, numpy.nan, numpy.inf, -numpy.inf, fstats.min, fstats.max, 0]))
    assert out.dtype == numpy.float64
    assert fstats.min / 2


def test_choose_missing_string_placeholder():
    out = dl.choose_missing_string_placeholder([None, "XA"])
    assert out == "NA"

    out = dl.choose_missing_string_placeholder([None, "NA"])
    assert out == "NA_"

    out = dl.choose_missing_string_placeholder(set(["NA", "NA_"]))
    assert out == "NA__"

    out = dl.choose_missing_string_placeholder(numpy.array(["NA", "NA_"]))
    assert out == "NA__"
