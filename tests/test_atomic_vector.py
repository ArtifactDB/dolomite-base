import dolomite_base as dl
from biocutils import StringList, IntegerList, FloatList, BooleanList
from tempfile import mkdtemp
import h5py
import numpy
import os
import pytest

def test_string_list_basic():
    sl = StringList([1,2,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, StringList)
    assert roundtrip == sl

    sl = StringList([1,2,None,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, StringList)
    assert roundtrip == sl


def test_string_list_names():
    sl = StringList([1,2,3,4])
    sl = sl.set_names(["A", "B", "C", "D"])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, StringList)
    assert roundtrip == sl


def test_integer_list_basic():
    sl = IntegerList([1,2,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, IntegerList)
    assert roundtrip == sl

    sl = IntegerList([1,2,None,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, IntegerList)
    assert roundtrip == sl


def test_integer_list_ndarray():
    sl = IntegerList([1,2,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip2 = dl.read_object(dir, atomic_vector_use_numeric_1darray=True)
    assert isinstance(roundtrip2, numpy.ndarray)
    assert roundtrip2.dtype == numpy.int32
    assert list(roundtrip2) == sl.as_list()

    sl = IntegerList([1,2,None,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip2 = dl.read_object(dir, atomic_vector_use_numeric_1darray=True)
    assert isinstance(roundtrip2, numpy.ma.MaskedArray)
    assert roundtrip2.dtype == numpy.int32
    assert [None if numpy.ma.is_masked(y) else int(y) for y in roundtrip2] == sl.as_list()


def test_integer_list_large():
    sl = IntegerList([2**32,4**20])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, FloatList)
    assert roundtrip == sl


def test_integer_list_names():
    sl = IntegerList([1,2,3,4])
    sl = sl.set_names(["A", "B", "C", "D"])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, IntegerList)
    assert roundtrip == sl

    with pytest.warns(UserWarning, match="skipping names"):
        roundtrip = dl.read_object(dir, atomic_vector_use_numeric_1darray=True)


def test_float_list_basic():
    sl = FloatList([1,2,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, FloatList)
    assert roundtrip == sl

    sl = FloatList([1,2,None,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, FloatList)
    assert roundtrip == sl


def test_float_list_ndarray():
    sl = FloatList([1.2,2.3,3.4,4.5])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip2 = dl.read_object(dir, atomic_vector_use_numeric_1darray=True)
    assert isinstance(roundtrip2, numpy.ndarray)
    assert roundtrip2.dtype == numpy.float64
    assert list(roundtrip2) == sl.as_list()

    sl = FloatList([1.2,None,2.3,None,3.4,4.5])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip2 = dl.read_object(dir, atomic_vector_use_numeric_1darray=True)
    assert isinstance(roundtrip2, numpy.ma.MaskedArray)
    assert roundtrip2.dtype == numpy.float64
    assert [None if numpy.ma.is_masked(y) else float(y) for y in roundtrip2] == sl.as_list()


def test_float_list_names():
    sl = FloatList([1,2,3,4])
    sl = sl.set_names(["A", "B", "C", "D"])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, FloatList)
    assert roundtrip == sl


def test_boolean_list_basic():
    sl = BooleanList([True, False, False, True])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BooleanList)
    assert roundtrip == sl

    sl = BooleanList([True, False, None, False, True])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BooleanList)
    assert roundtrip == sl


def test_boolean_list_ndarray():
    sl = BooleanList([True, False, False, True])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip2 = dl.read_object(dir, atomic_vector_use_numeric_1darray=True)
    assert isinstance(roundtrip2, numpy.ndarray)
    assert roundtrip2.dtype == numpy.bool_
    assert list(roundtrip2) == sl.as_list()

    sl = BooleanList([True, False, None, False, True])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip2 = dl.read_object(dir, atomic_vector_use_numeric_1darray=True)
    assert isinstance(roundtrip2, numpy.ma.MaskedArray)
    assert roundtrip2.dtype == numpy.bool_
    assert [None if numpy.ma.is_masked(y) else bool(y) for y in roundtrip2] == sl.as_list()


def test_boolean_list_names():
    sl = BooleanList([True, False, False, True])
    sl = sl.set_names(["A", "B", "C", "D"])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BooleanList)
    assert roundtrip == sl
