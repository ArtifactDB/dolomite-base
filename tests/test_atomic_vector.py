import dolomite_base as dl
from biocutils import StringList, IntegerList, FloatList, BooleanList
from tempfile import mkdtemp
import h5py
import os

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


def test_boolean_list_names():
    sl = BooleanList([True, False, False, True])
    sl = sl.set_names(["A", "B", "C", "D"])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BooleanList)
    assert roundtrip == sl
