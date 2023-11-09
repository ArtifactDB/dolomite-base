import dolomite_base as dl
from biocutils import StringList
from tempfile import mkdtemp
from dolomite_base.stage_atomic_vector import _stage_atomic_vector
import numpy


def test_string_list_basic():
    sl = StringList([1,2,3,4])

    dir = mkdtemp()
    meta = dl.stage_object(sl, dir, "foo")
    assert meta["atomic_vector"]["type"] == "string"
    assert meta["atomic_vector"]["length"] == 4
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, StringList)
    assert roundtrip == sl


def test_atomic_numpy_vector():
    na = numpy.random.rand(10)

    dir = mkdtemp()
    meta = _stage_atomic_vector(na, dir, "foo")
    assert meta["atomic_vector"]["type"] == "number"
    assert meta["atomic_vector"]["length"] == 10
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, numpy.ndarray)
    assert numpy.allclose(na, roundtrip)
