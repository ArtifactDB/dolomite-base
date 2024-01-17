import dolomite_base as dl
from biocutils import StringList
from tempfile import mkdtemp
import numpy
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


def test_atomic_numpy_vector_simple():
    dir = os.path.join(mkdtemp(), "temp")
    os.mkdir(dir)
    with open(os.path.join(dir, "OBJECT"), 'w', encoding="utf-8") as handle:
        handle.write('{ "type": "atomic_vector", "atomic_vector": { "version": "1.0" } }')

    # Floats.
    na = (numpy.random.rand(10) * 1000).astype(numpy.int32)
    with h5py.File(os.path.join(dir, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "float"
        dhandle = ghandle.create_dataset("values", data=na, dtype="i4")

    roundtrip = dl.read_atomic_vector(dir)
    assert isinstance(roundtrip, numpy.ndarray)
    assert roundtrip.dtype == numpy.float64
    assert (na == roundtrip).all()

    # Boolean.
    na = (numpy.random.rand(10) > 0.5)
    with h5py.File(os.path.join(dir, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "boolean"
        dhandle = ghandle.create_dataset("values", data=na, dtype="i1")

    roundtrip = dl.read_atomic_vector(dir)
    assert roundtrip.dtype == numpy.bool_
    assert (na == roundtrip).all()

    # Integer.
    na = (numpy.random.rand(10) * 100).astype(numpy.uint8)
    with h5py.File(os.path.join(dir, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "integer"
        dhandle = ghandle.create_dataset("values", data=na, dtype="u2")

    roundtrip = dl.read_atomic_vector(dir)
    assert roundtrip.dtype == numpy.uint16
    assert (na == roundtrip).all()


def test_atomic_numpy_vector_masked():
    dir = os.path.join(mkdtemp(), "temp")
    os.mkdir(dir)
    with open(os.path.join(dir, "OBJECT"), 'w', encoding="utf-8") as handle:
        handle.write('{ "type": "atomic_vector", "atomic_vector": { "version": "1.0" } }')

    # With a non-NaN placeholder.
    na = numpy.random.rand(10) 
    with h5py.File(os.path.join(dir, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "float"
        dhandle = ghandle.create_dataset("values", data=na, dtype="f8")
        dhandle.attrs["missing-value-placeholder"] = na[0]

    roundtrip = dl.read_atomic_vector(dir)
    assert isinstance(roundtrip, numpy.ma.MaskedArray)
    assert roundtrip.dtype == numpy.float64
    assert (roundtrip.mask == (na == na[0])).all()
    assert (na == roundtrip).all()

    # With a NaN placeholder.
    na[9] = numpy.NaN
    with h5py.File(os.path.join(dir, "contents.h5"), "w") as handle:
        ghandle = handle.create_group("atomic_vector")
        ghandle.attrs["type"] = "float"
        dhandle = ghandle.create_dataset("values", data=na, dtype="f8")
        dhandle.attrs["missing-value-placeholder"] = numpy.NaN

    roundtrip = dl.read_atomic_vector(dir)
    assert isinstance(roundtrip, numpy.ma.MaskedArray)
    assert roundtrip.dtype == numpy.float64
    assert (roundtrip.mask == numpy.isnan(na)).all()
    assert (na == roundtrip).all()
