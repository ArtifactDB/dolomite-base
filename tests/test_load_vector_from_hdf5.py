import dolomite_base as dl
from tempfile import mkdtemp
import h5py
import numpy
import os
import biocutils


def test_load_vector_from_hdf5_strings():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_string_vector_to_hdf5(ghandle, "FOO1", ["A", "B", "C", "D"])
        dl.write_string_vector_to_hdf5(ghandle, "FOO2", ["A", "B", None, "D"])

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]

        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], str, report_1darray=False)
        assert isinstance(foo1, biocutils.StringList)
        assert foo1.as_list() == ["A", "B", "C", "D"]
        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], str, report_1darray=True)
        assert isinstance(foo1, numpy.ndarray)
        assert list(foo1) == ["A", "B", "C", "D"]

        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], str, report_1darray=False)
        assert isinstance(foo2, biocutils.StringList)
        assert foo2.as_list() == ["A", "B", None, "D"]
        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], str, report_1darray=True)
        assert isinstance(foo2, numpy.ndarray)
        assert list(foo2.data) == ["A", "B", "NA", "D"]
        assert list(foo2.mask) == [False, False, True, False]


def test_load_vector_from_hdf5_integers():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_integer_vector_to_hdf5(ghandle, "FOO1", [1, 2, 3, 4])
        dl.write_integer_vector_to_hdf5(ghandle, "FOO2", [1, 2, None, 4])
        dl.write_integer_vector_to_hdf5(ghandle, "FOO3", [1, 2, 3, 2**32], h5type="u8")

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]

        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], int, report_1darray=False)
        assert isinstance(foo1, biocutils.IntegerList)
        assert foo1.as_list() == [1, 2, 3, 4]
        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], int, report_1darray=True)
        assert isinstance(foo1, numpy.ndarray)
        assert foo1.dtype == numpy.int32
        assert list(foo1) == [1, 2, 3, 4]

        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], int, report_1darray=False)
        assert isinstance(foo2, biocutils.IntegerList)
        assert foo2.as_list() == [1, 2, None, 4]
        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], int, report_1darray=True)
        assert isinstance(foo2, numpy.ndarray)
        assert foo2.dtype == numpy.int32
        assert list(foo2.data) == [1, 2, -2**31, 4]
        assert list(foo2.mask) == [False, False, True, False]

        foo3 = dl.load_vector_from_hdf5(ghandle["FOO3"], int, report_1darray=False)
        assert isinstance(foo3, biocutils.IntegerList)
        assert foo3.as_list() == [1, 2, 3, 2**32]
        foo3 = dl.load_vector_from_hdf5(ghandle["FOO3"], int, report_1darray=True)
        assert isinstance(foo3, numpy.ndarray)
        assert foo3.dtype == numpy.uint64
        assert list(foo3) == [1, 2, 3, 2**32]


def test_load_vector_from_hdf5_floats():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_float_vector_to_hdf5(ghandle, "FOO1", [1.1, 2.2, 3.3, 4.4])
        dl.write_float_vector_to_hdf5(ghandle, "FOO2", [1.1, 2.2, 4.4, None])
        dl.write_float_vector_to_hdf5(ghandle, "FOO3", [1.1, 2.2, 4.4, None, numpy.nan]) # check for correct behavior with non-NaN placeholders
        dl.write_integer_vector_to_hdf5(ghandle, "FOO4", [1, 2, 3, 2**32], h5type="u8") # check for correct promotion of integer storage types.

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]

        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], float, report_1darray=False)
        assert isinstance(foo1, biocutils.FloatList)
        assert foo1.as_list() == [1.1, 2.2, 3.3, 4.4]
        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], float, report_1darray=True)
        assert isinstance(foo1, numpy.ndarray)
        assert list(foo1) == [1.1, 2.2, 3.3, 4.4]

        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], float, report_1darray=False)
        assert isinstance(foo2, biocutils.FloatList)
        assert foo2.as_list() == [1.1, 2.2, 4.4, None]
        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], float, report_1darray=True)
        assert isinstance(foo2, numpy.ndarray)
        assert list(foo2.data[:-1]) == [1.1, 2.2, 4.4]
        assert numpy.isnan(foo2.data[-1])
        assert list(foo2.mask) == [False, False, False, True]

        foo3 = dl.load_vector_from_hdf5(ghandle["FOO3"], float, report_1darray=False)
        assert isinstance(foo3, biocutils.FloatList)
        assert foo3.as_list()[:-1] == [1.1, 2.2, 4.4, None]
        assert numpy.isnan(foo3.as_list()[-1])
        foo3 = dl.load_vector_from_hdf5(ghandle["FOO3"], float, report_1darray=True)
        assert isinstance(foo3, numpy.ndarray)
        assert list(foo3.data[:-1]) == [1.1, 2.2, 4.4, numpy.inf]
        assert numpy.isnan(foo3.data[-1])
        assert list(foo3.mask) == [False, False, False, True, False]

        foo4 = dl.load_vector_from_hdf5(ghandle["FOO4"], float, report_1darray=False)
        assert isinstance(foo4, biocutils.FloatList)
        assert foo4.as_list() == [1, 2, 3, 2**32]
        foo4 = dl.load_vector_from_hdf5(ghandle["FOO4"], float, report_1darray=True)
        assert isinstance(foo4, numpy.ndarray)
        assert list(foo4) == [1, 2, 3, 2**32]


def test_load_vector_from_hdf5_booleans():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_boolean_vector_to_hdf5(ghandle, "FOO1", [True, False, False, True])
        dl.write_boolean_vector_to_hdf5(ghandle, "FOO2", [True, True, None, False])

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]

        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], bool, report_1darray=False)
        assert isinstance(foo1, biocutils.BooleanList)
        assert foo1.as_list() == [True, False, False, True]
        foo1 = dl.load_vector_from_hdf5(ghandle["FOO1"], bool, report_1darray=True)
        assert isinstance(foo1, numpy.ndarray)
        assert foo1.dtype == numpy.bool_
        assert list(foo1) == [True, False, False, True]

        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], bool, report_1darray=False)
        assert isinstance(foo2, biocutils.BooleanList)
        assert foo2.as_list() == [True, True, None, False]
        foo2 = dl.load_vector_from_hdf5(ghandle["FOO2"], bool, report_1darray=True)
        assert isinstance(foo2, numpy.ndarray)
        assert list(foo2.data) == [True, True, True, False]
        assert list(foo2.mask) == [False, False, True, False]
