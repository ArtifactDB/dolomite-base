import dolomite_base as dl
from tempfile import mkdtemp
import h5py
import numpy
import os
import pytest


def test_write_string_vector_to_hdf5():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_string_vector_to_hdf5(ghandle, "FOO1", ["A", "B", "C", "D"])
        dl.write_string_vector_to_hdf5(ghandle, "FOO2", ["A", "B", None, "D"])
        dl.write_string_vector_to_hdf5(ghandle, "FOO3", numpy.array(["A", "B", "C", "D"]))
        dl.write_string_vector_to_hdf5(ghandle, "FOO4", numpy.ma.MaskedArray(numpy.array(["A", "B", "NA", "D"]), mask=[True, False, False, False]))

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]
        assert [x.decode() for x in ghandle["FOO1"]] == ["A", "B", "C", "D"]
        assert [x.decode() for x in ghandle["FOO2"]] == ["A", "B", "NA", "D"]
        assert ghandle["FOO2"].attrs["missing-value-placeholder"] == "NA"
        assert [x.decode() for x in ghandle["FOO3"]] == ["A", "B", "C", "D"]
        assert [x.decode() for x in ghandle["FOO4"]] == ["NA_", "B", "NA", "D"]
        assert ghandle["FOO4"].attrs["missing-value-placeholder"] == "NA_"


def test_write_integer_vector_to_hdf5_simple():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_integer_vector_to_hdf5(ghandle, "FOO1", [1, 2, 3, 4])
        dl.write_integer_vector_to_hdf5(ghandle, "FOO2", [1, 2, None, 3])
        dl.write_integer_vector_to_hdf5(ghandle, "FOO3", numpy.array([1, 2, 3, 4], dtype=numpy.uint8))
        dl.write_integer_vector_to_hdf5(ghandle, "FOO4", numpy.ma.MaskedArray(numpy.array([1, 2, 3, 4], dtype=numpy.uint8), mask=[True, False, False, False]))

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]
        assert ghandle["FOO1"].dtype == "i4"
        assert list(ghandle["FOO1"]) == [1, 2, 3, 4]
        assert list(ghandle["FOO2"]) == [1, 2, -2**31, 3]
        assert ghandle["FOO2"].attrs["missing-value-placeholder"] == -2**31
        assert ghandle["FOO3"].dtype == "i4"
        assert list(ghandle["FOO3"]) == [1, 2, 3, 4]
        assert list(ghandle["FOO4"]) == [0, 2, 3, 4]
        assert ghandle["FOO4"].attrs["missing-value-placeholder"] == 0 


def test_write_integer_vector_to_hdf5_exceeds_limit():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")

        with pytest.raises(Exception, match="out-of-range"):
            dl.write_integer_vector_to_hdf5(ghandle, "FOO1", [1, 2, -2**32, 4])
        dl.write_integer_vector_to_hdf5(ghandle, "FOO1", [1, 2, -2**32, 4], allow_float_promotion=True)

        y = numpy.ma.MaskedArray(numpy.array(range(257), dtype=numpy.uint32), mask=[False]*256 + [True])
        with pytest.raises(Exception, match="suitable missing value placeholder"):
            dl.write_integer_vector_to_hdf5(ghandle, "FOO2", y, h5type="u1")
        dl.write_integer_vector_to_hdf5(ghandle, "FOO2", y, h5type="u1", allow_float_promotion=True)

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]
        assert ghandle["FOO1"].dtype == "f8"
        assert list(ghandle["FOO1"]) == [1, 2, -2**32, 4]

        assert ghandle["FOO2"].dtype == "f8"
        assert numpy.isnan(ghandle["FOO2"][-1])
        assert list(ghandle["FOO2"][:-1]) == list(range(256))
        assert numpy.isnan(ghandle["FOO2"].attrs["missing-value-placeholder"])


def test_write_float_vector_to_hdf5_simple():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_float_vector_to_hdf5(ghandle, "FOO1", [1.5, 2.2, 3.5, 4.7])
        dl.write_float_vector_to_hdf5(ghandle, "FOO2", [1.5, 2.2, 4.7, None])
        dl.write_float_vector_to_hdf5(ghandle, "FOO3", numpy.array([1, 3, 2, 4], dtype=numpy.float32))
        dl.write_float_vector_to_hdf5(ghandle, "FOO4", numpy.ma.MaskedArray(numpy.array([1, 2, 3, 4], dtype=numpy.float32), mask=[True, False, False, False]))

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]
        assert ghandle["FOO1"].dtype == "f8"
        assert list(ghandle["FOO1"]) == [1.5, 2.2, 3.5, 4.7]

        assert list(ghandle["FOO2"][:-1]) == [1.5, 2.2, 4.7]
        assert numpy.isnan(ghandle["FOO2"][-1])
        assert numpy.isnan(ghandle["FOO2"].attrs["missing-value-placeholder"])

        assert ghandle["FOO3"].dtype == "f8"
        assert list(ghandle["FOO3"]) == [1, 3, 2, 4]

        assert list(ghandle["FOO4"][1:]) == [2, 3, 4]
        assert numpy.isnan(ghandle["FOO4"][0])
        assert numpy.isnan(ghandle["FOO4"].attrs["missing-value-placeholder"])


def test_write_boolean_vector_to_hdf5_simple():
    path = os.path.join(mkdtemp(), "foo.h5")
    with h5py.File(path, "w") as handle:
        ghandle = handle.create_group("yourmom")
        dl.write_boolean_vector_to_hdf5(ghandle, "FOO1", [True, False, False, True])
        dl.write_boolean_vector_to_hdf5(ghandle, "FOO2", [False, False, True, None])
        dl.write_boolean_vector_to_hdf5(ghandle, "FOO3", numpy.array([True, False, True, False], dtype=numpy.bool_))
        dl.write_boolean_vector_to_hdf5(ghandle, "FOO4", numpy.ma.MaskedArray(numpy.array([False, True, True, False], dtype=numpy.bool_), mask=[True, False, False, False]))

    with h5py.File(path, "r") as handle:
        ghandle = handle["yourmom"]
        assert ghandle["FOO1"].dtype == "i1"
        assert list(ghandle["FOO1"]) == [1, 0, 0, 1]
        assert list(ghandle["FOO2"]) == [0, 0, 1, -1]
        assert ghandle["FOO2"].attrs["missing-value-placeholder"] == -1
        assert list(ghandle["FOO3"]) == [1, 0, 1, 0]
        assert ghandle["FOO4"].dtype == "i1"
        assert list(ghandle["FOO4"]) == [-1, 1, 1, 0]
        assert ghandle["FOO4"].attrs["missing-value-placeholder"] == -1
