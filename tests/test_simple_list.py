import dolomite_base as dl
import numpy as np
from tempfile import mkdtemp
from biocframe import BiocFrame
from biocutils import Factor, StringList, NamedList
import os


def test_simple_list_basic():
    everything = {
        "i_am_a_string": "foo",
        "i_am_a_number": 1.23,
        "i_am_a_integer": -23,
        "i_am_a_boolean": False,
        "i_am_a_list": [ 1, True, 2.3, "bar" ],
        "i_am_a_dict": {
            "string": StringList([ "b", "c", "d", "e" ]),
            "float": np.random.rand(10),
            "int": (np.random.rand(10) * 10).astype(np.int32),
            "bool": np.random.rand(10) > 0.5
        },
        "i_am_nothing": None
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    dl.save_object(everything, dir, simple_list_mode="json")
    assert os.path.exists(os.path.join(dir, "list_contents.json.gz"))

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, NamedList)
    assert everything["i_am_a_string"] == roundtrip["i_am_a_string"]
    assert everything["i_am_a_number"] == roundtrip["i_am_a_number"]
    assert everything["i_am_a_integer"] == roundtrip["i_am_a_integer"]
    assert everything["i_am_a_boolean"] == roundtrip["i_am_a_boolean"]
    assert everything["i_am_a_list"] == roundtrip["i_am_a_list"].as_list()
    assert everything["i_am_nothing"] == roundtrip["i_am_nothing"]

    assert everything["i_am_a_dict"]["string"] == roundtrip["i_am_a_dict"]["string"]
    assert np.allclose(everything["i_am_a_dict"]["float"], roundtrip["i_am_a_dict"]["float"])
    assert (everything["i_am_a_dict"]["int"] == roundtrip["i_am_a_dict"]["int"]).all()
    assert (everything["i_am_a_dict"]["bool"] == roundtrip["i_am_a_dict"]["bool"]).all()

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(everything, dir, simple_list_mode="hdf5")
    assert os.path.exists(os.path.join(dir, "list_contents.h5"))

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, NamedList)
    assert everything["i_am_a_string"] == roundtrip["i_am_a_string"]
    assert everything["i_am_a_number"] == roundtrip["i_am_a_number"]
    assert everything["i_am_a_integer"] == roundtrip["i_am_a_integer"]
    assert everything["i_am_a_boolean"] == roundtrip["i_am_a_boolean"]
    assert everything["i_am_a_list"] == roundtrip["i_am_a_list"].as_list()
    assert everything["i_am_nothing"] == roundtrip["i_am_nothing"]

    assert everything["i_am_a_dict"]["string"] == roundtrip["i_am_a_dict"]["string"]
    assert isinstance(roundtrip["i_am_a_dict"]["string"], StringList)
    assert np.allclose(everything["i_am_a_dict"]["float"], roundtrip["i_am_a_dict"]["float"])
    assert (everything["i_am_a_dict"]["int"] == roundtrip["i_am_a_dict"]["int"]).all()
    assert (everything["i_am_a_dict"]["bool"] == roundtrip["i_am_a_dict"]["bool"]).all()


def test_simple_list_unnamed():
    everything = [ "foo", 1, 2, False, None ]

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, NamedList)
    assert roundtrip.get_names() is None
    assert roundtrip.as_list() == everything

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5 ")
    dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, NamedList)
    assert roundtrip.get_names() is None
    assert roundtrip.as_list() == everything


def test_simple_list_NamedList():
    everything = NamedList([NamedList([ "foo", 1, 2, False, None ]), "FOO"], names=["A", ""])

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, NamedList)
    assert roundtrip.get_names().as_list() == ["A", ""]
    assert roundtrip[""] == "FOO"
    assert isinstance(roundtrip["A"], NamedList)
    assert roundtrip["A"] == everything["A"]

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5 ")
    dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, NamedList)
    assert roundtrip.get_names().as_list() == ["A", ""]
    assert roundtrip[""] == "FOO"
    assert isinstance(roundtrip["A"], NamedList)
    assert roundtrip["A"] == everything["A"]


def test_simple_list_masking():
    everything = {
        "string": StringList([ None, "b", "c", "d" "e" ]),
        "float": np.ma.array(np.random.rand(5), mask=np.array([False, True, False, False, False])),
        "int": np.ma.array((np.random.rand(5) * 10).astype(np.int32), mask=np.array([False, False, True, False, False])),
        "bool": np.ma.array(np.random.rand(5) > 0.5, mask=np.array([False, False, False, True, False]))
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert everything["string"] == roundtrip["string"]
    assert np.allclose(everything["float"], roundtrip["float"])
    assert (everything["float"].mask == roundtrip["float"].mask).all()
    assert (everything["int"] == roundtrip["int"]).all()
    assert (everything["int"].mask == roundtrip["int"].mask).all()
    assert (everything["bool"] == roundtrip["bool"]).all()
    assert (everything["bool"].mask == roundtrip["bool"].mask).all()

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert everything["string"] == roundtrip["string"]
    assert (everything["float"] == roundtrip["float"]).all()
    assert (everything["float"].mask == roundtrip["float"].mask).all()
    assert (everything["int"] == roundtrip["int"]).all()
    assert (everything["int"].mask == roundtrip["int"].mask).all()
    assert (everything["bool"] == roundtrip["bool"]).all()
    assert (everything["bool"].mask == roundtrip["bool"].mask).all()


def test_simple_list_numpy_scalars():
    everything = {
        "float": np.float64(9.9),
        "int": np.int8(10),
        "bool": np.bool_(False),
        "float2": np.array(-9.9, dtype=np.float64),
        "int2": np.array(-10, dtype=np.int16),
        "bool2": np.array(True, dtype=np.bool_)
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip["float"], float)
    assert roundtrip["float"] == 9.9
    assert isinstance(roundtrip["int"], int)
    assert roundtrip["int"] == 10
    assert isinstance(roundtrip["bool"], bool)
    assert roundtrip["bool"] == False

    assert isinstance(roundtrip["float2"], float)
    assert roundtrip["float2"] == -9.9
    assert isinstance(roundtrip["int2"], int)
    assert roundtrip["int2"] == -10
    assert isinstance(roundtrip["bool"], bool)
    assert roundtrip["bool2"] == True 

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip["float"], float)
    assert roundtrip["float"] == 9.9
    assert isinstance(roundtrip["int"], int)
    assert roundtrip["int"] == 10
    assert isinstance(roundtrip["bool"], bool)
    assert roundtrip["bool"] == False

    assert isinstance(roundtrip["float2"], float)
    assert roundtrip["float2"] == -9.9
    assert isinstance(roundtrip["int2"], int)
    assert roundtrip["int2"] == -10
    assert isinstance(roundtrip["bool"], bool)
    assert roundtrip["bool2"] == True 


def test_simple_list_masked_scalars():
    everything = {
        "float": np.ma.array(np.array(-9.9, dtype=np.float64), mask=[True]),
        "int": np.ma.array(np.array(-10, dtype=np.int16), mask=[True]),
        "bool": np.ma.array(np.array(True, dtype=np.bool_), mask=[True]),
        "masked": np.ma.masked
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    meta = dl.save_object(everything, dir, simple_list_mode="json")

    # Type is kind of lost with the scalar... oh well.
    roundtrip = dl.read_object(dir)
    assert np.ma.is_masked(roundtrip["float"])
    assert np.ma.is_masked(roundtrip["int"])
    assert np.ma.is_masked(roundtrip["bool"])
    assert np.ma.is_masked(roundtrip["masked"])

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "hdf5")
    meta = dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert np.ma.is_masked(roundtrip["float"])
    assert np.ma.is_masked(roundtrip["int"])
    assert np.ma.is_masked(roundtrip["bool"])
    assert np.ma.is_masked(roundtrip["masked"])


def test_simple_list_large_integers():
    everything = {
        "a": 2**31 - 1,
        "b": 2**31,
        "c": np.array(-2**32, np.int64),
        "d": np.int64(-2**32),
        "e": np.array([2**32, -2**32], dtype=np.int64),
        "f": np.ma.array([2**32, -2**32], dtype=np.int64),
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    meta = dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert everything["a"] == roundtrip["a"]
    assert isinstance(roundtrip["a"], int)
    assert everything["b"] == roundtrip["b"]
    assert isinstance(roundtrip["b"], float)
    assert (everything["c"] == roundtrip["c"]).all()
    assert isinstance(roundtrip["c"], float)
    assert (everything["d"] == roundtrip["d"]).all()
    assert isinstance(roundtrip["d"], float)
    assert (everything["e"] == roundtrip["e"]).all()
    assert roundtrip["e"].dtype == np.float64
    assert (everything["f"] == roundtrip["f"]).all()
    assert roundtrip["f"].dtype == np.float64

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    meta = dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert everything["a"] == roundtrip["a"]
    assert isinstance(roundtrip["a"], int)
    assert everything["b"] == roundtrip["b"]
    assert isinstance(roundtrip["b"], float)
    assert (everything["c"] == roundtrip["c"]).all()
    assert isinstance(roundtrip["c"], float)
    assert (everything["d"] == roundtrip["d"]).all()
    assert isinstance(roundtrip["d"], float)
    assert (everything["e"] == roundtrip["e"]).all()
    assert roundtrip["e"].dtype == np.float64
    assert (everything["f"] == roundtrip["f"]).all()
    assert roundtrip["f"].dtype == np.float64


def test_simple_list_special_float():
    everything = {
        "a": np.NaN,
        "b": np.array(np.Inf, np.float64),
        "c": np.float64(-np.Inf),
        "d": np.array([np.Inf, np.NaN]),
        "e": np.ma.array([np.Inf, np.NaN, 2], mask=[0,0,1])
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    meta = dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert np.isnan(roundtrip["a"])
    assert roundtrip["b"] == np.Inf
    assert roundtrip["c"] == -np.Inf
    assert roundtrip["d"][0] == np.Inf
    assert np.isnan(roundtrip["d"][1])
    assert roundtrip["e"][0] == np.Inf
    assert np.isnan(roundtrip["e"][1])
    assert np.ma.is_masked(roundtrip["e"][2])

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    meta = dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert np.isnan(roundtrip["a"])
    assert roundtrip["b"] == np.Inf
    assert roundtrip["c"] == -np.Inf
    assert roundtrip["d"][0] == np.Inf
    assert np.isnan(roundtrip["d"][1])
    assert roundtrip["e"][0] == np.Inf
    assert np.isnan(roundtrip["e"][1])
    assert np.ma.is_masked(roundtrip["e"][2])


def test_simple_list_external():
    everything = {
        "a": BiocFrame({ "a_1": [ 1, 2, 3 ], "a_2": [ "A", "B", "C" ] }),
        "b": BiocFrame(number_of_rows=10),
    }


    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    dl.save_object(everything, dir, simple_list_mode='json')
    roundtrip = dl.read_object(dir)
    assert roundtrip["a"].get_column_names().as_list() == [ "a_1", "a_2" ]
    assert roundtrip["a"].shape[0] == 3
    assert roundtrip["b"].get_column_names().as_list() == []
    assert roundtrip["b"].shape[0] == 10

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(everything, dir, simple_list_mode="hdf5")
    roundtrip = dl.read_object(dir)
    assert roundtrip["a"].get_column_names().as_list() == [ "a_1", "a_2" ]
    assert roundtrip["a"].shape[0] == 3
    assert roundtrip["b"].get_column_names().as_list() == []
    assert roundtrip["b"].shape[0] == 10


def test_simple_list_factor():
    everything = {
        "regular": Factor.from_sequence([ "sydney", "brisbane", "sydney", "melbourne"]),
        "missing": Factor.from_sequence([ "sydney", None, "sydney", None]),
        "ordered": Factor.from_sequence([ "sydney", "brisbane", "sydney", "melbourne"], levels=["sydney", "melbourne", "brisbane"], ordered=True),
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    meta = dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip["regular"], Factor)
    assert list(roundtrip["regular"]) == list(everything["regular"])
    assert not roundtrip["regular"].get_ordered()
    assert list(roundtrip["missing"]) == list(everything["missing"])
    assert list(roundtrip["ordered"]) == list(everything["ordered"])
    assert roundtrip["ordered"].get_ordered()

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    meta = dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip["regular"], Factor)
    assert list(roundtrip["regular"]) == list(everything["regular"])
    assert not roundtrip["regular"].get_ordered()
    assert list(roundtrip["missing"]) == list(everything["missing"])
    assert list(roundtrip["ordered"]) == list(everything["ordered"])
    assert roundtrip["ordered"].get_ordered()


def test_simple_list_named():
    everything = {
        "factor": Factor.from_sequence([ "sydney", "brisbane", "sydney", "melbourne"]),
        "string": StringList(["Aria", "Akari", "Akira", "Aika"], names=["1", "2", "3", "4"])
    }
    everything["factor"].set_names(["A", "B", "C", "D"], in_place=True) # TODO: enable this in the constructor.

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    meta = dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert list(roundtrip["factor"]) == list(everything["factor"])
    assert roundtrip["factor"].get_names() == everything["factor"].get_names()
    assert list(roundtrip["string"]) == list(everything["string"])
    assert roundtrip["string"].get_names() == everything["string"].get_names()

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    meta = dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert list(roundtrip["factor"]) == list(everything["factor"])
    assert roundtrip["factor"].get_names() == everything["factor"].get_names()
    assert list(roundtrip["string"]) == list(everything["string"])
    assert roundtrip["string"].get_names() == everything["string"].get_names()
