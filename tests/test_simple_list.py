import dolomite_base as dl
import numpy as np
from tempfile import mkdtemp
from biocframe import BiocFrame
from biocutils import Factor, StringList, NamedList, FloatList, IntegerList, BooleanList
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
            "float": FloatList([ 1.5, 2.5, 3.5, 4.5, 5.5 ]),
            "int": IntegerList([ 1, 2, 3, 4, 5 ]),
            "bool": BooleanList([ True, False, False, True, False ])
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
    assert everything["i_am_a_dict"]["float"] == roundtrip["i_am_a_dict"]["float"]
    assert everything["i_am_a_dict"]["int"] == roundtrip["i_am_a_dict"]["int"]
    assert everything["i_am_a_dict"]["bool"] == roundtrip["i_am_a_dict"]["bool"]

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
    assert everything["i_am_a_dict"]["float"] == roundtrip["i_am_a_dict"]["float"]
    assert everything["i_am_a_dict"]["int"] == roundtrip["i_am_a_dict"]["int"]
    assert everything["i_am_a_dict"]["bool"] == roundtrip["i_am_a_dict"]["bool"]


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


def test_simple_list_numpy_scalars():
    everything = {
        "float": np.float64(9.9),
        "int": np.int8(10),
        "bool": np.bool_(False),
        "float2": np.array(1.45),
        "int2": np.array(-5),
        "bool2": np.array(True),
        "masked": np.ma.masked
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
    assert isinstance(roundtrip["float2"], float)
    assert roundtrip["float2"] == 1.45
    assert isinstance(roundtrip["int2"], int)
    assert roundtrip["int2"] == -5
    assert isinstance(roundtrip["bool2"], bool)
    assert roundtrip["bool2"]
    assert roundtrip["masked"] is None

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
    assert roundtrip["float2"] == 1.45
    assert isinstance(roundtrip["int2"], int)
    assert roundtrip["int2"] == -5
    assert isinstance(roundtrip["bool2"], bool)
    assert roundtrip["bool2"]
    assert roundtrip["masked"] is None


def test_simple_list_large_integers():
    everything = {
        "a": 2**31 - 1,
        "b": 2**31,
        "c": IntegerList([-2**32])
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    meta = dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert everything["a"] == roundtrip["a"]
    assert isinstance(roundtrip["a"], int)
    assert everything["b"] == roundtrip["b"]
    assert isinstance(roundtrip["b"], float)
    assert everything["c"].as_list() == roundtrip["c"].as_list()
    assert isinstance(roundtrip["c"], FloatList)

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    meta = dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert everything["a"] == roundtrip["a"]
    assert isinstance(roundtrip["a"], int)
    assert everything["b"] == roundtrip["b"]
    assert isinstance(roundtrip["b"], float)
    assert everything["c"].as_list() == roundtrip["c"].as_list()
    assert isinstance(roundtrip["c"], FloatList)


def test_simple_list_special_float():
    everything = {
        "a": np.nan,
        "b": FloatList([np.inf, -np.inf, np.nan])
    }

    # Stage as JSON.
    dir = os.path.join(mkdtemp(), "json")
    meta = dl.save_object(everything, dir, simple_list_mode="json")

    roundtrip = dl.read_object(dir)
    assert np.isnan(roundtrip["a"])
    assert roundtrip["b"][0] == np.inf
    assert roundtrip["b"][1] == -np.inf
    assert np.isnan(roundtrip["b"][2])

    # Stage as HDF5.
    dir = os.path.join(mkdtemp(), "hdf5")
    meta = dl.save_object(everything, dir, simple_list_mode="hdf5")

    roundtrip = dl.read_object(dir)
    assert np.isnan(roundtrip["a"])
    assert roundtrip["b"][0] == np.inf
    assert roundtrip["b"][1] == -np.inf
    assert np.isnan(roundtrip["b"][2])


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
        "factor": Factor.from_sequence([ "sydney", "brisbane", "sydney", "melbourne"], names=["A", "B", "C", "D"]),
        "string": StringList(["Aria", "Akari", "Akira", "Aika"], names=["1", "2", "3", "4"])
    }

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
