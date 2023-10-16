import dolomite_base as dl
import numpy as np
from tempfile import mkdtemp


def test_simple_list_basic():
    everything = {
        "i_am_a_string": "foo",
        "i_am_a_number": 1.23,
        "i_am_a_integer": -23,
        "i_am_a_boolean": False,
        "i_am_a_list": [ 1, True, 2.3, "bar" ],
        "i_am_a_dict": {
            "string": [ "b", "c", "d", "e" ],
            "float": np.random.rand(10),
            "int": (np.random.rand(10) * 10).astype(np.int32),
            "bool": np.random.rand(10) > 0.5
        },
        "i_am_nothing": None
    }

    dir = mkdtemp()

    # Stage as JSON.
    meta = dl.stage_object(everything, dir, "foo")
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_json_simple_list(meta, dir)
    assert everything["i_am_a_string"] == roundtrip["i_am_a_string"]
    assert everything["i_am_a_number"] == roundtrip["i_am_a_number"]
    assert everything["i_am_a_integer"] == roundtrip["i_am_a_integer"]
    assert everything["i_am_a_boolean"] == roundtrip["i_am_a_boolean"]
    assert everything["i_am_a_list"] == roundtrip["i_am_a_list"]
    assert everything["i_am_nothing"] == roundtrip["i_am_nothing"]

    assert everything["i_am_a_dict"]["string"] == roundtrip["i_am_a_dict"]["string"]
    assert np.allclose(everything["i_am_a_dict"]["float"], roundtrip["i_am_a_dict"]["float"])
    assert (everything["i_am_a_dict"]["int"] == roundtrip["i_am_a_dict"]["int"]).all()
    assert (everything["i_am_a_dict"]["bool"] == roundtrip["i_am_a_dict"]["bool"]).all()

    # Stage as HDF5.
    meta = dl.stage_object(everything, dir, "foo2", mode="hdf5")
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_hdf5_simple_list(meta, dir)
    assert everything["i_am_a_string"] == roundtrip["i_am_a_string"]
    assert everything["i_am_a_number"] == roundtrip["i_am_a_number"]
    assert everything["i_am_a_integer"] == roundtrip["i_am_a_integer"]
    assert everything["i_am_a_boolean"] == roundtrip["i_am_a_boolean"]
    assert everything["i_am_a_list"] == roundtrip["i_am_a_list"]
    assert everything["i_am_nothing"] == roundtrip["i_am_nothing"]

    assert everything["i_am_a_dict"]["string"] == roundtrip["i_am_a_dict"]["string"]
    assert np.allclose(everything["i_am_a_dict"]["float"], roundtrip["i_am_a_dict"]["float"])
    assert (everything["i_am_a_dict"]["int"] == roundtrip["i_am_a_dict"]["int"]).all()
    assert (everything["i_am_a_dict"]["bool"] == roundtrip["i_am_a_dict"]["bool"]).all()


def test_simple_list_masking():
    everything = {
        "string": [ None, "b", "c", "d" "e" ],
        "float": np.ma.array(np.random.rand(5), mask=np.array([False, True, False, False, False])),
        "int": np.ma.array((np.random.rand(5) * 10).astype(np.int32), mask=np.array([False, False, True, False, False])),
        "bool": np.ma.array(np.random.rand(5) > 0.5, mask=np.array([False, False, False, True, False]))
    }

    dir = mkdtemp()
    meta = dl.stage_object(everything, dir, "foo")
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_json_simple_list(meta, dir)
    assert everything["string"] == roundtrip["string"]
    assert np.allclose(everything["float"], roundtrip["float"])
    assert (everything["int"] == roundtrip["int"]).all()
    assert (everything["bool"] == roundtrip["bool"]).all()


def test_simple_list_numpy_scalars():
    everything = {
        "float": np.float64(9.9),
        "int": np.int8(10),
        "bool": np.bool_(False),
        "float2": np.array(-9.9, dtype=np.float64),
        "int2": np.array(-10, dtype=np.int16),
        "bool2": np.array(True, dtype=np.bool_)
    }

    dir = mkdtemp()
    meta = dl.stage_object(everything, dir, "foo")
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_json_simple_list(meta, dir)
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

    dir = mkdtemp()
    meta = dl.stage_object(everything, dir, "foo")
    dl.write_metadata(meta, dir)

    # Type is kind of lost with the scalar... oh well.
    roundtrip = dl.load_json_simple_list(meta, dir)
    assert np.ma.is_masked(roundtrip["float"])
    assert np.ma.is_masked(roundtrip["int"])
    assert np.ma.is_masked(roundtrip["bool"])
    assert np.ma.is_masked(roundtrip["masked"])
