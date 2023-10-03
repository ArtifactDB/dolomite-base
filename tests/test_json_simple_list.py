import dolomite_base as dl
import numpy as np
from tempfile import mkdtemp


def test_json_simple_list_basic():
    everything = {
        "i_am_a_string": "foo",
        "i_am_a_number": 1.23,
        "i_am_a_integer": -23,
        "i_am_a_boolean": False,
        "i_am_a_list": [ 1, True, 2.3, "bar" ],
        "i_am_a_dict": {
            "a": [ "b", "c", "d", "e" ],
            "float": np.random.rand(10),
            "int": (np.random.rand(10) * 10).astype(np.int32),
            "bool": np.random.rand(10) > 0.5
        }
    }

    dir = mkdtemp()
    meta = dl.stage_object(everything, dir, "foo")
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_json_simple_list(meta, dir)
    assert everything["i_am_a_string"] == roundtrip["i_am_a_string"]
    assert everything["i_am_a_number"] == roundtrip["i_am_a_number"]
    assert everything["i_am_a_integer"] == roundtrip["i_am_a_integer"]
    assert everything["i_am_a_boolean"] == roundtrip["i_am_a_boolean"]
    assert everything["i_am_a_list"] == roundtrip["i_am_a_list"]

    assert everything["i_am_a_dict"]["a"] == roundtrip["i_am_a_dict"]["a"]
    assert np.allclose(everything["i_am_a_dict"]["float"], roundtrip["i_am_a_dict"]["float"])
    assert (everything["i_am_a_dict"]["int"] == roundtrip["i_am_a_dict"]["int"]).all()
    assert (everything["i_am_a_dict"]["bool"] == roundtrip["i_am_a_dict"]["bool"]).all()



