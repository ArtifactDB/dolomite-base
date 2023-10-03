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
            "int": np.random.rand(10, dtype=np.int8),
            "bool": np.random.rand(10, dtype=np.bool_)
        }
    }

    dir = mkdtemp()
    meta = dl.stage_object(everything, dir, "foo")
    dl.write_metadata(meta, dir)
