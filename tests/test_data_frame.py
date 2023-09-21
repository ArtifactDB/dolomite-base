from biocframe import BiocFrame
import dolomite as dl
import numpy as np
from tempfile import mkdtemp

def test_data_frame_simple_list():
    df = BiocFrame({
        "akari": [ 1, 2, 3, 4, 5 ],
        "aika": [ "sydney", "melbourne", "", "perth", "adelaide" ],
        "alice": [ True, False, False, True, True ],
        "ai": [ 2.3, 1.2, 5.2, 3.1, -1.2 ],
        "alicia": [ 5, 2, 1, 3.2, -2 ], # mixed integers and numbers
    })

    dir = mkdtemp()
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "akari" }
    assert meta["data_frame"]["columns"][1] == { "type": "string", "name": "aika" }
    assert meta["data_frame"]["columns"][2] == { "type": "boolean", "name": "alice" }
    assert meta["data_frame"]["columns"][3] == { "type": "number", "name": "ai" }
    assert meta["data_frame"]["columns"][4] == { "type": "number", "name": "alicia" }
#    dl.write_metadata(meta, dir)

    roundtrip = dl.load_data_frame_csv(meta, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert list(roundtrip.column("akari")) == df.column("akari")
    assert roundtrip.column("aika") == df.column("aika")
    assert list(roundtrip.column("alice")) == df.column("alice")
    assert list(roundtrip.column("ai")) == df.column("ai")
    assert list(roundtrip.column("alicia")) == df.column("alicia")
