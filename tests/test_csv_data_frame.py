from biocframe import BiocFrame
import dolomite_base as dl
import numpy as np
from tempfile import mkdtemp

def test_csv_data_frame_list():
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
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)

    assert list(roundtrip.column("akari")) == df.column("akari")
    assert roundtrip.column("akari").dtype.type == np.int32

    assert roundtrip.column("aika") == df.column("aika")

    assert list(roundtrip.column("alice")) == df.column("alice")
    assert roundtrip.column("alice").dtype.type == np.bool_

    assert list(roundtrip.column("ai")) == df.column("ai")
    assert roundtrip.column("ai").dtype.type == np.float64

    assert list(roundtrip.column("alicia")) == df.column("alicia")
    assert roundtrip.column("alicia").dtype.type == np.float64


def test_csv_data_frame_row_names():
    df = BiocFrame({
        "akari": [ 1, 2, 3, 4, 5 ],
        "aika": [ "sydney", "melbourne", "", "perth", "adelaide" ],
        "alice": [ True, False, False, True, True ],
        "ai": [ 2.3, 1.2, 5.2, 3.1, -1.2 ],
    }, row_names = [ "kaori", "chihaya", "fuyuki", "azusa", "iori" ])

    dir = mkdtemp()
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["row_names"]
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.row_names == roundtrip.row_names


def test_csv_data_frame_wild_strings():
    df = BiocFrame({
        "lyrics": [ "Ochite\niku\tsunadokei\nbakari\tmiteru\tyo", "foo\"asdasd", "multie\"foo\"asdas" ],
    }, row_names = [ "nagisa\"", "\"fuko\"", "okazaki\nasdasd" ])

    dir = mkdtemp()
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["row_names"]
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.row_names == roundtrip.row_names
    assert df.column("lyrics") == roundtrip.column("lyrics")


def test_csv_data_frame_none():
    df = BiocFrame({
        "akari": [ 1, 2, None, 4, 5 ],
        "aika": [ "sydney", None, "", "perth", "adelaide" ],
        "alice": [ True, None, False, None, True ],
        "ai": [ 2.3, None, 5.2, None, -1.2 ],
    })

    dir = mkdtemp()
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "akari" }
    assert meta["data_frame"]["columns"][1] == { "type": "string", "name": "aika" }
    assert meta["data_frame"]["columns"][2] == { "type": "boolean", "name": "alice" }
    assert meta["data_frame"]["columns"][3] == { "type": "number", "name": "ai" }
    dl.write_metadata(meta, dir)

    def compare_masked_to_list(l, m):
        assert len(l) == len(m)
        for i, x in enumerate(l):
            if x is not None:
                assert x == m[i]
            else:
                assert np.ma.is_masked(m[i])

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)

    compare_masked_to_list(df.column("akari"), roundtrip.column("akari"))
    assert roundtrip.column("akari").dtype.type == np.int32

    assert roundtrip.column("aika") == df.column("aika")

    compare_masked_to_list(df.column("alice"), roundtrip.column("alice"))
    assert roundtrip.column("alice").dtype.type == np.bool_

    compare_masked_to_list(df.column("ai"), roundtrip.column("ai"))
    assert roundtrip.column("ai").dtype.type == np.float64


def test_csv_data_frame_numpy():
    df = BiocFrame({
        "alicia": np.array([ 1, 2, 3, 4, 5 ]),
        "akira": np.array([ True, True, False, False, True ]),
        "athena": np.array([ 2.3, 2.3, 5.2, 32, -1.2 ]),
    })

    dir = mkdtemp()
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "alicia" }
    assert meta["data_frame"]["columns"][1] == { "type": "boolean", "name": "akira" }
    assert meta["data_frame"]["columns"][2] == { "type": "number", "name": "athena" }
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert (roundtrip.column("alicia") == df.column("alicia")).all()
    assert roundtrip.column("alicia").dtype == np.int32
    assert (roundtrip.column("akira") == df.column("akira")).all()
    assert (roundtrip.column("athena") == df.column("athena")).all()


def test_csv_data_frame_masked():
    df = BiocFrame({
        "alicia": np.ma.array(np.array([ 1, 2, 3, 4, 5 ]), mask=[0, 1, 0, 1, 0]),
        "akira": np.ma.array(np.array([ True, True, False, False, True ]), mask=[1, 1, 0, 0, 0]),
        "athena": np.ma.array(np.array([ 2.3, 2.3, 5.2, 32, -1.2 ]), mask=[0, 0, 0, 1, 1]),
    })

    dir = mkdtemp()
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "alicia" }
    assert meta["data_frame"]["columns"][1] == { "type": "boolean", "name": "akira" }
    assert meta["data_frame"]["columns"][2] == { "type": "number", "name": "athena" }
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert (roundtrip.column("alicia") == df.column("alicia")).all()
    assert (roundtrip.column("akira") == df.column("akira")).all()
    assert (roundtrip.column("athena") == df.column("athena")).all()


def test_csv_data_frame_empty():
    dir = mkdtemp()

    # Empty except for row names.
    df = BiocFrame({}, row_names=["chihaya", "mami", "ami", "miki", "haruka"])
    meta = dl.stage_object(df, dir, "foo")
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.row_names == roundtrip.row_names

    # Fully empty
    df = BiocFrame(number_of_rows=10)
    meta = dl.stage_object(df, dir, "empty")
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "empty/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.shape == roundtrip.shape
    assert roundtrip.row_names is None
