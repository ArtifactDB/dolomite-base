from biocframe import BiocFrame
from biocutils import Factor, StringList
import dolomite_base as dl
import numpy as np
from tempfile import mkdtemp


def test_data_frame_list():
    df = BiocFrame({
        "akari": [ 1, 2, 3, 4, 5 ],
        "aika": [ "sydney", "melbourne", "", "perth", "adelaide" ],
        "alice": [ True, False, False, True, True ],
        "ai": [ 2.3, 1.2, 5.2, 3.1, -1.2 ],
        "alicia": [ 5, 2, 1, 3.2, -2 ], # mixed integers and numbers
        "akira": StringList(["A", "B", "C", "D", "E"]),
    })

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "akari" }
    assert meta["data_frame"]["columns"][1] == { "type": "string", "name": "aika" }
    assert meta["data_frame"]["columns"][2] == { "type": "boolean", "name": "alice" }
    assert meta["data_frame"]["columns"][3] == { "type": "number", "name": "ai" }
    assert meta["data_frame"]["columns"][4] == { "type": "number", "name": "alicia" }
    assert meta["data_frame"]["columns"][5] == { "type": "string", "name": "akira" }
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)

    assert list(roundtrip.column("akari")) == df.column("akari")
    assert roundtrip.column("akari").dtype.type == np.int32

    assert roundtrip.column("aika") == df.column("aika")
    assert isinstance(roundtrip.column("aika"), StringList)

    assert list(roundtrip.column("alice")) == df.column("alice")
    assert roundtrip.column("alice").dtype.type == np.bool_

    assert list(roundtrip.column("ai")) == df.column("ai")
    assert roundtrip.column("ai").dtype.type == np.float64

    assert list(roundtrip.column("alicia")) == df.column("alicia")
    assert roundtrip.column("alicia").dtype.type == np.float64

    assert roundtrip.column("akira") == df.column("akira")
    assert isinstance(roundtrip.column("akira"), StringList)

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    assert meta["data_frame"]["columns"] == meta2["data_frame"]["columns"]
    dl.write_metadata(meta2, dir)

    meta2 = dl.acquire_metadata(dir, "foo2/simple.h5")
    roundtrip2 = dl.load_object(meta2, dir)
    assert isinstance(roundtrip2, BiocFrame)

    assert list(roundtrip2.column("akari")) == df.column("akari")
    assert roundtrip2.column("akari").dtype.type == np.int32

    assert roundtrip2.column("aika") == df.column("aika")
    assert isinstance(roundtrip2.column("aika"), StringList)

    assert list(roundtrip2.column("alice")) == df.column("alice")
    assert roundtrip2.column("alice").dtype.type == np.bool_

    assert list(roundtrip2.column("ai")) == df.column("ai")
    assert roundtrip2.column("ai").dtype.type == np.float64

    assert list(roundtrip2.column("alicia")) == df.column("alicia")
    assert roundtrip2.column("alicia").dtype.type == np.float64

    assert roundtrip2.column("akira") == df.column("akira")
    assert isinstance(roundtrip2.column("akira"), StringList)


def test_data_frame_external_list():
    df = BiocFrame({
        "akari": [ 1, 2, 3, 4, 5 ],
        "aika": [ "sydney", "melbourne", "", "perth", "adelaide" ],
        "alice": [ True, False, False, True, True ],
        "ai": [ 2.3, 1.2, 5.2, 3.1, -1.2 ],
    })

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo", convert_list_to_vector=False)
    assert meta["data_frame"]["columns"][0]["type"] == "other"
    assert meta["data_frame"]["columns"][1]["type"] == "other"
    assert meta["data_frame"]["columns"][2]["type"] == "other"
    assert meta["data_frame"]["columns"][3]["type"] == "other"
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert roundtrip.column("akari") == df.column("akari")
    assert roundtrip.column("aika") == df.column("aika")
    assert roundtrip.column("alice") == df.column("alice")
    assert roundtrip.column("ai") == df.column("ai")

    # Now for HDF5.
    meta = dl.stage_object(df, dir, "foo2", format="hdf5", convert_list_to_vector=False)
    dl.write_metadata(meta, dir)
    meta2 = dl.acquire_metadata(dir, "foo2/simple.h5")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert roundtrip.column("akari") == df.column("akari")
    assert roundtrip.column("aika") == df.column("aika")
    assert roundtrip.column("alice") == df.column("alice")
    assert roundtrip.column("ai") == df.column("ai")


def test_data_frame_factor():
    df = BiocFrame({
        "regular": Factor.from_sequence([ "sydney", "melbourne", "", "perth", "adelaide" ]),
        "missing": Factor.from_sequence([ "sydney", None, "", "perth", "adelaide" ]),
        "ordered": Factor.from_sequence([ "sydney", "melbourne", "", "perth", "adelaide" ], ordered=True),
    })

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0]["type"] == "factor"
    assert not meta["data_frame"]["columns"][0]["ordered"]
    assert meta["data_frame"]["columns"][1]["type"] == "factor"
    assert not meta["data_frame"]["columns"][1]["ordered"]
    assert meta["data_frame"]["columns"][2]["type"] == "factor"
    assert meta["data_frame"]["columns"][2]["ordered"]
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert isinstance(roundtrip["regular"], Factor)
    assert list(roundtrip["regular"]) == list(df["regular"])
    assert list(roundtrip["missing"]) == list(df["missing"])
    assert list(roundtrip["ordered"]) == list(df["ordered"])
    assert roundtrip["ordered"].get_ordered()

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    dl.write_metadata(meta2, dir)

    meta2 = dl.acquire_metadata(dir, "foo2/simple.h5")
    roundtrip2 = dl.load_object(meta2, dir)
    assert isinstance(roundtrip2["regular"], Factor)
    assert list(roundtrip2["regular"]) == list(df["regular"])
    assert list(roundtrip2["missing"]) == list(df["missing"])
    assert list(roundtrip2["ordered"]) == list(df["ordered"])
    assert roundtrip["ordered"].get_ordered()


def test_data_frame_row_names():
    df = BiocFrame({
        "akari": [ 1, 2, 3, 4, 5 ],
        "aika": [ "sydney", "melbourne", "", "perth", "adelaide" ],
        "alice": [ True, False, False, True, True ],
        "ai": [ 2.3, 1.2, 5.2, 3.1, -1.2 ],
    }, row_names = [ "kaori", "chihaya", "fuyuki", "azusa", "iori" ])

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["row_names"]
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.row_names == roundtrip.row_names

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    assert meta["data_frame"]["columns"] == meta2["data_frame"]["columns"]
    dl.write_metadata(meta2, dir)

    meta2 = dl.acquire_metadata(dir, "foo2/simple.h5")
    roundtrip2 = dl.load_object(meta2, dir)
    assert df.row_names == roundtrip2.row_names


def test_data_frame_wild_strings():
    df = BiocFrame({
        "lyrics": [ "Ochite\niku\tsunadokei\nbakari\tmiteru\tyo", "foo\"asdasd", "multie\"foo\"asdas" ],
    }, row_names = [ "nagisa\"", "\"fuko\"", "okazaki\nasdasd" ])

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["row_names"]
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.row_names == roundtrip.row_names
    assert df.column("lyrics") == roundtrip.column("lyrics")

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    assert meta["data_frame"]["columns"] == meta2["data_frame"]["columns"]
    dl.write_metadata(meta2, dir)

    meta2 = dl.acquire_metadata(dir, "foo2/simple.h5")
    roundtrip2 = dl.load_object(meta2, dir)
    assert df.row_names == roundtrip2.row_names
    assert df.column("lyrics") == roundtrip2.column("lyrics")


def test_data_frame_none():
    df = BiocFrame({
        "akari": [ 1, 2, None, 4, 5 ],
        "aika": [ "sydney", None, "", "perth", "adelaide" ],
        "alice": [ True, None, False, None, True ],
        "ai": [ 2.3, None, 5.2, None, -1.2 ],
        "aria": [ None, None, None, None, None ],
        "akira": [ "south", None, "north", "east", "west" ],
    })

    def compare_masked_to_list(l, m):
        assert len(l) == len(m)
        for i, x in enumerate(l):
            if x is not None:
                assert x == m[i]
            else:
                assert np.ma.is_masked(m[i])

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "akari" }
    assert meta["data_frame"]["columns"][1] == { "type": "string", "name": "aika" }
    assert meta["data_frame"]["columns"][2] == { "type": "boolean", "name": "alice" }
    assert meta["data_frame"]["columns"][3] == { "type": "number", "name": "ai" }
    assert meta["data_frame"]["columns"][4] == { "type": "string", "name": "aria" }
    assert meta["data_frame"]["columns"][5] == { "type": "string", "name": "akira" }
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)

    compare_masked_to_list(df.column("akari"), roundtrip.column("akari"))
    assert roundtrip.column("akari").dtype.type == np.int32

    assert roundtrip.column("aika") == df.column("aika")
    assert isinstance(roundtrip.column("aika"), StringList)

    compare_masked_to_list(df.column("alice"), roundtrip.column("alice"))
    assert roundtrip.column("alice").dtype.type == np.bool_

    compare_masked_to_list(df.column("ai"), roundtrip.column("ai"))
    assert roundtrip.column("ai").dtype.type == np.float64

    assert roundtrip.column("aria") == df.column("aria")

    assert roundtrip.column("akira") == df.column("akira")
    assert isinstance(roundtrip.column("akira"), StringList)

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    assert meta["data_frame"]["columns"] == meta2["data_frame"]["columns"]
    dl.write_metadata(meta2, dir)
    roundtrip2 = dl.load_object(meta2, dir)

    compare_masked_to_list(df.column("akari"), roundtrip2.column("akari"))
    assert roundtrip2.column("akari").dtype.type == np.int32

    assert roundtrip2.column("aika") == df.column("aika")
    assert isinstance(roundtrip2.column("aika"), StringList)

    compare_masked_to_list(df.column("alice"), roundtrip2.column("alice"))
    assert roundtrip2.column("alice").dtype.type == np.bool_

    compare_masked_to_list(df.column("ai"), roundtrip2.column("ai"))
    assert roundtrip2.column("ai").dtype.type == np.float64

    assert roundtrip2.column("aria") == df.column("aria")

    assert roundtrip2.column("akira") == df.column("akira")
    assert isinstance(roundtrip2.column("akira"), StringList)


def test_data_frame_numpy():
    df = BiocFrame({
        "alicia": np.array([ 1, 2, 3, 4, 5 ]),
        "akira": np.array([ True, True, False, False, True ]),
        "athena": np.array([ 2.3, 2.3, 5.2, 32, -1.2 ]),
    })

    dir = mkdtemp()

    # Test with CSV.
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

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    assert meta["data_frame"]["columns"] == meta2["data_frame"]["columns"]
    dl.write_metadata(meta2, dir)
    roundtrip2 = dl.load_object(meta2, dir)

    assert (roundtrip2.column("alicia") == df.column("alicia")).all()
    assert roundtrip2.column("alicia").dtype == np.int32
    assert (roundtrip2.column("akira") == df.column("akira")).all()
    assert (roundtrip2.column("athena") == df.column("athena")).all()


def test_data_frame_large_integers():
    df = BiocFrame({
        "alicia": [ 2**31 - 1, 1, 2, 3 ],
        "akira": [ 2**32, 1, 2, 3 ],
        "alice": np.array([ -2**31, 4, 5, 6 ], dtype=np.int64),
        "athena": np.array([ -2**32, 4, 5, 6 ], dtype=np.int64),
    })

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "alicia" }
    assert meta["data_frame"]["columns"][1] == { "type": "number", "name": "akira" }
    assert meta["data_frame"]["columns"][2] == { "type": "integer", "name": "alice" }
    assert meta["data_frame"]["columns"][3] == { "type": "number", "name": "athena" }
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)

    assert (roundtrip.column("alicia") == df.column("alicia")).all()
    assert roundtrip.column("alicia").dtype == np.int32
    assert (roundtrip.column("akira") == df.column("akira")).all()
    assert roundtrip.column("akira").dtype == np.float64
    assert (roundtrip.column("alice") == df.column("alice")).all()
    assert roundtrip.column("alice").dtype == np.int32
    assert (roundtrip.column("athena") == df.column("athena")).all()
    assert roundtrip.column("athena").dtype == np.float64

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    assert meta["data_frame"]["columns"] == meta2["data_frame"]["columns"]
    dl.write_metadata(meta2, dir)
    roundtrip2 = dl.load_object(meta2, dir)

    assert (roundtrip2.column("alicia") == df.column("alicia")).all()
    assert roundtrip2.column("alicia").dtype == np.int32
    assert (roundtrip2.column("akira") == df.column("akira")).all()
    assert roundtrip2.column("akira").dtype == np.float64
    assert (roundtrip2.column("alice") == df.column("alice")).all()
    assert roundtrip2.column("alice").dtype == np.int32
    assert (roundtrip2.column("athena") == df.column("athena")).all()
    assert roundtrip2.column("athena").dtype == np.float64


def test_data_frame_special_floats():
    df = BiocFrame({
        "sumire": [ np.NaN, np.Inf, -np.Inf ],
        "kanon": np.array([ np.NaN, np.Inf, -np.Inf ]),
        "chisato": np.ma.array([ np.NaN, 4, 5 ], mask=[0, 1, 1]) # distinguish NaN from missing.
    })

    def as_expected(x):
        assert np.isnan(x[0])
        assert x[1] == np.Inf
        assert x[2] == -np.Inf

    def as_expected_masked(x):
        assert np.isnan(x[0])
        assert np.ma.is_masked(x[1])
        assert np.ma.is_masked(x[2])

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_object(meta, dir)
    assert isinstance(roundtrip, BiocFrame)
    as_expected(roundtrip.column("sumire"))
    as_expected(roundtrip.column("kanon"))
    as_expected_masked(roundtrip.column("chisato"))

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    dl.write_metadata(meta2, dir)

    roundtrip2 = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    as_expected(roundtrip2.column("sumire"))
    as_expected(roundtrip2.column("kanon"))
    as_expected_masked(roundtrip2.column("chisato"))


def test_data_frame_masked():
    df = BiocFrame({
        "alicia": np.ma.array(np.array([ 1, 2, 3, 4, 5 ]), mask=[0, 1, 0, 1, 0]),
        "akira": np.ma.array(np.array([ True, True, False, False, True ]), mask=[1, 1, 0, 0, 0]), # important: test masking at the front.
        "athena": np.ma.array(np.array([ 2.3, -12.8, 5.2, 32, -1.2 ]), mask=[0, 0, 0, 1, 1]),
        "aika": np.ma.array(np.array([ 0, 0, 0, 0, 0 ]), mask=[1, 1, 1, 1, 1]),
    })

    dir = mkdtemp()

    # Test with CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][0] == { "type": "integer", "name": "alicia" }
    assert meta["data_frame"]["columns"][1] == { "type": "boolean", "name": "akira" }
    assert meta["data_frame"]["columns"][2] == { "type": "number", "name": "athena" }
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert (roundtrip.column("alicia") == df.column("alicia")).all()
    assert (roundtrip.column("alicia").mask == df.column("alicia").mask).all()
    assert (roundtrip.column("akira") == df.column("akira")).all()
    assert (roundtrip.column("akira").mask == df.column("akira").mask).all()
    assert (roundtrip.column("athena") == df.column("athena")).all()
    assert (roundtrip.column("athena").mask == df.column("athena").mask).all()
    assert (roundtrip.column("aika").mask == df.column("aika").mask).all()

    # Test with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    dl.write_metadata(meta2, dir)

    roundtrip2 = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)
    assert (roundtrip2.column("alicia") == df.column("alicia")).all()
    assert (roundtrip2.column("alicia").mask == df.column("alicia").mask).all()
    assert (roundtrip2.column("akira") == df.column("akira")).all()
    assert (roundtrip2.column("akira").mask == df.column("akira").mask).all()
    assert (roundtrip2.column("athena") == df.column("athena")).all()
    assert (roundtrip2.column("athena").mask == df.column("athena").mask).all()
    assert (roundtrip2.column("aika").mask == df.column("aika").mask).all()


def test_data_frame_empty():
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

    # Same for HDF5.
    df = BiocFrame(number_of_rows=10)
    meta = dl.stage_object(df, dir, "empty2", format="hdf5")
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "empty2/simple.h5")
    roundtrip2 = dl.load_object(meta2, dir)
    assert isinstance(roundtrip2, BiocFrame)
    assert df.shape == roundtrip2.shape
    assert roundtrip2.row_names is None


def test_data_frame_nested():
    df = BiocFrame({
        "odd": [ 1, 3, 5, 7, 9 ],
        "liella": BiocFrame({ 
            "first": [ "kanon", "keke", "chisato", "sumire", "ren" ],
            "last": [ "shibuya", "tang", "arashi", "heanna", "hazuki" ],
            "best": [ False, False, False, True, False ],
        }),
        "even": [ 0, 2, 4, 6, 8 ],
        "bsb": BiocFrame({ 
            "first": [ "nick", "kevin", "brian", "AJ", "howie" ],
            "last": [ "carter", "richardson", "litrell", "maclean", "dorough" ]
        }),
    })

    dir = mkdtemp()

    # Works for CSV.
    meta = dl.stage_object(df, dir, "foo")
    assert meta["data_frame"]["columns"][1]["type"] == "other"
    assert meta["data_frame"]["columns"][3]["type"] == "other"
    dl.write_metadata(meta, dir)

    meta2 = dl.acquire_metadata(dir, "foo/simple.csv.gz")
    roundtrip = dl.load_object(meta2, dir)
    assert isinstance(roundtrip, BiocFrame)

    liella_df = roundtrip.column("liella")
    assert isinstance(liella_df, BiocFrame)
    assert liella_df.column("first") == df.column("liella").column("first")
    assert liella_df.column("last") == df.column("liella").column("last")
    assert (liella_df.column("best") == np.array([False, False, False, True, False ])).all()

    bsb_df = roundtrip.column("bsb")
    assert isinstance(bsb_df, BiocFrame)
    assert bsb_df.column("first") == df.column("bsb").column("first")
    assert bsb_df.column("last") == df.column("bsb").column("last")

    # Works for HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    dl.write_metadata(meta2, dir)
    roundtrip2 = dl.load_object(meta2, dir)
    assert isinstance(roundtrip2, BiocFrame)

    liella_df = roundtrip2.column("liella")
    assert isinstance(liella_df, BiocFrame)
    assert liella_df.column("first") == df.column("liella").column("first")
    assert liella_df.column("last") == df.column("liella").column("last")

    bsb_df = roundtrip2.column("bsb")
    assert isinstance(bsb_df, BiocFrame)
    assert bsb_df.column("first") == df.column("bsb").column("first")
    assert bsb_df.column("last") == df.column("bsb").column("last")


def test_data_frame_dict_to_list():
    df = BiocFrame({"weird_dict": { "A": 1, "B": 2 } })
    dir = mkdtemp()
    meta = dl.stage_object(df, dir, "foo")
    dl.write_metadata(meta, dir)
    roundtrip = dl.load_object(meta, dir)
    assert roundtrip.get_column("weird_dict") == [1,2]


def test_data_frame_metadata():
    df = BiocFrame(
        { "foo": [ 1, 3, 5, 7, 9 ] },
        column_data = BiocFrame({ "args": [ 99 ] }),
        metadata = { "a": 2, "b": ['a', 'b', 'c', 'd'] }
    )

    dir = mkdtemp()

    # CSV first.
    meta = dl.stage_object(df, dir, "foo")
    assert "other_data" in meta["data_frame"]
    assert "column_data" in meta["data_frame"]
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_object(meta, dir)
    assert df.metadata == roundtrip.metadata
    assert roundtrip.get_column_data().column("args") == [ 99 ]
    assert isinstance(roundtrip, BiocFrame)

    # Any row names are deliberately stripped out.
    cd = df.get_column_data()
    cd2 = cd.set_row_names([ "ARGH" ])
    df2 = df.set_column_data(cd2)
    meta = dl.stage_object(df2, dir, "foo-empty")
    roundtrip = dl.load_object(meta, dir)
    assert roundtrip.get_column_data(with_names = False).get_row_names() is None

    # Trying with HDF5.
    meta2 = dl.stage_object(df, dir, "foo2", format="hdf5")
    dl.write_metadata(meta2, dir)
    roundtrip2 = dl.load_object(meta2, dir)
    assert df.metadata == roundtrip2.metadata
    assert roundtrip2.get_column_data().column("args") == [ 99 ]


def test_data_frame_format():
    assert dl.choose_data_frame_format() == "csv"
    old = dl.choose_data_frame_format("hdf5") 
    assert old == "csv"
    assert dl.choose_data_frame_format() == "hdf5"
    dl.choose_data_frame_format(old)
    assert dl.choose_data_frame_format() == "csv"
