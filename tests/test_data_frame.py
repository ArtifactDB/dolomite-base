from biocframe import BiocFrame
from biocutils import Factor, StringList, IntegerList, BooleanList, FloatList
import dolomite_base as dl
import numpy as np
import os
import h5py
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

    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BiocFrame)

    assert list(roundtrip.get_column("akari")) == df.get_column("akari")
    assert roundtrip.get_column("akari").dtype.type == np.int32

    assert isinstance(roundtrip.get_column("aika"), StringList)
    assert roundtrip.get_column("aika").as_list() == df.get_column("aika")

    assert list(roundtrip.get_column("alice")) == df.get_column("alice")
    assert roundtrip.get_column("alice").dtype.type == np.bool_

    assert list(roundtrip.get_column("ai")) == df.get_column("ai")
    assert roundtrip.get_column("ai").dtype.type == np.float64

    assert list(roundtrip.get_column("alicia")) == df.get_column("alicia")
    assert roundtrip.get_column("alicia").dtype.type == np.float64

    assert roundtrip.get_column("akira") == df.get_column("akira")
    assert isinstance(roundtrip.get_column("akira"), StringList)


def test_data_frame_external_list():
    df = BiocFrame({
        "akari": [ 1, 2, 3, 4, 5 ],
        "aika": [ "sydney", "melbourne", "", "perth", "adelaide" ],
        "alice": [ True, False, False, True, True ],
        "ai": [ 2.3, 1.2, 5.2, 3.1, -1.2 ],
    })

    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(df, dir, data_frame_convert_list_to_vector=False)
    assert os.path.exists(os.path.join(dir, "other_columns", "0"))

    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BiocFrame)
    assert roundtrip.get_column("akari").as_list() == df.get_column("akari")
    assert roundtrip.get_column("aika").as_list() == df.get_column("aika")
    assert roundtrip.get_column("alice").as_list() == df.get_column("alice")
    assert roundtrip.get_column("ai").as_list() == df.get_column("ai")


def test_data_frame_factor():
    df = BiocFrame({
        "regular": Factor.from_sequence([ "sydney", "melbourne", "", "perth", "adelaide" ]),
        "missing": Factor.from_sequence([ "sydney", None, "", "perth", "adelaide" ]),
        "ordered": Factor.from_sequence([ "sydney", "melbourne", "", "perth", "adelaide" ], ordered=True),
    })

    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    assert isinstance(roundtrip["regular"], Factor)
    assert list(roundtrip["regular"]) == list(df["regular"])
    assert list(roundtrip["missing"]) == list(df["missing"])
    assert list(roundtrip["ordered"]) == list(df["ordered"])
    assert roundtrip["ordered"].get_ordered()


def test_data_frame_row_names():
    df = BiocFrame({
        "akari": [ 1, 2, 3, 4, 5 ],
        "aika": [ "sydney", "melbourne", "", "perth", "adelaide" ],
        "alice": [ True, False, False, True, True ],
        "ai": [ 2.3, 1.2, 5.2, 3.1, -1.2 ],
    }, row_names = [ "kaori", "chihaya", "fuyuki", "azusa", "iori" ])

    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    assert df.row_names == roundtrip.row_names


def test_data_frame_wild_strings():
    df = BiocFrame({
        "lyrics": [ "Ochite\niku\tsunadokei\nbakari\tmiteru\tyo", "foo\"asdasd", "multie\"foo\"asdas" ],
    }, row_names = [ "nagisa\"", "\"fuko\"", "okazaki\nasdasd" ])

    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    assert df.row_names == roundtrip.row_names
    assert df.get_column("lyrics") == roundtrip.get_column("lyrics").as_list()


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

    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    compare_masked_to_list(df.get_column("akari"), roundtrip.get_column("akari"))
    assert roundtrip.get_column("akari").dtype.type == np.int32

    assert roundtrip.get_column("aika").as_list() == df.get_column("aika")
    assert isinstance(roundtrip.get_column("aika"), StringList)

    compare_masked_to_list(df.get_column("alice"), roundtrip.get_column("alice"))
    assert roundtrip.get_column("alice").dtype.type == np.bool_

    compare_masked_to_list(df.get_column("ai"), roundtrip.get_column("ai"))
    assert roundtrip.get_column("ai").dtype.type == np.float64

    assert roundtrip.get_column("aria").as_list() == df.get_column("aria")

    assert roundtrip.get_column("akira").as_list() == df.get_column("akira")
    assert isinstance(roundtrip.get_column("akira"), StringList)


def test_data_frame_numpy():
    df = BiocFrame({
        "alicia": np.array([ 1, 2, 3, 4, 5 ]),
        "akira": np.array([ True, True, False, False, True ]),
        "athena": np.array([ 2.3, 2.3, 5.2, 32, -1.2 ]),
        "aika": np.array([ "Chihaya", "Haruka", "Iori", "Azusa", "Ritsuko" ]),
    })

    dir = os.path.join(mkdtemp(), "foo")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    assert (roundtrip.get_column("alicia") == df.get_column("alicia")).all()
    assert roundtrip.get_column("alicia").dtype == np.int32
    assert (roundtrip.get_column("akira") == df.get_column("akira")).all()
    assert roundtrip.get_column("akira").dtype == np.bool_
    assert (roundtrip.get_column("athena") == df.get_column("athena")).all()
    assert roundtrip.get_column("athena").dtype == np.float64
    assert roundtrip.get_column("aika").as_list() == list(df.get_column("aika"))

    # Coerces integers to floats.
    with h5py.File(os.path.join(dir, "basic_columns.h5"), "a") as handle:
        ghandle = handle["data_frame"]
        dhandle = ghandle["data"]
        xhandle = dhandle["0"]
        del xhandle.attrs["type"]
        xhandle.attrs.create("type", data="number")

    roundtrip = dl.read_object(dir)
    assert (roundtrip.get_column("alicia") == df.get_column("alicia")).all()
    assert roundtrip.get_column("alicia").dtype == np.float64


def test_data_frame_no_numpy():
    df = BiocFrame({
        "alicia": np.array([ 1, 2, 3, 4, 5 ]),
        "akira": np.array([ True, True, False, False, True ]),
        "athena": np.array([ 2.3, 2.3, 5.2, 32, -1.2 ]),
    })

    dir = os.path.join(mkdtemp(), "foo")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir, data_frame_represent_numeric_column_as_1darray=False)

    assert isinstance(roundtrip.get_column("alicia"), IntegerList)
    assert roundtrip.get_column("alicia").as_list() == list(df.get_column("alicia"))
    assert isinstance(roundtrip.get_column("akira"), BooleanList)
    assert roundtrip.get_column("akira").as_list() == list(df.get_column("akira"))
    assert isinstance(roundtrip.get_column("athena"), FloatList)
    assert roundtrip.get_column("athena").as_list() == list(df.get_column("athena"))


def test_data_frame_NamedList():
    df = BiocFrame({
        "alicia": IntegerList([ 1, 2, 3, 4, 5 ]),
        "akira": BooleanList([ True, True, False, False, True ]),
        "athena": FloatList([ 2.3, 2.3, 5.2, 32, -1.2 ]),
    })

    dir = os.path.join(mkdtemp(), "foo")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir, data_frame_represent_numeric_column_as_1darray=False)

    assert roundtrip.get_column("alicia") == df.get_column("alicia")
    assert roundtrip.get_column("akira") == df.get_column("akira")
    assert roundtrip.get_column("athena") == df.get_column("athena")


def test_data_frame_large_integers():
    df = BiocFrame({
        "alicia": [ 2**31 - 1, 1, 2, 3 ],
        "akira": [ 2**32, 1, 2, 3 ],
        "alice": np.array([ -2**31, 4, 5, 6 ], dtype=np.int64),
        "athena": np.array([ -2**32, 4, 5, 6 ], dtype=np.int64),
    })

    dir = os.path.join(mkdtemp(), "hdf5")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    assert (roundtrip.get_column("alicia") == df.get_column("alicia")).all()
    assert roundtrip.get_column("alicia").dtype == np.int32
    assert (roundtrip.get_column("akira") == df.get_column("akira")).all()
    assert roundtrip.get_column("akira").dtype == np.float64
    assert (roundtrip.get_column("alice") == df.get_column("alice")).all()
    assert roundtrip.get_column("alice").dtype == np.int32
    assert (roundtrip.get_column("athena") == df.get_column("athena")).all()
    assert roundtrip.get_column("athena").dtype == np.float64


def test_data_frame_special_floats():
    df = BiocFrame({
        "sumire": [ np.nan, np.inf, -np.inf ],
        "kanon": np.array([ np.nan, np.inf, -np.inf ]),
        "chisato": np.ma.array([ np.nan, 4, 5 ], mask=[0, 1, 1]) # distinguish NaN from missing.
    })

    def as_expected(x):
        assert np.isnan(x[0])
        assert x[1] == np.inf
        assert x[2] == -np.inf

    def as_expected_masked(x):
        assert np.isnan(x[0])
        assert np.ma.is_masked(x[1])
        assert np.ma.is_masked(x[2])

    dir = os.path.join(mkdtemp(), "foo2")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    assert isinstance(roundtrip, BiocFrame)
    as_expected(roundtrip.get_column("sumire"))
    as_expected(roundtrip.get_column("kanon"))
    as_expected_masked(roundtrip.get_column("chisato"))


def test_data_frame_masked():
    df = BiocFrame({
        "alicia": np.ma.array(np.array([ 1, 2, 3, 4, 5 ]), mask=[0, 1, 0, 1, 0]),
        "akira": np.ma.array(np.array([ True, True, False, False, True ]), mask=[1, 1, 0, 0, 0]), # important: test masking at the front.
        "athena": np.ma.array(np.array([ 2.3, -12.8, 5.2, 32, -1.2 ]), mask=[0, 0, 0, 1, 1]),
        "aika": np.ma.array(np.array([ 0, 0, 0, 0, 0 ]), mask=[1, 1, 1, 1, 1]),
    })

    dir = os.path.join(mkdtemp(), "foo")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)

    assert isinstance(roundtrip, BiocFrame)
    assert (roundtrip.get_column("alicia") == df.get_column("alicia")).all()
    assert (roundtrip.get_column("alicia").mask == df.get_column("alicia").mask).all()
    assert (roundtrip.get_column("akira") == df.get_column("akira")).all()
    assert (roundtrip.get_column("akira").mask == df.get_column("akira").mask).all()
    assert (roundtrip.get_column("athena") == df.get_column("athena")).all()
    assert (roundtrip.get_column("athena").mask == df.get_column("athena").mask).all()
    assert (roundtrip.get_column("aika").mask == df.get_column("aika").mask).all()


def test_data_frame_empty():
    # Empty except for row names.
    df = BiocFrame({}, row_names=["chihaya", "mami", "ami", "miki", "haruka"])
    dir = os.path.join(mkdtemp(), "empty")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.row_names == roundtrip.row_names

    # Fully empty.
    df = BiocFrame(number_of_rows=10)
    dir = os.path.join(mkdtemp(), "empty")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BiocFrame)
    assert df.shape == roundtrip.shape
    assert roundtrip.row_names is None


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

    dir = os.path.join(mkdtemp(), "foo2")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)
    assert isinstance(roundtrip, BiocFrame)

    liella_df = roundtrip.get_column("liella")
    assert isinstance(liella_df, BiocFrame)
    assert liella_df.get_column("first").as_list() == df.get_column("liella").get_column("first")
    assert liella_df.get_column("last").as_list() == df.get_column("liella").get_column("last")
    assert list(liella_df.get_column("best")) == df.get_column("liella").get_column("best")

    bsb_df = roundtrip.get_column("bsb")
    assert isinstance(bsb_df, BiocFrame)
    assert bsb_df.get_column("first").as_list() == df.get_column("bsb").get_column("first")
    assert bsb_df.get_column("last").as_list() == df.get_column("bsb").get_column("last")


def test_data_frame_metadata():
    df = BiocFrame(
        { "foo": [ 1, 3, 5, 7, 9 ] },
        column_data = BiocFrame({ "args": [ 99 ] }),
        metadata = { "a": 2, "b": ['a', 'b', 'c', 'd'] }
    )

    dir = os.path.join(mkdtemp(), "foo")
    dl.save_object(df, dir)
    roundtrip = dl.read_object(dir)
    roundtrip.metadata["b"] = roundtrip.metadata["b"].as_list()
    assert df.metadata == roundtrip.metadata
    assert roundtrip.get_column_data().get_column("args") == [ 99 ]
    assert isinstance(roundtrip, BiocFrame)

    # Any row names are deliberately stripped out.
    cd = df.get_column_data()
    cd2 = cd.set_row_names([ "ARGH" ])
    df2 = df.set_column_data(cd2)

    dir = os.path.join(mkdtemp(), "foo-empty")
    dl.save_object(df2, dir)
    roundtrip = dl.read_object(dir)
    assert roundtrip.get_column_data(with_names = False).get_row_names() is None
