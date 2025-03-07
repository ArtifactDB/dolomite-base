import tempfile 
import dolomite_base
import biocframe
import biocutils
import os


def test_list_objects():
    tmp = tempfile.mkdtemp()

    df = biocframe.BiocFrame({ "A": list(range(10)), "B": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j" ]})
    dolomite_base.save_object(df, os.path.join(tmp, "whee"))

    ll = { "A": 1, "B": ["a", "b", "c", "d"], "C": biocframe.BiocFrame({ "X": list(range(5)) }) }
    dolomite_base.save_object(ll, os.path.join(tmp, "stuff"))

    listing = dolomite_base.list_objects(tmp)
    assert listing.shape[0] == 2
    m = biocutils.match(["whee", "stuff"], listing["path"])
    assert all([x >= 0 for x in m])
    assert biocutils.subset_sequence(listing["type"], m) == ["data_frame", "simple_list"] 
    assert not any(listing["child"])

    listing = dolomite_base.list_objects(tmp, include_children=True)
    assert listing.shape[0] == 3
    m = biocutils.match(["whee", "stuff", "stuff/other_contents/0"], listing["path"])
    assert all([x >= 0 for x in m])
    assert biocutils.subset_sequence(listing["type"], m) == ["data_frame", "simple_list", "data_frame"] 
    assert biocutils.subset_sequence(listing["child"], m) == [False, False, True]
