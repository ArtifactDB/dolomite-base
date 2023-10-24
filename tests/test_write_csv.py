import dolomite_base as dl
from biocframe import BiocFrame
from tempfile import mkdtemp
import os


def test_write_csv_compressed():
    x = BiocFrame({ "a": [1,2,3,4], "b": [ "A", "B", "C", "D" ] })

    dir = mkdtemp()
    full = os.path.join(dir, "foo.csv.gz")
    dl.write_csv(x, full, compressed = True)

    out = dl.read_csv(full, 4, True)
    assert out["names"] == ["a", "b"]
    assert list(out["fields"][0]) == x.column('a')
    assert out["fields"][1] == x.column('b')


def test_write_csv_uncompressed():
    x = BiocFrame({ "a": [1,2,3,4], "b": [ "A", "B", "C", "D" ] })

    dir = mkdtemp()
    full = os.path.join(dir, "foo.csv.gz")
    dl.write_csv(x, full, compressed = False)

    out = dl.read_csv(full, 4, False)
    assert out["names"] == ["a", "b"]
    assert list(out["fields"][0]) == x.column('a')
    assert out["fields"][1] == x.column('b')
