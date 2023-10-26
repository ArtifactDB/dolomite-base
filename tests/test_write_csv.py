import dolomite_base as dl
from biocframe import BiocFrame
from tempfile import mkdtemp
import os
import pytest


def test_write_csv_compressed():
    x = BiocFrame({ "a": [1,2,3,4], "b": [ "A", "B", "C", "D" ] })

    dir = mkdtemp()
    full = os.path.join(dir, "foo.csv.gz")
    dl.write_csv(x, full, compression = "gzip")

    out = dl.read_csv(full, 4, "gzip")
    assert out["names"] == ["a", "b"]
    assert list(out["fields"][0]) == x.column('a')
    assert out["fields"][1] == x.column('b')

    with pytest.raises(NotImplementedError) as ex:
        out = dl.read_csv(full, 4, "foo")
    assert str(ex.value).find("foo") >= 0


def test_write_csv_uncompressed():
    x = BiocFrame({ "a": [1,2,3,4], "b": [ "A", "B", "C", "D" ] })

    dir = mkdtemp()
    full = os.path.join(dir, "foo.csv.gz")
    dl.write_csv(x, full, compression = "none")

    out = dl.read_csv(full, 4, "none")
    assert out["names"] == ["a", "b"]
    assert list(out["fields"][0]) == x.column('a')
    assert out["fields"][1] == x.column('b')
