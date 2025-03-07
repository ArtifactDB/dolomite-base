import dolomite_base
import biocframe
import tempfile
import os
import pytest


df = biocframe.BiocFrame({
    "X": ["a", "b", "c", "d"],
    "Y": list(range(4)),
    "Z": biocframe.BiocFrame({
        "AA": [5,4,3,2]
    })
})


def test_validate_directory():
    tmp = tempfile.mkdtemp()

    dolomite_base.save_object(df, os.path.join(tmp, "foo"))
    os.mkdir(os.path.join(tmp, "whee"))
    dolomite_base.save_object(df, os.path.join(tmp, "whee", "stuff"))

    output = dolomite_base.validate_directory(tmp)
    assert sorted(output) == ["foo", "whee/stuff"]

    with open(os.path.join(tmp, "foo", "OBJECT"), "w") as handle:
        handle.write('{ "type": "WHEEE" }')
    with pytest.raises(ValueError, match="WHEEE"):
        dolomite_base.validate_directory(tmp)
