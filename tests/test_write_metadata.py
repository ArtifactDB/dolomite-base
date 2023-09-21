import dolomite_base as dl
import tempfile
import json
import os


def _mockup(tmpdir, basename): 
    with open(os.path.join(tmpdir, basename), "w") as handle:
        handle.write("your mom")

    return {
        "$schema": "generic_file/v1.json",
        "path": basename,
        "generic_file": {
            "format": "text/plain"
        }
    }


def test_write_metadata_basic():
    tmpdir = tempfile.mkdtemp()
    basename = "foo.txt"
    full_meta = _mockup(tmpdir, basename)

    out = dl.write_metadata(full_meta, tmpdir)
    assert out["type"] == "local"
    assert out["path"] == basename

    with open(os.path.join(tmpdir, basename + ".json"), "r") as handle:
        roundtrip = json.load(handle)
    assert isinstance(roundtrip["md5sum"], str)
    assert roundtrip["generic_file"]["format"] == "text/plain"


def test_write_metadata_none():
    tmpdir = tempfile.mkdtemp()
    basename = "foo.txt"

    full_meta = _mockup(tmpdir, basename)
    full_meta["generic_file"]["foo"] = None

    # Nones are automatically stripped out.
    out = dl.write_metadata(full_meta, tmpdir)
    with open(os.path.join(tmpdir, basename + ".json"), "r") as handle:
        roundtrip = json.load(handle)
    assert "foo" not in roundtrip["generic_file"]


def test_write_metadata_metaonly():
    tmpdir = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmpdir, "yay"))

    basename = "yay/whee.json"
    full_meta = {
        "$schema": "summarized_experiment/v1.json",
        "path": basename,
        "summarized_experiment": {
            "dimensions": [ 100, 10 ],
            "assays": [
                {
                    "name": "counts",
                    "resource": {
                        "type": "local",
                        "path": "yay/assay-1/simple.h5"
                    }
                } 
            ]
        }
    }

    # Nones are automatically stripped out.
    out = dl.write_metadata(full_meta, tmpdir)
    with open(os.path.join(tmpdir, basename), "r") as handle:
        roundtrip = json.load(handle)
    assert roundtrip == full_meta
    assert "md5sum" not in roundtrip


def test_write_metadata_sanitized():
    tmpdir = tempfile.mkdtemp()
    basename = "./foo.txt"
    full_meta = _mockup(tmpdir, basename)

    # We automatically remove the preceding './'.
    out = dl.write_metadata(full_meta, tmpdir)
    with open(os.path.join(tmpdir, basename + ".json"), "r") as handle:
        roundtrip = json.load(handle)
    assert roundtrip["path"] == "foo.txt"


def test_write_metadata_tuple():
    tmpdir = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmpdir, "stuff"))

    basename = "./stuff/foo.txt"
    full_meta = _mockup(tmpdir, basename)
    full_meta["$schema"] = (full_meta["$schema"], "dolomite_schemas")

    out = dl.write_metadata(full_meta, tmpdir)
    with open(os.path.join(tmpdir, basename + ".json"), "r") as handle:
        roundtrip = json.load(handle)
    assert roundtrip["$schema"] == "generic_file/v1.json"
    assert roundtrip["path"] == "stuff/foo.txt"
