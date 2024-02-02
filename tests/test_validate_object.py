import dolomite_base as dl
import os
from tempfile import mkdtemp
import pytest
import json
from biocutils import StringList


def test_validate_object_with_metadata():
    sl = StringList([1,2,3,4])
    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(sl, dir)

    # Check that validate_object accepts metadata arguments.
    with open(os.path.join(dir, "OBJECT"), "r") as handle:
        metadata = json.load(handle)
    assert isinstance(metadata, dict)
    dl.validate_object(dir, metadata=metadata)


def test_validate_object_registration():
    dir = mkdtemp()
    with open(os.path.join(dir, "OBJECT"), "w") as handle:
        handle.write('{ "type": "aaron" }')
    with pytest.raises(Exception, match="no registered") as ex:
        dl.validate_object(dir)

    def fun(path, metadata):
        pass
    dl.validate_object_registry["aaron"] = fun
    dl.validate_object(dir)

    def fun2(path, metadata):
        raise ValueError("WHEEE") 
    dl.validate_object_registry["aaron"] = fun2
    with pytest.raises(Exception, match="WHEEE") as ex:
        dl.validate_object(dir)

    dl.validate_object_registry["aaron"] = "WHEE"
    with pytest.raises(Exception, match="to contain a function") as ex:
        dl.validate_object(dir)

    del dl.validate_object_registry["aaron"] 
    dl.validate_object_registry[1] = fun
    with pytest.raises(Exception, match="should be strings") as ex:
        dl.validate_object(dir)

    del dl.validate_object_registry[1]
