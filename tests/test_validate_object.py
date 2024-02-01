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
    dl.register_validate_object_function("aaron", fun)
    dl.validate_object(dir)

    # Doesn't override by default...
    def fun2(path, metadata):
        raise ValueError("WHEEE") 
    dl.register_validate_object_function("aaron", fun2)
    dl.validate_object(dir)

    # Until explicitly requested...
    with pytest.raises(Exception, match="already been registered") as ex:
        dl.register_validate_object_function("aaron", fun2, existing="error")
    dl.register_validate_object_function("aaron", fun2, existing="new")
    with pytest.raises(Exception, match="WHEEE") as ex:
        dl.validate_object(dir)

    dl.register_validate_object_function("aaron", None)
    with pytest.raises(Exception, match="no registered") as ex:
        dl.validate_object(dir)
