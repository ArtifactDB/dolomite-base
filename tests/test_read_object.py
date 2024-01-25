import dolomite_base as dl
import os
from tempfile import mkdtemp
import pytest


def test_read_object_failures():
    dir = mkdtemp()
    with open(os.path.join(dir, "OBJECT"), "w") as handle:
        handle.write('{ "type": "aaron" }')

    with pytest.raises(NotImplementedError, match="could not find") as ex:
        dl.read_object(dir)

    dl.read_object_registry["aaron"] = "dolomite_aaron.read_aaron"
    with pytest.raises(ModuleNotFoundError, match="no module named 'dolomite_aaron'") as ex:
        dl.read_object(dir)
