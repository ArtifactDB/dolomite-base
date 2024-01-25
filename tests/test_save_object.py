import dolomite_base as dl
import os
from tempfile import mkdtemp
import pytest
import numpy
import sys


def test_save_object_dispatch():
    class WHEE:
        def __init__(self):
            pass

    tmp = os.path.join(mkdtemp(), "foo")
    a = WHEE()
    with pytest.raises(NotImplementedError, match="not implemented for type 'WHEE'") as ex:
        dl.save_object(a, tmp)

    # This next part checks that dolomite_matrix was successfully loaded,
    # but only if it is installed; otherwise the test is omitted.
    is_okay = True 
    y = numpy.random.rand(100, 200)
    tmp = os.path.join(mkdtemp(), "foo")

    try:
        dl.save_object(y, tmp)
    except:
        is_okay = False

    has_matrix = True
    try: 
        import dolomite_matrix
    except:
        has_matrix = False

    if has_matrix:
        assert is_okay
