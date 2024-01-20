import dolomite_base as dl
from biocutils import StringList, Factor
from tempfile import mkdtemp
import os
import numpy


def test_string_factor():
    regular = Factor.from_sequence([ "sydney", "melbourne", "brisbane", "perth", "adelaide" ])

    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(regular, dir)
    roundtrip = dl.read_object(dir)

    assert isinstance(roundtrip, Factor)
    assert (roundtrip.get_codes() == regular.get_codes()).all()
    assert roundtrip.get_levels() == regular.get_levels()
    assert roundtrip.get_ordered() == regular.get_ordered()


def test_string_factor_missing():
    missing = Factor.from_sequence([ "sydney", None, None, None, None ])

    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(missing, dir)
    roundtrip = dl.read_object(dir)

    assert isinstance(roundtrip, Factor)
    assert (roundtrip.get_codes() == missing.get_codes()).all()
    assert roundtrip.get_levels() == missing.get_levels()
    assert roundtrip.get_ordered() == missing.get_ordered()


def test_string_factor_ordered():
    ordered = Factor.from_sequence([ "sydney", "melbourne", "brisbane", "perth", "adelaide" ], ordered=True, levels=["sydney", "melbourne", "adelaide", "perth", "brisbane"])

    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(ordered, dir)
    roundtrip = dl.read_object(dir)

    assert isinstance(roundtrip, Factor)
    assert (roundtrip.get_codes() == ordered.get_codes()).all()
    assert roundtrip.get_levels() == ordered.get_levels()
    assert roundtrip.get_ordered() == ordered.get_ordered()


def test_string_factor_named():
    named = Factor.from_sequence([ "sydney", "melbourne", "brisbane", "perth", "adelaide" ])
    named.set_names(["A", "B", "C", "D", "E"], in_place=True)

    dir = os.path.join(mkdtemp(), "temp")
    dl.save_object(named, dir)
    roundtrip = dl.read_object(dir)

    assert isinstance(roundtrip, Factor)
    assert roundtrip.get_names() == named.get_names()
