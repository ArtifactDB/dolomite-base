import dolomite_base as dl
from biocutils import StringList, Factor
from tempfile import mkdtemp
from dolomite_base.stage_atomic_vector import _stage_atomic_vector
import numpy


def test_string_factor_basic():
    regular = Factor.from_sequence([ "sydney", "melbourne", "brisbane", "perth", "adelaide" ])
    missing = Factor.from_sequence([ "sydney", None, None, None, None ])
    ordered = Factor.from_sequence([ "sydney", "melbourne", "brisbane", "perth", "adelaide" ], ordered=True, levels=["sydney", "melbourne", "adelaide", "perth", "brisbane"])

    dir = mkdtemp()

    # Regular:
    meta = dl.stage_object(regular, dir, "regular")
    assert meta["factor"]["length"] == 5
    assert meta["string_factor"]["ordered"] is None
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_object(meta, dir)
    assert isinstance(roundtrip, Factor)
    assert (roundtrip.get_codes() == regular.get_codes()).all()
    assert roundtrip.get_levels() == regular.get_levels()
    assert roundtrip.get_ordered() == regular.get_ordered()

    # With a missing value:
    meta = dl.stage_object(missing, dir, "missing")
    assert meta["factor"]["length"] == 5
    assert meta["string_factor"]["ordered"] is None
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_object(meta, dir)
    assert isinstance(roundtrip, Factor)
    assert (roundtrip.get_codes() == missing.get_codes()).all()
    assert roundtrip.get_levels() == missing.get_levels()
    assert roundtrip.get_ordered() == missing.get_ordered()

    # Ordered:
    meta = dl.stage_object(ordered, dir, "ordered")
    assert meta["factor"]["length"] == 5
    assert meta["string_factor"]["ordered"]
    dl.write_metadata(meta, dir)

    roundtrip = dl.load_object(meta, dir)
    assert isinstance(roundtrip, Factor)
    assert (roundtrip.get_codes() == ordered.get_codes()).all()
    assert roundtrip.get_levels() == ordered.get_levels()
    assert roundtrip.get_ordered() == ordered.get_ordered()
