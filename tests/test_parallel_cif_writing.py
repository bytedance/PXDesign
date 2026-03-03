"""Test that annotations are set once before the CIF writing loop."""

from pathlib import Path


def test_annotations_set_before_loop():
    """Verify b_factor and occupancy are set before the per-sample loop."""
    source = (
        Path(__file__).parent.parent / "pxdesign" / "runner" / "dumper.py"
    ).read_text()
    method_start = source.find("def _save_structure")
    method_source = source[method_start:]
    b_factor_pos = method_source.find("b_factor")
    loop_pos = method_source.find("for sample_idx")
    assert b_factor_pos < loop_pos, (
        "Annotations (b_factor, occupancy) should be set once before the per-sample loop"
    )
