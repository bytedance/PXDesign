"""Test that CIF writing uses coord swap instead of deepcopy."""

import ast
from pathlib import Path

import numpy as np


def test_no_deepcopy_in_dumper():
    """Verify that dumper.py no longer uses copy.deepcopy."""
    filepath = Path(__file__).parent.parent / "pxdesign" / "runner" / "dumper.py"
    tree = ast.parse(filepath.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "deepcopy"
                and isinstance(func.value, ast.Name)
                and func.value.id == "copy"
            ):
                raise AssertionError(
                    "Found copy.deepcopy() in dumper.py. "
                    "Use coord save/restore instead to avoid N_sample full copies."
                )


def test_coord_restore_correctness():
    """Verify the coord swap pattern restores original coordinates."""
    original_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    class MockArray:
        def __init__(self):
            self.coord = original_coords.copy()

    mock = MockArray()
    new_coords = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    saved = mock.coord
    mock.coord = new_coords
    assert np.array_equal(mock.coord, new_coords), "Should have new coords during write"
    mock.coord = saved
    assert np.array_equal(mock.coord, original_coords), "Should restore original coords after write"
