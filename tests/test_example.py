"""
Checks the plate-with-holes example: selections for several hole counts.

Exact counts, because a band that is too wide or too narrow would still
produce non-empty selections.
"""
import importlib.util
from pathlib import Path

import pytest

import mphkit as mk

example = Path(__file__).parents[1]/'examples'/'plate_with_holes.py'
spec = importlib.util.spec_from_file_location('plate_with_holes', example)
plate_with_holes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plate_with_holes)


@pytest.mark.parametrize('holes', [1, 2, 5])
def test_selections(model, holes):
    geom, selections = plate_with_holes.build_geometry(model, holes)
    counts = {name: len(mk.sel.entities(geom, selection))
              for name, selection in selections.items()}
    # COMSOL splits the wall of each hole into four faces
    assert counts == {'hot end': 1, 'cold end': 1, 'hole walls': 4*holes}


def test_solve(client):
    model, geom, selections = plate_with_holes.build_model(client, 2)
    try:
        model.solve()
        temperature = model.evaluate('T', 'degC')
        assert temperature.max() == pytest.approx(100, abs=0.1)
        assert 20 < temperature.min() < 100
    finally:
        client.remove(model)
