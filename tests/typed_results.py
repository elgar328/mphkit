"""
Return types of the results helpers as a caller sees them. Checked by
mypy (test_types.py) and pyright, never run.
"""
from pathlib import Path
from typing import Any

import numpy
from mph.node import Node
from numpy.typing import NDArray
from typing_extensions import assert_type  # typing has it from 3.11

import mphkit as mk


def check(geom: Node, flag: bool, step: int | str) -> None:
    assert_type(mk.integral(geom, 'boundary', 'ht.ntflux', unit='W'), float)
    assert_type(mk.average(geom, 'domain', 'T', step='last'), float)
    assert_type(mk.average(geom, 'domain', 'T', step=numpy.int64(2)), float)
    assert_type(mk.average(geom, 'domain', 'T', step='all'),
                NDArray[Any])
    assert_type(mk.integral(geom, 'domain', 'T', step=[1, 2]),
                NDArray[Any])
    assert_type(mk.maximum(geom, 'domain', 'T'), float)
    assert_type(mk.maximum(geom, 'domain', 'T', position=True),
                tuple[float, NDArray[Any]])
    assert_type(mk.minimum(geom, 'domain', 'T', step='all', position=True),
                tuple[NDArray[Any], NDArray[Any]])
    assert_type(mk.value(geom, 'T', (0, 0, 0)), float | NDArray[Any])
    assert_type(mk.plot(geom, 'T', 'T.png', view='top', step='last'), Path)
    assert_type(mk.image(geom, 'mesh.png', mesh=True), Path)
    assert_type(mk.physics_types(geom, search='heat'), list[dict])
    assert_type(mk.feature_types(geom, search='flux'), list[dict])
    assert_type(mk.properties(geom, 'Block'), dict[str, dict])
    assert_type(mk.variables(geom), list[dict])
    assert_type(mk.materials(search='steel'), list[dict])
    assert_type(mk.check(geom.model), list[dict])
    assert_type(mk.material(geom, 'Copper', [1], library='basic_material'),
                Node)
    # values not known before the call
    mk.average(geom, 'domain', 'T', step=step)
    mk.maximum(geom, 'domain', 'T', position=flag)
    if mk.integral(geom, 'domain', 'T') > 0:
        pass
