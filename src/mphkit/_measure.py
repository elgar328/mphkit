"""
Measurements of the finished geometry. Public as `mk.measure` and
`mk.bounding_box`.

Both return plain numbers in the geometry's length unit and leave nothing
in the model. Curved geometry is measured on a rendering mesh, so volumes,
areas and lengths of curved entities are approximate (a cylinder about
0.3 % too small, a sphere 0.3 to 0.6 % depending on the geometry kernel);
planar geometry is exact.
"""
from __future__ import annotations

import numbers

import numpy
from mph import Node

from . import _comsol

AXES = 'xyz'


def _final(geom: Node, entity: str, selection):
    """
    Returns a measurement of the finished geometry, or `None` if empty.

    `selection` is an entity number, a list of numbers, a selection node at
    the level of `entity`, or `None` for all entities of that level.
    """
    _comsol.check_built(geom)
    dim = _comsol.entity_dim(geom, entity)
    count = _comsol.entity_count(geom, dim)
    if selection is None:
        found = list(range(1, count + 1))
    elif isinstance(selection, Node):
        java = _comsol.check_selection(geom, selection)
        level = [int(d) for d in java.dimension()]
        if level != [dim]:
            raise ValueError(f'Selection "{selection}" is not a {entity} '
                             'selection.')
        found = [int(e) for e in java.entities()]
    else:
        items = selection if isinstance(
            selection, (list, tuple, numpy.ndarray)) else [selection]
        found = []
        for item in items:
            if isinstance(item, bool) or not isinstance(item, numbers.Integral):
                raise TypeError(f'Expected entity numbers, a selection node '
                                f'or None, not {selection!r}.')
            found.append(int(item))
        wrong = [n for n in found if not 1 <= n <= count]
        if wrong:
            raise ValueError(f'No {entity} {wrong} in geometry "{geom}"; it '
                             f'has {count} {entity} entities.')
    if not found:
        return None
    measurement = geom.java.measureFinal()
    measurement.selection().geom(geom.tag(), dim)
    measurement.selection().set(found)
    return measurement


def measure(geom: Node, entity: str, /, selection=None) -> float:
    """
    Returns the size of entities: volume, area or length.

    What is measured follows the level of `entity`: the volume of domains
    (the area of 2D domains), the area of boundaries, the length of edges.
    `selection` is an entity number, a list of them, a selection node, or
    `None` for all entities of that kind. Several entities give the sum; a
    number given twice counts once. An empty selection gives 0.

    The geometry must be built. Curved entities are measured on a
    rendering mesh and are approximate (a cylinder about 0.3 % too small);
    planar ones are exact.
    """
    _comsol.check_not_workplane(geom, 'measure')
    if _comsol.entity_dim(geom, entity) == 0:
        raise ValueError('Points have no size; use bounding_box() for their '
                         'coordinates.')
    measurement = _final(geom, entity, selection)
    return 0.0 if measurement is None else float(measurement.getVolume())


def bounding_box(geom: Node, entity: str, /, selection=None) -> dict | None:
    """
    Returns the bounding box of entities as `{'x': (min, max), ...}`.

    Has one pair per space dimension, so it can be passed on as
    `sel.box(geom, entity, **box)`. Allow a margin then: the values may
    carry single-precision noise, which makes `condition='inside'` miss
    the entity itself. For a single point, min equals max: its
    coordinates. `selection` works as in `measure()`. Returns `None` for
    an empty selection.
    """
    _comsol.check_not_workplane(geom, 'bounding_box')
    measurement = _final(geom, entity, selection)
    if measurement is None:
        return None
    values = [float(v) for v in measurement.getBoundingBox()]
    return {axis: (values[2*i], values[2*i + 1])
            for i, axis in enumerate(AXES[:len(values) // 2])}
