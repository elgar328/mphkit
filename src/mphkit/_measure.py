"""
Measurements of the finished geometry. Public as `mk.measure`,
`mk.bounding_box`, `mk.summary` and `mk.coordinates`.

They return plain values in the geometry's length unit and leave nothing
in the model. Curved geometry is measured on a rendering mesh, so volumes,
areas and lengths of curved entities are approximate (a cylinder about
0.3 % too small, a sphere 0.3 to 0.6 % depending on the geometry kernel).
Planar geometry is exact up to single precision: the measurement rounds
coordinates to about seven digits, so a thickness that is small compared
with the coordinate across it loses accuracy, e.g. a 0.001 thick plate at
z = 1000 measures 0.00098 and at z = 1000.3 measures 0.00104. The
geometry itself is not affected.
"""
from __future__ import annotations


import numpy
from mph.node import Node

from . import _comsol

AXES = 'xyz'


def numbers_of(geom: Node, entity: str, selection,
               every: bool = True) -> list[int]:
    """
    Returns the entity numbers `selection` stands for in the built geometry.

    `selection` is an entity number, a list of numbers, a selection node at
    the level of `entity`, or `None` for all entities of that level. With
    `every`, a string such as `'boundary'` or `'all'` is answered with
    leaving out the selection; callers where `None` means something else
    turn it off.
    """
    _comsol.check_built(geom)
    dim = _comsol.entity_dim(geom, entity)
    count = _comsol.entity_count(geom, dim)
    if selection is None:
        return list(range(1, count + 1))
    if isinstance(selection, Node):
        java = _comsol.selection_at(geom, selection, entity, dim)
        return [int(e) for e in java.entities()]
    many = (isinstance(selection, (list, tuple))
            or isinstance(selection, numpy.ndarray) and selection.ndim > 0)
    items = selection if many else [selection]
    found = []
    for item in items:
        if not _comsol.is_integer(item):
            named = ''
            if isinstance(item, str):
                every_ = (f'leave out the selection (None) for all {entity} '
                          'entities' if every else
                          "a selection by name is model/'selections'/"
                          f'{item!r}')
                named = '; ' + _comsol.selection_hint(geom, item, every_)
            raise TypeError(f'Expected entity numbers, a selection node '
                            f'or None, not {selection!r}{named}.')
        found.append(int(item))
    wrong = [n for n in found if not 1 <= n <= count]
    if wrong:
        raise ValueError(f'No {entity} {wrong} in geometry "{geom}"; it '
                         f'has {count} {entity} entities.')
    return found


def _final(geom: Node, entity: str, selection):
    """
    Returns a measurement of the finished geometry, or `None` if empty.

    `selection` works as in `numbers_of()`.
    """
    found = numbers_of(geom, entity, selection)
    if not found:
        return None
    dim = _comsol.entity_dim(geom, entity)
    measurement = _comsol.java_of(geom).measureFinal()
    with _comsol.history_off(geom.model.java):
        measurement.selection().geom(geom.tag(), dim)
        measurement.selection().set(found)
    return measurement


def measure(geom: Node, entity: str, /, selection=None) -> float:
    """
    Returns the size of entities: volume, area or length, in the
    geometry's length unit (mm³ for a volume with `length_unit='mm'`).

    What is measured follows the level of `entity`: the volume of domains
    (the area of 2D domains), the area of boundaries, the length of edges.
    `selection` is an entity number, a list of them, a selection node, or
    `None` for all entities of that kind. Several entities give the sum; a
    number given twice counts once. An empty selection gives 0.

    The geometry must be built. Curved entities are measured on a
    rendering mesh and are approximate (a cylinder about 0.3 % too small);
    planar ones are exact up to single precision: coordinates are rounded
    to about seven digits, so a 0.001 thick plate at z = 1000 measures
    0.00098. Compare sizes with a tolerance.
    """
    _comsol.check_not_workplane(geom, 'measure')
    if _comsol.entity_dim(geom, entity) == 0:
        raise ValueError('Points have no size; use bounding_box() for their '
                         'coordinates.')
    measurement = _final(geom, entity, selection)
    return 0.0 if measurement is None else float(measurement.getVolume())


def bounding_box(geom: Node, entity: str, /, selection=None) -> dict | None:
    """
    Returns the bounding box of entities as `{'x': (min, max), ...}`, in
    the geometry's length unit; of the whole geometry with
    `bounding_box(geom, 'domain')` (or `mk.summary(geom)['bounding_box']`).

    Has one pair per space dimension, so it can be passed on as
    `sel.box(geom, entity, **bbox)`. For a single point, min equals max:
    its coordinates. `selection` works as in `measure()`. Returns `None`
    for an empty selection. Values at the `'domain'`, `'boundary'` and
    `'edge'` levels are single precision (1.1 reads 1.100000023841858);
    points keep double precision. After a rotation, 0 may read about
    1e-16 at any level. Compare with a tolerance; `sel.box` allows for
    that.
    """
    _comsol.check_not_workplane(geom, 'bounding_box')
    measurement = _final(geom, entity, selection)
    if measurement is None:
        return None
    values = [float(v) for v in measurement.getBoundingBox()]
    return {axis: (values[2*i], values[2*i + 1])
            for i, axis in enumerate(AXES[:len(values) // 2])}


def summary(geom: Node, /) -> dict:
    """
    Returns a summary of the finished geometry, for example::

        {'dimension': 3, 'domains': 2, 'boundaries': 11, 'edges': 20,
         'points': 12, 'voids': 0,
         'bounding_box': {'x': (0.0, 2.0), 'y': (0.0, 1.0), 'z': (0.0, 1.0)},
         'length_unit': 'mm'}

    It has the dimension, the number of entities of each kind (2D has no
    `'edges'`, 1D no `'boundaries'` either) and of voids, the bounding box
    and the length unit. Voids are enclosed empty regions, e.g. left by a
    sphere subtracted from a block, or a hole in 2D; a through-hole is not
    one. The bounding box is in double precision, unlike `bounding_box()`,
    but may differ from the drawn size in the last digits; compare with a
    tolerance. A quick check that the geometry came out as meant, e.g. one
    domain after a union.
    """
    _comsol.check_not_workplane(geom, 'summary')
    _comsol.check_built(geom)
    java = _comsol.java_of(geom)
    dim = _comsol.sdim(geom)
    result: dict = {'dimension': dim}
    for entity in ('domain', 'boundary', 'edge', 'point'):
        if entity == 'boundary' and dim < 2 or entity == 'edge' and dim < 3:
            continue
        level = _comsol.entity_dim(geom, entity)
        plural = 'boundaries' if entity == 'boundary' else f'{entity}s'
        result[plural] = _comsol.entity_count(geom, level)
    result['voids'] = int(java.getNFiniteVoids())
    values = [float(v) for v in java.getBoundingBox()]
    result['bounding_box'] = {axis: (values[2*i], values[2*i + 1])
                              for i, axis in enumerate(AXES[:dim])}
    result['length_unit'] = str(java.lengthUnit())
    return result


def coordinates(geom: Node, entity: str = 'point', /,
                selection=None) -> dict[int, tuple]:
    """
    Returns the vertices of entities as `{number: (x, y, z)}` (`(x, y)` in
    2D, `(x,)` in 1D), in the geometry's length unit, e.g. the corners of a
    face with `coordinates(geom, 'boundary', 6)`.

    `selection` works as in `measure()`; left out, the vertices of all
    entities of that kind, so `coordinates(geom)` gives every vertex.
    Unlike `bounding_box()`, which gives single-precision extents for
    domains, boundaries and edges, these are the exact vertex coordinates.
    """
    _comsol.check_not_workplane(geom, 'coordinates')
    found = numbers_of(geom, entity, selection)
    java = _comsol.java_of(geom)
    level = _comsol.entity_dim(geom, entity)
    vertices: set[int] = set()
    for number in found:
        if level == 0:
            vertices.add(number)
        else:
            vertices.update(int(v) for v in java.getAdj(level, 0, number))
    table = [[float(v) for v in row] for row in java.getVertexCoord()]
    return {n: tuple(row[n - 1] for row in table) for n in sorted(vertices)}
