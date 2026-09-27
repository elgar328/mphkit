"""
Geometry-based selections: select entities by location, not by number.

Every helper takes the geometry node first and returns an MPh node that
physics, materials, mesh and other features can `select()`. The queries
`entities` and `find` return entity numbers instead and leave nothing in
the model.

With `where='geometry'`, the level can also be `'object'`: whole geometry
objects, as COMSOL's "Object" level. Such a selection is only an input for
geometry operations (e.g. deleting or uniting the unnamed copies of an
array in a region), not something physics can use, so it is returned as
the selection feature in the geometry sequence rather than a
`selections/` node.

`where='component'` (the default for a geometry) creates a selection in
the component. It is evaluated on the finished geometry.
`where='geometry'` creates it inside the geometry sequence instead. Then
it can also be the input of later geometry operations, but it only sees
objects created before it.

A work plane can take the place of the geometry in `box`, `ball`, `disk`,
`all` and the set operations and `adjacent`, e.g. to fillet single
corners: `mk.fillet(plane, mk.sel.box(plane, 'point', x=1, y=1), 0.3)`.
Coordinates are the plane's own. Such selections live in the plane's
sequence (the default there), only see objects created before them, and
are inputs of operations in that plane only: COMSOL derives no
model-level selection from them, and physics refuses them ("Unknown
selection"). They are returned as the selection feature in the plane.
"""
from ._sel import adjacent, all_ as all, ball, box, complement, \
    cumulative, cylinder, difference, disk, entities, find, intersection, \
    layer, result, union

__all__ = ['adjacent', 'all', 'ball', 'box', 'complement', 'cumulative',
           'cylinder', 'difference', 'disk', 'entities', 'find',
           'intersection', 'layer', 'result', 'union']
