"""
Helpers on top of MPh for building COMSOL geometries and selections.

mphkit builds geometry and named selections; physics, mesh, studies and
results stay plain MPh (or the COMSOL Java API through `node.java`).
Every helper takes MPh nodes, and those that create something return one.

    import mph
    import mphkit as mk

    client = mph.start()
    model = client.create('demo')
    geom = mk.geometry(model, 3, length_unit='mm')
    plate = mk.block(geom, (100, 100, 10))
    hole = mk.cylinder(geom, 5, 10, (50, 50, 0))
    mk.difference(geom, plate, [hole])
    model.build(geom)
    bottom = mk.sel.box(geom, 'boundary', z=0)    # select by location
    physics = (model/'physics').create('HeatTransfer', geom)
    physics.create('TemperatureBoundary', 2).select(bottom)   # 2: boundary level

Rules:

- Select entities by location with `mk.sel` (box, ball, cylinder, disk,
  adjacent, set operations), never by entity number: numbers change with
  the geometry, selections are re-evaluated by COMSOL.
- Entity kinds are COMSOL's: 'domain' (volumes in 3D), 'boundary' (faces
  in 3D, edges in 2D), 'edge', 'point' (vertices).
- Extra keyword arguments are COMSOL property names (`r`, `h`, `pos`,
  `size`, `rot`, ...); an unknown name raises an error that often suggests
  the right one.
- Build the geometry (`model.build(geom)`) before querying it, and again
  after adding a `where='geometry'` selection.

Check the result without looking at it: `mk.sel.entities(geom, sel)`
(entity numbers), `mk.sel.find(geom, 'domain', x=...)`,
`mk.measure(geom, 'domain')` (volume, area or length) and
`mk.bounding_box(geom, 'boundary', ...)`.

More: `help(mk.sel)` for selections (also in work planes), `help(mk.block)`
etc. for each helper, `mk.feature(geom, 'Type', ...)` for any other
geometry feature, `mk.set(node_or_java, **properties)` to set properties
of anything else (mesh, study, material, probe), and `node.java` for the
rest of the COMSOL API.
"""
import sys as _sys

from . import sel
from ._hints import HintModule as _HintModule
from ._measure import bounding_box, measure
from ._props import set_ as set  # not in __all__: keeps builtin set
from .errors import LicenseError
from .geometry import (array, block, chamfer, circle, component_of,
                       coordinate_system, cylinder, delete, difference,
                       extrude, feature, fillet, geometry, import_,
                       intersection, interval, line_segment, mirror, move,
                       partition, point, polygon, rectangle, revolve,
                       rigid_transform, rotate, sphere, square, union,
                       workplane)

__version__ = '0.2.0.dev0'

__all__ = ['LicenseError', 'array', 'block', 'bounding_box', 'chamfer',
           'circle', 'component_of', 'coordinate_system', 'cylinder',
           'delete', 'difference', 'extrude', 'feature', 'fillet',
           'geometry', 'import_', 'intersection', 'interval',
           'line_segment', 'measure', 'mirror', 'move', 'partition', 'point',
           'polygon', 'rectangle', 'revolve', 'rigid_transform', 'rotate',
           'sel', 'sphere', 'square', 'union', 'workplane']

# Unknown names raise errors that name the right helper (see _hints).
_sys.modules[__name__].__class__ = _HintModule
del _sys, _HintModule
