"""
Helpers on top of MPh for building COMSOL geometries and selections,
inserting materials from COMSOL's libraries, and reading and drawing
results.

mphkit builds geometry and named selections, inserts materials from
COMSOL's libraries, and reads and draws results; physics, mesh and
studies stay plain MPh (or the COMSOL Java API through `node.java`), and
mphkit looks up the COMSOL names they need.
Helpers take MPh nodes (`mk.geometry` the model, `mk.set` also Java
objects), and those that create something return one.

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
    # 2: boundaries in 3D
    temp = physics.create('TemperatureBoundary', 2)
    temp.select(bottom)

Rules:

- Select entities by location with `mk.sel` (box, ball, cylinder, disk,
  adjacent, set operations), never by entity number: numbers change with
  the geometry, selections are re-evaluated by COMSOL.
- Besides locations, `mk.sel.result` selects what a feature left in the
  geometry and `mk.sel.cumulative` collects groups across operations
  (features join with `contributeto=`).
- Entity kinds are COMSOL's: 'domain' (volumes in 3D, areas in 2D),
  'boundary' (faces in 3D, edges in 2D, where 'edge' means the same),
  'edge', 'point' (vertices). Physics features take the level as a
  number: 3 for domains and 2 for boundaries in 3D, 2 and 1 in 2D.
- Objects that touch or overlap stay separate domains, even after
  `mk.union`; merge them with `mk.union(geom, [a, b], intbnd=False)`,
  which also removes layers and partition cuts of its inputs. A full
  `mk.revolve` (no `angle`) keeps its cross-section as an interior face
  unless `origfaces=False`.
- Extra keyword arguments are COMSOL property names (`intbnd`, `keep`,
  `layername`, `base`, ...); an unknown name raises an error that often
  suggests the right one.
- Sizes, positions, angles and counts may be numbers or COMSOL expressions
  with parameters and units, e.g. `'L-2*t'` or `'5[mm]'`; define
  parameters with `model.parameter('t', '0.3[mm]')`.
- Sizes and coordinates are in the geometry's length unit.
- Build the geometry (`model.build(geom)`) before querying it, and again
  after adding a `where='geometry'` selection.

Check the geometry from code:

- `mk.sel.entities(geom, selection)`: entity numbers of a selection
- `mk.sel.find(geom, 'domain', x=...)`: entity numbers inside a box
- `mk.measure(geom, 'domain')`: volume, area or length
- `mk.bounding_box(geom, 'boundary', 3)`: extents
- `mk.summary(geom)`: counts, voids, size
- `mk.sel.neighbors(geom, 'domain', boundary=3)`: adjacent entities
- `mk.coordinates(geom, 'boundary', 3)`: vertex coordinates
- `mk.image(geom, 'geom.png')`: a picture of the geometry
- `mk.image(geom, 'selection.png', selection, labels=True)`: a selection
  highlighted, with entity numbers
- `mk.image(geom, 'mesh.png', mesh=True)`: the mesh, coloured by element
  quality (after `model.mesh()`)

Curved entities are measured approximately (see `help(mk.measure)`).
`mk.measure` and `mk.bounding_box` use single precision, about seven
digits of the coordinates (`mk.bounding_box` of points excepted);
`mk.coordinates` and `mk.summary` use double precision. Compare with a
tolerance either way.

Selection tips: `mk.sel.result(geom, feature, 'boundary')` gives the faces
that an object and its copies leave in the geometry, curved ones
included, and the faces it cuts into other objects, e.g. the spherical
faces a subtracted sphere leaves in a block. Curved surfaces are split
into several faces: the side of a cylinder or cone, the wall of a hole or
of an extruded circle into four, a sphere into eight. For the side of a
cylinder that no other object cut, use a thin shell:
`mk.sel.cylinder(geom, 'boundary', pos, r, rin=0.99*r, bottom=0, top=h,
name='side')` (`rin` needs about 1 % room, see
`help(mk.sel.cylinder)`; a name avoids a clash with the cylinder's own
label).

Finding COMSOL's names, instead of guessing them (search first, the
full lists are long):

    mk.physics_types(geom, search='heat')        # 'HeatTransfer', ...
    mk.feature_types(physics, search='convective')  # types and levels
    mk.feature_types(geom)        # also a mesh, mesh feature or study
    mk.properties(physics, 'HeatFluxBoundary')   # before creating one
    mk.properties(temp, search='temperature')    # 'T0': value, default
    mk.variables(physics, search='heat flux')    # 'ht.ntflux', ...

For physics features, `levels` gives the number `create()` takes for each
level, e.g. `{'boundary': 2}`; geometry, mesh and study types have
`levels=None`. `properties` also takes geometry, mesh and study features,
materials and `mk.properties(mesh, 'FreeTet')`-style types.

Materials from COMSOL's libraries, instead of typing property values:

    mk.materials(search='structural steel')    # names, groups, properties
    steel = mk.material(geom, 'Structural steel')      # all domains
    water = mk.material(geom, 'Water, liquid', channel)  # a selection

Add the background material first, without a selection; later materials
need one, since each domain takes the material added last among those
that select it. `mk.materials()` lists the basic library, a `search`
looks in all of them (`library=` in `mk.material` for the others).

Results, once the model is solved (`model.solve()`):

    mk.integral(geom, 'boundary', 'ht.ntflux', bottom, unit='W')
    mk.average(geom, 'domain', 'T', unit='degC')
    mk.maximum(geom, 'domain', 'T', unit='degC', position=True)
    mk.value(geom, 'T', [(50, 20, 5), (0, 0, 0)], unit='degC')
    mk.plot(geom, 'T', 'T.png', unit='degC')           # a picture
    mk.plot(geom, 'T', 'mid.png', unit='degC', z=2.5, view='top')

They check what COMSOL would silently get wrong: a unit that does not
fit, a geometry changed since the solve (unless built and meshed again),
a point outside the geometry. Changed physics, materials or parameters
need `model.solve()` again; that goes unnoticed.
Values are in SI units unless `unit` is given, also in an mm geometry;
positions and points in the geometry's length unit. `ht.ntflux` is the
flux out of the domain, negative where heat enters. With several
solutions pass `dataset=`, with several steps (time, sweep, frequency)
`step='last'`, a number, a list or `'all'` (a picture shows one step).
`mk.plot` also draws a selection only, slices (`x=`, `y=` or `z=`), views
from a side (`view='top'`, ...) and deformed shapes (`deform=True`).
Global values and values at all mesh nodes:
`model.evaluate('expression', 'unit')` in MPh.

Existing models (e.g. built in the COMSOL Desktop): load one with
`old = client.load('file.mph')`; `old.reset()` compacts its history and
`old.save('old.java')` writes its current state as Java, with COMSOL's
feature and property names (`old.save()` without a path would overwrite
the .mph). Rebuild the geometry with mphkit and the rest with plain MPh.
A plain number, as in `selection().set(4)`, counts entities of the
finished geometry. Look it up in the loaded one,
`g = (old/'geometries').children()[0]` if it is the only one: check that
`mk.sel.find(g, 'boundary', **mk.bounding_box(g, 'boundary', 4))` gives
`[4]`, then pass the same ranges to `mk.sel.box(geom, 'boundary', ...)`.
Numbers after an object name, as in `set("dif1(1)", 3)`, count that
object's entities during the build, not in the finished geometry: select
them by location from the drawn shapes, with `where='geometry'`, or in a
work plane.

More: `help(mk.sel)` for selections (also in work planes), `help(mk.block)`
etc. for each helper, `mk.feature(geom, 'Type', ...)` for any other
geometry feature, `mk.set(node_or_java, **properties)` to set properties
of anything else (mesh, study, material, probe), and `node.java` for the
rest of the COMSOL API.
"""
import sys as _sys

from . import sel
from ._catalog import feature_types, physics_types, properties, variables
from ._hints import HintModule as _HintModule
from ._image import image
from ._materials import material, materials
from ._measure import bounding_box, coordinates, measure, summary
from ._plot import plot
from ._props import set_ as set  # not in __all__: keeps builtin set
from ._results import average, integral, maximum, minimum, value
from .errors import LicenseError
from .geometry import (array, block, chamfer, circle, component_of,
                       coordinate_system, cylinder, delete, difference,
                       extrude, feature, fillet, geometry, import_,
                       intersection, interval, line_segment, mirror, move,
                       partition, point, polygon, rectangle, revolve,
                       rigid_transform, rotate, sphere, square, union,
                       workplane)

__version__ = '0.3.0.dev0'

__all__ = ['LicenseError', 'array', 'average', 'block', 'bounding_box',
           'chamfer', 'circle', 'component_of', 'coordinate_system',
           'coordinates', 'cylinder', 'delete', 'difference', 'extrude',
           'feature', 'feature_types', 'fillet', 'geometry', 'image',
           'import_', 'integral', 'intersection', 'interval',
           'line_segment', 'material', 'materials', 'maximum', 'measure',
           'minimum', 'mirror', 'move',
           'partition', 'physics_types', 'plot', 'point', 'polygon',
           'properties', 'rectangle', 'revolve', 'rigid_transform',
           'rotate', 'sel', 'sphere', 'square', 'summary', 'union', 'value',
           'variables', 'workplane']

# Unknown names raise errors that name the right helper (see _hints).
_sys.modules[__name__].__class__ = _HintModule
del _sys, _HintModule
