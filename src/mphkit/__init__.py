"""
Helpers on top of MPh for COMSOL; overview: print(mphkit.__doc__), not help().

mphkit builds geometry and named selections, inserts materials from
COMSOL's libraries, looks up COMSOL's names, and reads and draws results;
physics, mesh and studies stay plain MPh (or COMSOL's Java API through
`node.java`). Helpers take MPh nodes, and those that create something
return one. This overview names every helper (Index, at the end);
`help(mk.<name>)` has the details, and `help(mphkit)` appends all of
them, at length.

    import mph
    import mphkit as mk

    client = mph.start()
    model = client.create('demo')
    model.parameter('Th', '100[degC]')
    geom = mk.geometry(model, 3, length_unit='mm')
    plate = mk.block(geom, (100, 100, 10))
    hole = mk.cylinder(geom, 5, 10, (50, 50, 0))
    mk.difference(geom, plate, [hole])
    model.build(geom)
    bottom = mk.sel.box(geom, 'boundary', z=0)    # by location
    mk.sel.entities(geom, bottom)                 # [3]: check it
    mk.physics_types(geom, search='heat')         # 'HeatTransfer', ...
    physics = (model/'physics').create('HeatTransfer', geom)
    mk.feature_types(physics, search='temperature')  # levels: boundary 2
    temp = physics.create('TemperatureBoundary', 2)
    temp.select(bottom)
    mk.properties(temp, search='temperature')     # 'T0': ...
    temp.property('T0', 'Th')
    steel = mk.material(geom, 'Structural steel')  # own values: see help
    (model/'meshes').create(geom)                 # COMSOL's default mesh
    study = (model/'studies').create(name='heating')
    study.create('Transient').property('tlist', 'range(0,1,10)')
    sweep = study.create('Parametric')
    mk.set(sweep, pname=['Th'], plistarr=['100 200 300'], punit=['degC'])
    problems = [p for p in mk.check(model) if p['severity'] == 'warning']
    assert not problems, problems
    model.solve('heating')
    mk.variables(physics, search='heat flux')     # 'ht.ntflux', ...
    heat = mk.integral(geom, 'boundary', 'ht.ntflux', bottom, unit='W',
                       outer='all', step='last')  # a value per Th
    tmax = mk.maximum(geom, 'domain', 'T', unit='degC', outer='all',
                      step='last')
    table = [dict(values, Tmax=value)             # a row per Th
             for values, value in zip(mk.outer_values(geom), tmax)]
    mk.plot(geom, 'T', 'T_{outer}.png', unit='degC', outer='all',
            step='last')                          # a picture per Th
    # a stationary sweep is stored as steps: mk.outer_values raises
    # "as steps"; read it with step= and mk.step_values

Results are in SI units unless `unit` is given, also in an mm geometry
(`ht.ntflux` is the flux out of the domain, negative where heat enters).
`dataset=` picks one of several solutions, a dataset or its study.
`step=` picks steps by position from 1 ('first', 'last', a number, a
list, 'all') or by value: `step=10` is the tenth step, `step={'t': 10}`
the one at t = 10 s (a warning tells when another step has t = 10 s). A
sweep stored as an outer loop needs `outer=` in every call, in the same
forms; `mk.outer_values(geom)` tells which kind a sweep is. MPh's
`model.evaluate` reads only the last value of such a sweep unless given
its dataset.

Rules:

- Solve again after any change: changed physics, materials, parameters
  or a geometry built and meshed again are read with the old solution.
- Select by location with `mk.sel`, never by entity number: numbers
  change with the geometry.
- Entity kinds are COMSOL's: 'domain', 'boundary' (in 2D 'edge' means
  the same), 'edge', 'point'. Physics features take the level as a
  number: 3 for domains and 2 for boundaries in 3D, 2 and 1 in 2D.
- Sizes and coordinates are in the geometry's length unit, and may be
  COMSOL expressions with parameters and units, e.g. 'L-2*t', '5[mm]'.
- Build the geometry (`model.build(geom)`) before querying it, and again
  after adding a `where='geometry'` selection.
- Extra keyword arguments are COMSOL property names; an unknown one
  raises an error that often suggests the right one or the
  `mk.properties` call that lists them.
- Objects that touch or overlap stay separate domains, also after
  `mk.union`; `mk.union(geom, [a, b], intbnd=False)` merges them.

Existing models: `old = client.load('file.mph')`, then `old.reset()` (it
keeps the solutions) and `old.save('old.java')` show it as Java
(`old.save()` without a path overwrites the .mph); `help(mk.sel.find)`
moves numbered selections and tells how to inspect the old model.

Index:

- Make geometry: mk.geometry, mk.block, mk.cylinder, mk.sphere,
  mk.rectangle, mk.square, mk.circle, mk.polygon, mk.interval, mk.point,
  mk.line_segment, mk.workplane, mk.extrude, mk.revolve, mk.union,
  mk.difference, mk.intersection, mk.partition, mk.delete, mk.move,
  mk.rotate, mk.mirror, mk.array, mk.rigid_transform, mk.fillet,
  mk.chamfer, mk.import_ (CAD files), mk.feature (any other feature).
- Select by location: mk.sel.box, mk.sel.ball, mk.sel.cylinder,
  mk.sel.disk, mk.sel.all, mk.sel.union, mk.sel.intersection,
  mk.sel.difference, mk.sel.complement, mk.sel.adjacent (boundaries of
  domains), mk.sel.result (what a feature left), mk.sel.layer (a layer of
  a block), mk.sel.cumulative (collected across features); common
  points, e.g. where='geometry' or work planes: print(mk.sel.__doc__).
- Check the geometry: mk.sel.entities, mk.sel.find, mk.sel.neighbors
  (adjacent entities), mk.measure, mk.bounding_box, mk.summary,
  mk.coordinates, mk.image.
- COMSOL's names: mk.physics_types, mk.feature_types, mk.properties,
  mk.variables; search first.
- Materials: mk.materials, mk.material.
- Before solving: mk.check (what COMSOL gets wrong silently),
  mk.problem_size, mk.mesh_quality.
- Long solves: mk.log_progress, mk.progress (from another process).
- Results: mk.integral, mk.average, mk.maximum, mk.minimum, mk.value,
  mk.plot; sweeps: mk.outer_values, mk.step_values (values of the outer
  loop and of the steps).
- Other: mk.set (properties of any node or Java object), mk.component_of
  (a geometry's component), mk.coordinate_system (e.g. for perfectly
  matched layers), mk.LicenseError (no CAD license), node.java.
"""
import sys as _sys

from . import sel
from ._catalog import feature_types, physics_types, properties, variables
from ._check import check
from ._hints import HintModule as _HintModule
from ._image import image
from ._materials import material, materials
from ._measure import bounding_box, coordinates, measure, summary
from ._mesh import mesh_quality
from ._plot import plot
from ._props import set_ as set  # not in __all__: keeps builtin set
from ._results import average, integral, maximum, minimum, value
from ._solve import log_progress, problem_size, progress
from ._sweep import outer_values, step_values
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
           'chamfer', 'check', 'circle', 'component_of', 'coordinate_system',
           'coordinates', 'cylinder', 'delete', 'difference', 'extrude',
           'feature', 'feature_types', 'fillet', 'geometry', 'image',
           'import_', 'integral', 'intersection', 'interval', 'line_segment',
           'log_progress', 'material', 'materials', 'maximum', 'measure',
           'mesh_quality', 'minimum', 'mirror', 'move', 'outer_values',
           'partition', 'physics_types', 'plot', 'point', 'polygon',
           'problem_size', 'progress', 'properties', 'rectangle', 'revolve',
           'rigid_transform', 'rotate', 'sel', 'sphere', 'square',
           'step_values', 'summary', 'union', 'value', 'variables',
           'workplane']

# Unknown names raise errors that name the right helper (see _hints).
_sys.modules[__name__].__class__ = _HintModule
del _sys, _HintModule
