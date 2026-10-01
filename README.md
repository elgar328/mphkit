# mphkit

[![PyPI](https://img.shields.io/pypi/v/mphkit)](https://pypi.org/project/mphkit/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/elgar328/mphkit/main/pyproject.toml)](https://pypi.org/project/mphkit/)
[![License](https://img.shields.io/pypi/l/mphkit)](https://github.com/elgar328/mphkit/blob/main/LICENSE)
[![COMSOL](https://img.shields.io/badge/COMSOL-6.4-blue)](https://www.comsol.com/)

Helpers on top of [MPh](https://github.com/MPh-py/MPh) for building COMSOL
geometries and geometry-based selections in Python, inserting materials
from COMSOL's libraries, and reading and drawing results.
Select boundaries by location instead of by entity number, so selections
keep working when the geometry changes.

> [!WARNING]
> **Early stage.** The API may change at any time, without deprecation
> warnings.

Not affiliated with COMSOL AB.

## Example

```python
import mph
import mphkit as mk

client = mph.start()
model = client.create('demo')
geom = mk.geometry(model, 3, length_unit='mm')

plate = mk.block(geom, (100, 100, 10), name='plate')
hole = mk.cylinder(geom, 5, 10, (50, 50, 0))
mk.difference(geom, plate, [hole])
model.build(geom)

bottom = mk.sel.box(geom, 'boundary', z=0)     # faces on the plane z = 0

# plain MPh from here on
physics = (model/'physics').create('HeatTransfer', geom)
physics.create('TemperatureBoundary', 2).select(bottom)  # 2: boundaries in 3D
model.save('demo.mph')
```

Helpers take MPh `Node`s (`mk.geometry` the model, `mk.set` also Java
objects), and those that create something return one, so mphkit and MPh
mix freely. Physics, mesh and study stay plain MPh (or the COMSOL Java
API through `node.java`); mphkit looks up the COMSOL names they need.

`help(mphkit)` sums up the workflow, the rules and the common pitfalls,
and each helper has its own `help()`. Point an AI assistant to
`help(mphkit)` first.

## Requirements

- COMSOL Multiphysics with a license, installed where MPh can find it
  (see the [MPh documentation](https://mph.readthedocs.io)).
- Python 3.10 or newer, MPh 1.4 or newer.
- Importing CAD files (`mk.import_` with STEP, IGES, ...) needs a license
  for CAD import (CAD Import Module, Design Module or a LiveLink).
  Everything else needs COMSOL only.

mphkit is developed and tested with COMSOL 6.4 and MPh 1.4. Feature types,
property names and selection behavior can differ between COMSOL versions,
so other versions may need adjustments; reports are welcome. Checked on
macOS and Windows; it should run wherever MPh runs. The name lookups read
COMSOL's code-completion data in the installation (`data/completion`),
which COMSOL does not document; they were checked with COMSOL 6.4 on
macOS only. A missing or changed catalogue raises an error rather than
giving wrong names.

## Installation

```
pip install mphkit
```

or `uv add mphkit` in a uv project.

What changed between versions is listed in
[CHANGELOG.md](https://github.com/elgar328/mphkit/blob/main/CHANGELOG.md).

## What it covers

Geometry, in 3D, 2D and in work planes:

```python
mk.block(geom, (10, 10, 5)); mk.cylinder(geom, r, h, pos); mk.sphere(geom, r)
mk.union(geom, [a, b]); mk.difference(geom, a, [b]); mk.intersection(geom, [a, b])
mk.move(geom, part, (10, 0, 0)); mk.rotate(geom, part, 90, axis='z')
mk.mirror(geom, part, (1, 0, 0)); mk.array(geom, part, size=(5, 5, 1), displ=(10, 10, 0))
mk.fillet(geom, part, 0.5); mk.chamfer(geom, part, 0.5)     # all edges, or a selection
mk.partition(geom, part, tool); mk.delete(geom, part)
plane = mk.workplane(geom, quickz=0)
mk.circle(plane, 2); mk.extrude(geom, plane, 5); mk.revolve(geom, plane)
mk.import_(geom, 'part.step')
mk.feature(geom, 'AnyType', ...)   # any other geometry feature
```

Selections by location (`mk.sel`), usable in physics, materials and mesh:

```python
mk.sel.box(geom, 'boundary', z=0)                 # a range per axis, or a value
mk.sel.ball(geom, 'domain', center, r); mk.sel.cylinder(...); mk.sel.disk(...)
mk.sel.union(geom, 'boundary', [a, b])            # also intersection, difference, complement
mk.sel.adjacent(geom, domains)                    # boundaries around a domain selection
mk.sel.result(geom, feature, 'domain')            # what a feature produced
holes = mk.sel.cumulative(geom, 'holes', 'domain', create=True)
mk.cylinder(geom, 1, 5, pos, contributeto=holes)  # collect from several features
```

`where='geometry'` makes a selection inside the geometry sequence instead,
for use as input of a later operation, e.g. `mk.sel.box(geom, 'object',
x=(20, 40), where='geometry')` to pick whole objects for `mk.delete`. In a
work plane, selections can pick single corners or edges, e.g. to fillet
one corner: `mk.fillet(plane, mk.sel.box(plane, 'point', x=1, y=1), 0.3)`.

Queries on the plate from the Example section return plain Python values
and leave nothing in the model:

```python
mk.sel.entities(geom, bottom)            # [3]
mk.sel.find(geom, 'boundary', x=0)       # [1]: the face at x = 0
mk.measure(geom, 'domain')               # 99216.5: volume, approximate where curved
mk.bounding_box(geom, 'boundary', 3)     # {'x': (0.0, 100.0), 'y': ..., 'z': (0.0, 0.0)}
mk.summary(geom)                         # counts, voids, bounding box, unit
mk.sel.neighbors(geom, 'domain', boundary=3)  # [1]: the domain beside it
mk.coordinates(geom, 'boundary', 3)      # {1: (0.0, 0.0, 0.0), 3: (0.0, 100.0, 0.0), ...}
```

Before solving, a check of what COMSOL would get wrong silently or
vaguely (wrong units, domains without material, conditions that apply
nowhere, no mesh, physics no study solves):

```python
warnings = [p for p in mk.check(model) if p['severity'] == 'warning']
```

Results of the solved
[example script](https://github.com/elgar328/mphkit/blob/main/examples/plate_with_holes.py),
over entities or at points, in SI units unless `unit` is given; they
leave nothing in the model either:

```python
mk.integral(geom, 'boundary', 'ht.ntflux', selections['hot end'], unit='W')  # -2.74: flows in
mk.average(geom, 'domain', 'T', unit='degC')                   # 90.2
mk.minimum(geom, 'domain', 'T', unit='degC', position=True)    # (83.4, array([88.0, 20.0, ...])): a hole wall
mk.value(geom, 'T', [(50, 20, 2.5), (100, 20, 2.5)], unit='degC')  # array([89.8, 84.1])
```

`ht.ntflux` is the flux out of the domain. With several solutions, pass
`dataset=`; with time steps or a sweep, `step=`. A unit that does not fit,
a point outside the geometry or, in most cases, a geometry changed since
the solve raise instead of giving a wrong number. Global values come from MPh:
`model.evaluate('expression', 'unit')`.

Pictures, written to a file:

```python
mk.image(geom, 'geom.png')                              # the geometry
mk.image(geom, 'selection.png', selection, labels=True) # with numbers
mk.image(geom, 'mesh.png', mesh=True)                   # element quality
mk.plot(geom, 'T', 'T.png', unit='degC')                # a solved result
mk.plot(geom, 'T', 'mid.png', unit='degC', z=2.5, view='top')  # a slice from above
```

`mk.plot` also draws a selection only and deformed shapes
(`deform=True`); like the other helpers it leaves nothing in the model.

COMSOL's names, looked up instead of guessed (search first, the full
lists are long), here on the example script's model with
`heat = model/'physics'/'heat'`, `mesh = model/'meshes'/'mesh'` and
`study = model/'studies'/'static'`:

```python
mk.physics_types(geom, search='heat')         # 'HeatTransfer', ...
mk.feature_types(heat, search='convective')   # 'ConvectiveOutflow', then 'HeatFluxBoundary' ({'boundary': 2}) via a choice
mk.feature_types(mesh)                        # 'FreeTet', 'Size', ...: what mesh.create() takes
mk.feature_types(study, search='time')        # ..., 'Transient', ...: by name first
mk.properties(heat, 'HeatFluxBoundary')       # descriptions, defaults, choices
mk.properties(heat/'cooling', search='flux')  # and current values
mk.variables(heat, search='heat flux')        # 'ht.ntflux' [W/m^2], on boundaries
```

Physics features come with the level numbers `create()` takes.
`mk.properties` also takes geometry, mesh and study features and
materials, or a type to create there, e.g. `mk.properties(mesh, 'FreeTet')`.

Materials from COMSOL's libraries, instead of typing property values:

```python
mk.materials(search='structural steel')             # names, groups, properties
steel = mk.material(geom, 'Structural steel')       # background first: all domains
water = mk.material(geom, 'Water, liquid', channel) # later ones take a selection
```

`mk.materials()` lists the basic library and a search looks in all of
them; `mk.material(..., library='acdc')` takes the others. Each domain
takes the material added last among those that select it.

And a few helpers outside geometry:

```python
mk.coordinate_system(geom, 'PML', selection=layer)   # e.g. perfectly matched layers
mk.set(mesh_size, hmax=0.5, hgrad=2)   # any node or Java object; converts ints and lists
```

[`examples/plate_with_holes.py`](https://github.com/elgar328/mphkit/blob/main/examples/plate_with_holes.py)
is a complete script, from geometry to solved results: heat conduction in
a plate with a row of cooling holes, for one or more holes.

## Limitations

- mphkit covers geometry, selections, materials from COMSOL's
  libraries, reading and drawing results, and looking up COMSOL's names. Physics, mesh, studies and plots beyond
  `mk.plot` (arrows, streamlines, graphs, animations) are left to MPh.
- Parametric sweeps that COMSOL stores as an outer loop, e.g. around a
  time-dependent study or over a geometry parameter, cannot be read yet;
  other sweeps of a stationary study can.
- No named helpers yet for geometry parts (`PartInstance`), sweeps, cones
  and the other remaining primitives, virtual operations or repair. They
  work through `mk.feature(geom, 'Sweep', ...)`, which handles arguments
  like the named helpers (expressions, lists, nodes as inputs).

## Development

```
uv run pytest
```

The tests start COMSOL and build real models. Without a CAD import
license, the CAD import test is skipped.

## License

MIT, see [LICENSE](https://github.com/elgar328/mphkit/blob/main/LICENSE).
