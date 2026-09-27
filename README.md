# mphkit

[![PyPI](https://img.shields.io/pypi/v/mphkit)](https://pypi.org/project/mphkit/)
[![Python](https://img.shields.io/pypi/pyversions/mphkit)](https://pypi.org/project/mphkit/)
[![License](https://img.shields.io/pypi/l/mphkit)](https://github.com/elgar328/mphkit/blob/main/LICENSE)

Helpers on top of [MPh](https://github.com/MPh-py/MPh) for building COMSOL
geometries and geometry-based selections in Python. Select boundaries by
location instead of by entity number, so selections keep working when the
geometry changes.

> [!WARNING]
> **Early stage.** mphkit is at an early stage of development. The API may
> change at any time, without deprecation warnings.

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
physics.create('TemperatureBoundary', 2).select(bottom)
model.save('demo.mph')
```

Every helper takes MPh `Node`s, and those that create something return
one, so mphkit and MPh mix freely. Physics, mesh, study and results stay
plain MPh (or the COMSOL Java API through `node.java`).

## Requirements

- COMSOL Multiphysics with a license, installed where MPh can find it
  (see the [MPh documentation](https://mph.readthedocs.io)).
- Python 3.10 or newer, MPh 1.4 or newer.
- Importing CAD files (`mk.import_` with STEP, IGES, ...) needs a license
  for CAD import (CAD Import Module, Design Module or a LiveLink).
  Everything else needs COMSOL only.

Tested with COMSOL 6.4 and MPh 1.4.0, on macOS (Apple silicon) with
Python 3.10 and 3.13 and on Windows with Python 3.13. Linux is not tested
yet; since mphkit only goes through MPh, it is expected to work the same
way.

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
mk.fillet(geom, block, 0.5); mk.chamfer(geom, block, 0.5)   # all edges, or a selection
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
work plane, selections pick single corners or edges:
`mk.fillet(plane, mk.sel.box(plane, 'point', x=1, y=1), 0.3)`.

Queries return plain Python values and leave nothing in the model:

```python
mk.sel.entities(geom, bottom)            # [3]
mk.sel.find(geom, 'domain', x=(0, 10))   # entity numbers inside a box
mk.measure(geom, 'domain')               # volume (area, length for other levels)
mk.bounding_box(geom, 'boundary', 3)     # {'x': (0, 10), 'y': ..., 'z': ...}
```

And a few helpers outside geometry:

```python
mk.coordinate_system(geom, 'PML', selection=layer)   # e.g. perfectly matched layers
mk.set(mesh_size, hmax=0.5, hgrad=2)   # any node or Java object; converts ints and lists
```

Arguments are COMSOL property names, so the COMSOL documentation of each
feature applies. The docstrings describe each helper in detail.

[`examples/plate_with_holes.py`](https://github.com/elgar328/mphkit/blob/main/examples/plate_with_holes.py)
is a complete script, from geometry to solved results: heat conduction in
a plate with a row of cooling holes, for any number of holes.

## Limitations

- mphkit covers geometry and selections only. Physics, mesh, studies and
  results are left to MPh.
- No named helpers yet for geometry parts (`PartInstance`), sweeps, cones
  and the other remaining primitives, virtual operations or repair. They
  work through `mk.feature(geom, 'Sweep', ...)` with the same conversions
  and input handling.

## Development

```
uv run pytest
```

The tests start COMSOL and build real models. Without a CAD import
license, the CAD import test is skipped.

## License

MIT, see [LICENSE](https://github.com/elgar328/mphkit/blob/main/LICENSE).
