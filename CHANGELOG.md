# Changelog

All notable changes to mphkit are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html);
until 1.0, a minor release (0.2, 0.3, ...) may change the API. See
[docs/RELEASING.md](docs/RELEASING.md) for how entries are written.

## [Unreleased]

### Added

- `mk.integral`, `mk.average`, `mk.maximum`, `mk.minimum` and `mk.value`
  read results of a solved model over entities, selections or points.
  They raise in most cases where COMSOL would silently give a wrong
  number (a unit that does not fit, several solutions, a geometry changed
  since the solve, a point outside the geometry) and leave nothing in the
  model.
- Unknown names such as `mk.volume_integral`, `mk.mphint2` or `mk.probe`
  suggest the results helpers; `mk.evaluate` points to MPh's
  `model.evaluate`.
- `mk.plot` saves a picture of an expression on a solved model: on the
  surface, on a selection, on slices (`x=`, `y=`, `z=`), seen from a side
  (`view='top'`, ...) or on the deformed shape (`deform=True`), with the
  checks of the results helpers, and leaves nothing in the model.
- `mk.image(geom, filename, mesh=True)` saves a picture of the mesh,
  coloured by element quality, also of a selection only.
- Unknown names such as `mk.mphplot`, `mk.slice` or `mk.surface_plot`
  suggest `mk.plot`, and `mk.mphmesh` or `mk.mesh_plot` suggest
  `mk.image`.
- `mk.physics_types`, `mk.feature_types`, `mk.properties` and
  `mk.variables` look up COMSOL's names instead of guessing them: physics
  interfaces for a geometry, the features of a physics interface with the
  levels they go on (also the geometry, mesh and study types), the
  properties of a node or feature type with their descriptions, defaults
  and choices, and the variables for result
  expressions, each with `search=`. They read the undocumented
  code-completion data of the installed COMSOL, raise if it is missing or
  changed, and leave nothing in the model. Unknown names such as
  `mk.list_features`, `mk.property_values` or `mk.vars` suggest them.

- `mk.materials` lists the materials in COMSOL's material libraries
  (names, property groups and basic properties, with `search=`), and
  `mk.material` inserts one into the component of a geometry, on all
  domains or a selection, instead of typing property values by hand.
  Unknown material names suggest close ones or the library to pass.
  Unknown names such as `mk.list_materials` or `mk.matlib` suggest them.
  The example script takes its structural steel from the library.

### Fixed

- `mk.measure`, `mk.bounding_box` and `mk.sel.find` no longer leave lines
  in the model's history, which showed up in a Java export of the model.

## [0.2.0] - 2026-09-29

### Added

- Unknown names on `mphkit` and `mphkit.sel` raise an `AttributeError`
  that suggests the helper that was probably meant: `mk.box` suggests
  `mphkit.sel.box` to select or `mphkit.block` to create, and `mk.cone`
  suggests `mphkit.feature(geom, 'Cone', ...)`. The message also points
  to `help(mphkit)` or `help(mphkit.sel)`.
- Unknown entity kinds, including entity levels given as numbers, suggest
  COMSOL's name (in 3D, `'face'` → `'boundary'` and `2` → `'boundary'`),
  and unknown property names suggest COMSOL's (`radius` → `r`, or `rmaj`
  and `rmin` on a torus).
- `mk.summary` gives the entity counts, voids, bounding box and length
  unit of a geometry, `mk.sel.neighbors` returns the entities adjacent to
  others (e.g. the domains on either side of a boundary) and
  `mk.coordinates` the vertices of entities; they leave nothing in the
  model.
- `mk.image` saves a picture of a geometry, or of a selection highlighted
  on it, optionally with entity numbers, and leaves the model as it was.

### Changed

- Using a node that no longer exists in the model raises a `LookupError`
  naming the node, instead of an unrelated `AttributeError` or
  `TypeError`.
- `sel.box` and `sel.find` raise a `ValueError` for a reversed range such
  as `x=(14.1, 1.1)`, for which COMSOL silently selected what lies outside
  it. Give `(min, max)`, or use `sel.complement` to select the outside.

### Fixed

- Type hints of optional arguments accept `None`, so type checkers accept
  calls that pass the defaults explicitly.
- Integers outside the 32-bit range, such as `2**31` or `10**400`, are
  passed to COMSOL as they are, instead of raising `OverflowError` in
  vectors or silently wrapping around otherwise (`2**32 + 1` became 1).
- `sel.box` and `sel.find` find faces, edges and domains at coordinates
  such as 0.3 or 1.1 that single precision cannot represent exactly, also
  when given as expressions (`z='L'`). COMSOL compares them in single
  precision, so these were missed; each bound now gets a margin of a
  millionth of its value. With `condition='intersects'`, entities that
  only touch such a bound are now picked too, as they already were at
  round coordinates; pass `xmin=`, `xmax=`, ... to `sel.box` for exact
  bounds.
- `sel.ball`, `sel.disk` and `sel.cylinder` find a drawn sphere, circle
  or cylinder given its exact radius, top and bottom, also for values
  such as 0.3 and for shapes far from the origin; COMSOL compares in
  single precision, so these were missed. `r`, `top`, `bottom` and `rin`
  now get a margin of a millionth of the size of the coordinates
  involved. So with `condition='intersects'`, entities that only touch
  them are picked too; for a strict bound, give room the other way, e.g.
  `0.999*r`.

## [0.1.0] - 2026-09-27

First release.

### Added

- Geometry helpers create features and return MPh nodes: `geometry`,
  `block`, `cylinder`, `sphere`, `point`, `workplane`, `square`,
  `rectangle`, `circle`, `polygon`, `line_segment`, `interval`, `extrude`,
  `revolve` and `import_`, and `feature` creates any other geometry
  feature. `component_of` returns the component a geometry belongs to.
- Operations combine and transform objects: `union`, `difference`,
  `intersection`, `delete`, `partition`, `fillet`, `chamfer`, `array`,
  `move`, `rotate`, `mirror` and `rigid_transform`.
- `mk.sel` selects entities by location instead of by number, for the
  whole model or inside a geometry sequence or work plane (`where=`):
  `box`, `ball`, `cylinder`, `disk`, `all`, `union`, `intersection`,
  `difference`, `complement`, `adjacent`, `result`, `layer` and
  `cumulative`.
- `sel.entities`, `sel.find`, `measure` and `bounding_box` query the
  geometry and leave nothing in the model.
- `coordinate_system` creates coordinate systems such as perfectly matched
  layers, and `set` sets properties of any node or Java object.
- `LicenseError` is raised when a feature needs a license that is not
  available.

[Unreleased]: https://github.com/elgar328/mphkit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/elgar328/mphkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/elgar328/mphkit/releases/tag/v0.1.0
