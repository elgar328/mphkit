# Changelog

All notable changes to mphkit are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html);
until 1.0, a minor release (0.2, 0.3, ...) may change the API. See
[docs/RELEASING.md](docs/RELEASING.md) for how entries are written.

## [Unreleased]

### Added

- `mk.describe(model)` returns a model's settings as plain, JSON-ready
  values in which nodes can be matched by type and place rather than by
  tags, feature order or entity numbers:
  parameters, definitions, probes, materials, physics, multiphysics,
  meshes, mass properties and studies, with only the properties that differ from the
  defaults (leaving out those other settings make unused) and their SI
  values where COMSOL can evaluate them, the names expressions call
  functions, operators and probes by, and the selections described by
  the location and size of their entities (with the geometry's length
  in metres and whether it forms a union or an assembly);
  with `solver=True` also the solver settings that differ from the ones
  COMSOL would choose. It is meant for checking a model rebuilt by a
  script against the one made in the COMSOL Desktop. It leaves the model
  as it was; with `solver=True` it compiles the equations, which in a
  model saved by another COMSOL version or build may update its solver
  sequences and build empty meshes of layered materials (listed in the
  result's `notes`).
- `mk.compare(a, b)` lists how two models (or results of `mk.describe`)
  differ: parameters, geometry, and nodes paired by name or by type and
  the place of their selections rather than by tags or order, with
  values compared after translating tags and in SI units. Each item
  has a kind, both paths and a one-line message; what follows from a
  differing geometry is grouped under it.

### Fixed

- `mk.problem_size` leaves the model as it was. It changed a note the
  COMSOL Desktop keeps in a solver sequence not solved yet, left nodes of
  COMSOL's own (derived variables `iexpr1`, ..., operators `maxOp1`, ...)
  in a model not solved since it was made or loaded, and the first time
  also set the study steps' values of variables not solved for from `1`
  to `auto`. In a model saved by another COMSOL version or build,
  compiling may still update its solver sequences, as solving would; a
  warning then says what changed.

## [0.3.0] - 2026-10-04

### Added

- `mk.geometry(model, 2, axisymmetric=True)` makes a 2D axisymmetric
  geometry: the r-z half plane of a body of revolution.
- `mk.materials` lists the materials in COMSOL's material libraries
  (names, property groups and basic properties, with `search=`), and
  `mk.material` inserts one into the component of a geometry, on all
  domains or a selection, instead of typing property values by hand.
  Unknown material names suggest close ones or the library to pass.
- `mk.physics_types`, `mk.feature_types`, `mk.properties` and
  `mk.variables` look up COMSOL's names instead of guessing them: the
  physics interfaces for a geometry, the features of a physics interface
  with the levels they go on (also geometry, mesh and study types), the
  properties of a node or feature type with descriptions, defaults and
  choices, and the variables for result expressions, each with
  `search=`. They read the installed COMSOL's code-completion data,
  which is undocumented, and raise if it is missing or changed.
- `mk.check(model)` lists, before solving, what COMSOL would get wrong
  silently or vaguely: expressions in the wrong unit, domains without a
  material, conditions that select nothing or apply nowhere, a component
  without a mesh, physics that no study step solves, and (as info)
  boundaries left at the default condition.
- `mk.mesh_quality(geom)` gives a mesh's quality in numbers, for all or
  some domains, or boundaries in 3D: the lowest and mean quality in any
  of COMSOL's six measures (`quality=`), a histogram, the worst elements
  and where they are, the values per entity, the element sizes, the
  entities left without elements and what COMSOL reported when building
  the mesh. It matches COMSOL's mesh statistics for all element types.
- `mk.problem_size(model)` gives, before solving, the degrees of freedom
  of each study step, the solver COMSOL would use, the mesh elements and
  the computer's memory (macOS and Windows) and cores.
- `mk.log_progress(path)` makes COMSOL write its progress log, and
  `mk.progress(path)` reads it from another process (percent, task,
  memory, degrees of freedom, sweep parameter, time steps, last lines),
  with the state, CPU and memory of the solving processes (macOS and
  Windows), for long solves run in the background; `help(mk.progress)`
  shows how to start and stop one.
- `mk.integral`, `mk.average`, `mk.maximum`, `mk.minimum` and `mk.value`
  read the results of a solved model over entities, selections or
  points. They raise in most cases where COMSOL would silently give a
  wrong number (a unit that does not fit, several solutions, a geometry
  changed since the solve, a point outside the geometry). `dataset=` also
  takes the study that made it. They refuse the results of a study whose
  last solve failed, and the copy COMSOL makes of a parametric sweep's
  last value, pointing to the sweep's own dataset instead.
- Parametric sweeps that COMSOL stores as an outer loop (around a
  time-dependent or eigenvalue study or a list of frequencies, or over
  the geometry, mesh, materials or functions) are read with `outer=`, in
  the forms of `step=` or by value (`{'Th': '200[degC]'}`, or a number
  in SI units), each value from its own solution. `mk.outer_values` gives
  their parameter values and `mk.step_values` the times, frequencies or
  parameter values of the steps by name, both in SI units; `step=` also
  picks a step by value, e.g. `{'t': 10}`, in each outer value.
- A number given as `step=` or `outer=` is a position. One that is also
  the value of another position, such as `step=10` on times 0, 1, ..., 10
  (the tenth step is t = 9), gives an `mk.StepWarning`, a `UserWarning`:
  pick by value to mean the time, `step={'t': 10}`, or turn the warning
  off with `warnings.filterwarnings('ignore', category=mk.StepWarning)`.
  For the warning, numbers given as `outer=` are compared with the values
  as swept, e.g. Th in degC.
- A sweep that changes the geometry is read over selection nodes or all
  entities, evaluated on each value's geometry; entity numbers, explicit
  selections and selections that are empty for a value raise. `mk.plot`
  does not draw it.
- `mk.plot` saves a picture of an expression on a solved model: on the
  surface, on a selection, on slices (`x=`, `y=`, `z=`), seen from a side
  (`view='top'`, ...) or on the deformed shape (`deform=True`), in chosen
  colours (`color_range=`, `color_table=`), with the checks of the
  results helpers. With `outer=` it draws values of a parametric sweep
  stored as an outer loop, one picture each (`'T_{outer}.png'`), and
  checks that the title shows the value asked for. Numbers given as
  `step=` or `outer=` warn as in the results helpers. An error leaves
  existing files as they were (unless replacing several fails halfway)
  and removes the folders made for the pictures.
- `mk.image(geom, filename, mesh=True)` saves a picture of the mesh,
  coloured by element quality, also of a selection only.
- The new helpers leave nothing in the model.
- Unknown names suggest the new helpers: `mk.volume_integral`,
  `mk.mphint2`, `mk.probe` → the results helpers; `mk.time_values`,
  `mk.sweep_values` → `mk.step_values`, `mk.outer_values`; `mk.mphplot`,
  `mk.slice` → `mk.plot`; `mk.mesh_plot` → `mk.image`;
  `mk.list_features`, `mk.vars` → the name lookups; `mk.matlib` →
  `mk.materials`; `mk.lint`, `mk.validate` → `mk.check`; `mk.dofs`,
  `mk.stop_solve` → `mk.problem_size`, `mk.progress`; `mk.mesh_stats` →
  `mk.mesh_quality`; `mk.axisymmetric` → `mk.geometry`. Others say how
  to do it with plain MPh: `mk.evaluate` (MPh's `model.evaluate`),
  `mk.sweep`, `mk.parametric_sweep` and `mk.solve` (making a sweep or
  solving), `mk.load`, `mk.import_model` (opening an existing model) and
  `mk.java` (what `node.java` is).

### Changed

- `mk.geometry` raises a `TypeError` for a `dim` that is not an integer
  (e.g. `'2D'` or `2.0`) and a `ValueError` for one other than 1, 2 or 3,
  before creating anything, instead of COMSOL's error and an empty
  component left in the model.
- mphkit requires MPh below 2 (`mph>=1.4,<2`): it uses some of MPh's
  internals, which a major release may change.
- The package overview, `print(mphkit.__doc__)`, is half as long: one
  runnable script, the rules and an index of every helper, with the
  details in each helper's `help()`. Errors for unknown names end with
  "print(mphkit.__doc__) lists all helpers." (on `mk.sel` with
  "..., mphkit.sel's too.") instead of pointing to `help(mphkit)`.
- `sel.union`, `sel.intersection`, `sel.difference`, `sel.complement` and
  `sel.adjacent` raise a `TypeError` that says to select by location when
  given entity numbers, instead of "'int' object is not iterable", a
  `ValueError` about the property `input` or COMSOL's "Unknown
  selection". `sel.entities`, `mk.image` and `mk.coordinate_system` say
  that a number given as a selection is an entity number, and a string
  such as `'boundary'` or a selection's name gets what to pass instead.
  Geometry operations (`mk.fillet`, `mk.chamfer`, `mk.delete`, the
  Boolean operations and inputs of `mk.feature`) given numbers say to
  select by location or to pass the objects, instead of "Cannot use 3 as
  a geometry input"; in a work plane, they no longer ask for a selection
  made with `where='geometry'`.
- The error for an unknown property name given to the geometry helpers,
  or to `mk.set` on a feature of a geometry, physics, mesh or study, or
  on a material, ends with the call that lists the properties, e.g.
  `mk.properties(geom, 'Block', search=...)`, also when no close name is
  suggested.

### Fixed

- `mk.measure`, `mk.bounding_box` and `mk.sel.find` no longer leave lines
  in the model's history, which showed up in a Java export of the model.
- An entity number given as a 0-d numpy array, e.g. `numpy.array(2)`, is
  taken as a selection by `mk.measure`, `mk.bounding_box`,
  `mk.coordinates` and `mk.sel.neighbors`, instead of raising "iteration
  over a 0-d array".
- `mk.image` takes numpy integers in `size`, e.g. `numpy.int64(800)`.

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

[Unreleased]: https://github.com/elgar328/mphkit/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/elgar328/mphkit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/elgar328/mphkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/elgar328/mphkit/releases/tag/v0.1.0
