# Changelog

All notable changes to mphkit are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html);
until 1.0, a minor release (0.2, 0.3, ...) may change the API. See
[docs/RELEASING.md](docs/RELEASING.md) for how entries are written.

## [Unreleased]

### Added

- Unknown names on `mphkit` and `mphkit.sel` raise an `AttributeError`
  that names the helper probably meant, e.g. `mk.box` suggests
  `mphkit.sel.box` to select or `mphkit.block` to create, and points to
  `help(mphkit)`.
- Unknown entity kinds suggest COMSOL's name (`'face'` → `'boundary'` in
  3D), and unknown property names suggest COMSOL's (`radius` → `r`).

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

[Unreleased]: https://github.com/elgar328/mphkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/elgar328/mphkit/releases/tag/v0.1.0
