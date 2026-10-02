"""
Result plots of a solved model. Public as `mk.plot`.

A plot is drawn with a temporary plot group (and view, for `view=`) and
written with a temporary image export; everything is removed afterwards,
with the model history switched off meanwhile. The checks of the results
helpers apply: the dataset, a geometry changed since the solve, the step
and the unit. Where COMSOL would silently draw an empty or wrong picture
(a slice outside the geometry, an unknown colour table, `deform` without a
displacement), this raises instead.
"""
from __future__ import annotations

import math
import numbers
import os
import secrets
import stat
import tempfile
from difflib import get_close_matches
from pathlib import Path
from typing import Literal, overload

import numpy
from mph.node import Node

from . import _comsol, _datasets, _image, _results, _sweep
from ._results import One, Outer, OuterMany, OuterOne
from ._measure import bounding_box, summary

# Where the camera sits, seen from the centre, and which way is up.
VIEWS = {
    'top': ((0, 0, 1), (0, 1, 0)), 'bottom': ((0, 0, -1), (0, 1, 0)),
    'front': ((0, -1, 0), (0, 0, 1)), 'back': ((0, 1, 0), (0, 0, 1)),
    'left': ((-1, 0, 0), (0, 0, 1)), 'right': ((1, 0, 0), (0, 0, 1)),
    'iso': ((-1, -1, 1), (0, 0, 1)),
}
View = Literal['top', 'bottom', 'front', 'back', 'left', 'right', 'iso']

# The slice plane across each axis
PLANES = {'x': 'yz', 'y': 'zx', 'z': 'xy'}

# Stands for the outer value's number in file names
PLACEHOLDER = '{outer}'

# The displacement a deformed plot uses, by kind of geometry
DISPLACEMENT = {3: '(u, v, w)', 2: '(u, v)', 'axisymmetric': '(u, w)'}


@overload
def plot(geom: Node, expr: str, filename, /, selection: Node | None = None,
         *, unit: str | None = None, dataset=None, step: One = None,
         outer: OuterOne = None, x=None, y=None, z=None,
         view: View | None = None, color_range=None,
         colortable: str | None = None, deform: bool | float = False,
         size=(800, 600)) -> Path: ...
@overload
def plot(geom: Node, expr: str, filename, /, selection: Node | None = None,
         *, unit: str | None = None, dataset=None, step: One = None,
         outer: OuterMany, x=None, y=None, z=None,
         view: View | None = None, color_range=None,
         colortable: str | None = None, deform: bool | float = False,
         size=(800, 600)) -> list[Path]: ...
@overload
def plot(geom: Node, expr: str, filename, /, selection: Node | None = None,
         *, unit: str | None = None, dataset=None, step: One = None,
         outer: Outer = None, x=None, y=None, z=None,
         view: View | None = None, color_range=None,
         colortable: str | None = None, deform: bool | float = False,
         size=(800, 600)) -> Path | list[Path]: ...
def plot(geom: Node, expr: str, filename, /, selection: Node | None = None,
         *, unit: str | None = None, dataset=None, step: One = None,
         outer: Outer = None, x=None, y=None, z=None,
         view: View | None = None, color_range=None,
         colortable: str | None = None, deform: bool | float = False,
         size=(800, 600)) -> Path | list[Path]:
    """
    Saves a picture of an expression on a solved model and returns the file
    path, e.g. to look at a temperature field without the COMSOL Desktop,
    also for an AI assistant that reads images:

    ```python
    mk.plot(geom, 'T', 'T.png', unit='degC')
    mk.plot(geom, 'T', 'walls.png', walls)          # these faces only
    mk.plot(geom, 'T', 'mid.png', z=2.5)            # a slice at z = 2.5
    mk.plot(geom, 'T', 'top.png', view='top')
    mk.plot(geom, 'solid.mises', 's.png', unit='MPa', deform=True)
    ```

    Draws the expression on the surface of the geometry in 3D, on the
    domains in 2D (in an axisymmetric geometry, the r-z half section), with
    a title (expression, unit, time or frequency) and a colour legend.
    Complex values are drawn by their real part; use `'abs(p)'` for the
    magnitude. For numbers, use `mk.maximum` and the other results helpers.

    `selection` draws only part of the model, the rest as lines: in 3D a
    boundary or domain selection (the boundaries of those domains), in 2D a
    domain selection. `x`, `y` or `z` draw slices across one axis instead:
    a position or a list of positions in the geometry's length unit, not a
    range as in `sel.box` (`z=(1, 4)` draws two slices); with a domain
    selection, only inside those domains. A slice outside the extent of
    every domain (their bounding boxes) raises, as does an expression
    that exists on boundaries only (such as `ht.ntflux`), which can be
    drawn in 3D without slices.

    `view` (3D) looks from `'top'`, `'bottom'`, `'front'`, `'back'`,
    `'left'`, `'right'` or `'iso'` (isometric), without perspective;
    left out, the geometry's own view, isometric for a new model (in which
    entities hidden in that view stay hidden). The whole plot is always in
    the picture. `color_range=(min, max)` fixes the colours, in `unit` or
    SI units, e.g. to compare pictures; `colortable` names a COMSOL colour
    table such as `'HeatCamera'`. `deform=True` draws on the deformed shape
    (solid mechanics: displacements u, v, w), exaggerated by a scale COMSOL
    picks and does not show; a number sets the scale, 1 for the true
    shape. The undeformed outline stays in the picture.

    `unit`, `dataset`, `step` and `outer` work as in `mk.integral`, but a
    picture shows one step: `'first'`, `'last'`, a number or a value
    such as `{'t': 10}`; for a picture per step, loop over the values,
    `for t in mk.step_values(geom)['t']: ... step={'t': t}`. Several
    outer values (`outer='all'` or a list) give a picture each, in a list
    of paths; `{outer}` in the file name stands for the value's number,
    e.g. `mk.plot(geom, 'T', 'T_{outer}.png', outer='all', step='last')`
    (a plain string, not an f-string). The title shows the value, checked
    against the one asked for. A sweep that changes the geometry is not
    drawn: every value must have been solved on the geometry as built,
    which each call checks for all values, so `outer='all'` draws them
    faster than a loop.
    A selection set on the solution dataset in the COMSOL Desktop also
    limits the picture. `size` and the file type work as in `mk.image`.

    Pictures are drawn to temporary files next to them and replace the
    files once checked, so an error leaves existing files as they were
    (unless replacing several fails halfway). The model is left as it
    was.
    """
    name = 'plot'
    if (isinstance(expr, os.PathLike) or isinstance(filename, Node)
            or isinstance(expr, str)
            and expr.lower().endswith(tuple(_image.FORMATS))):
        raise TypeError('The file name comes third: '
                        f'{_results.CALLS[name]}.')
    _results.check_expr(name, expr)
    path = _image.picture_path(filename)
    pixels = _image.picture_size(size)
    _sweep.check_step(step, name)
    slices = _slices(x, y, z)
    if view is not None and (not isinstance(view, str) or view not in VIEWS):
        choices = ', '.join(repr(v) for v in VIEWS)
        raise ValueError(f'view must be None or one of {choices}, not '
                         f'{view!r}.')
    scale = _scale(deform)
    limits = _limits(color_range)
    if colortable is not None and not isinstance(colortable, str):
        raise TypeError(f'colortable must be a name such as "HeatCamera", '
                        f'not {colortable!r}.')
    _results.check_geometry(name, geom)
    sdim = _comsol.sdim(geom)
    if sdim == 2 and (slices is not None or view is not None):
        raise ValueError('Slices and views are for 3D geometries.')
    if slices is not None and scale is not None:
        raise ValueError('deform= draws on surfaces; leave out x/y/z.')
    _check_folder(filename)
    _sweep.check_outer(outer)
    model = geom.model.java
    with _datasets.scratch(model) as create:
        pictures, many = _sweep.pictures(
            create, geom, dataset, step, outer,
            lambda many: _check_name(path, many))
        files = _files(path, pictures)
        level, entities = None, None
        if selection is not None:
            level, _, entities = _image.drawn_selection(geom, selection)
            _check_level(selection, level, sdim, slices is not None)
        # measuring the domains goes into the history otherwise
        facts = summary(geom)
        if slices is not None:
            slices = _check_slices(geom, facts, *slices,
                                   entities if level == 3 else None)
        if view is None:
            shown = _image.geometry_view(geom)
            view_tag = 'auto' if shown is None else str(shown.tag())
        else:
            view_tag = _temporary_view(create, geom, facts, view)
        drawn: list[Path] = []
        try:
            for picture, file in zip(pictures, files):
                # the file is replaced once every picture is checked
                temporary = _temporary(file)
                drawn.append(temporary)
                group, feature = _group(create, model, geom, picture,
                                        view_tag, expr, unit, slices, level,
                                        entities)
                if scale is not None:
                    _deform(feature, scale, geom)
                if limits is not None:
                    for key, value in (('rangecoloractive', 'on'),
                                       ('rangecolormin', repr(limits[0])),
                                       ('rangecolormax', repr(limits[1]))):
                        _comsol.set_property(feature, key, value)
                if colortable is not None:
                    _comsol.set_property(feature, 'colortable',
                                         _colortable(feature, colortable))
                _draw(create, model, group, temporary, pixels, sdim, expr,
                      slices)
                if picture.title is not None:
                    _check_title(group, picture.title)
                if unit is not None and len(drawn) == 1:
                    _check_unit(create, model, geom, picture, view_tag, expr,
                                unit, slices, level, entities, sdim,
                                str(feature.getString('rangeunit')))
            for temporary, file in zip(drawn, files):
                _settle(temporary, file)
            for temporary, file in zip(drawn, files):
                os.replace(temporary, file)
        finally:
            for temporary in drawn:
                _remove(temporary)
    return files if many else files[0]


def _check_unit(create, model, geom: Node, picture, view: str, expr: str,
                unit: str, slices, level: int | None, entities: list | None,
                sdim: int, applied: str):
    """
    Raises if COMSOL ignored `unit`: `applied` is the unit it drew in,
    filled in when drawing. Draws the unit COMSOL uses without one, in a
    group of its own so that the picture keeps a single plot.
    """
    if applied == unit:
        return
    plain, plain_feature = _group(create, model, geom, picture, view, expr,
                                  None, slices, level, entities)
    with tempfile.TemporaryDirectory() as folder:
        _draw(create, model, plain, Path(folder)/'unit.png', (64, 48), sdim,
              expr, slices)
    if str(plain_feature.getString('rangeunit')) == applied:
        raise _results.unit_error(expr, applied, unit)


def _check_title(group, title):
    """
    Raises unless the drawn picture's title shows the outer value asked
    for: COMSOL fills it in when drawing.
    """
    indicator = str(group.getString('evaluatedparamindicator'))
    problem = _sweep.title_problem(indicator, title)
    if problem == 'missing':
        raise RuntimeError(
            f'COMSOL drew {title.where} without its values in the title '
            f'("{indicator}"), so which value it drew cannot be checked; no '
            'file was written. Please report it, with this message, at '
            'https://github.com/elgar328/mphkit/issues.')
    if problem == 'wrong':
        raise RuntimeError(f'COMSOL drew another value than {title.where}: '
                           f'its title says "{indicator}".')


def _check_folder(filename):
    """Checks that `{outer}` is not in the folder of the file name."""
    folder = os.path.dirname(os.fspath(filename))
    if PLACEHOLDER in folder:
        raise ValueError(f'{PLACEHOLDER} goes in the file name, not in the '
                         f'folder: "{filename}".')


def _check_name(path: Path, many: bool):
    """
    Checks that the file name has `{outer}` for several pictures of a
    sweep; called once the dataset is known to be one.
    """
    if many and PLACEHOLDER not in path.name:
        example = path.with_name(f'{path.stem}_{PLACEHOLDER}{path.suffix}')
        raise ValueError(f'mk.plot draws a picture per outer value; put '
                         f'{PLACEHOLDER} in the file name, e.g. '
                         f'{str(example)!r} (a plain string, not an '
                         'f-string).')


def _files(path: Path, pictures: list) -> list[Path]:
    """
    Returns the file of each picture: `{outer}` in the file name stands
    for the outer value's number.
    """
    if PLACEHOLDER not in path.name:
        return [path]
    if pictures[0].number is None:
        raise ValueError(f'"{path.name}" has {PLACEHOLDER} for the outer '
                         f'value, but dataset '
                         f'{_datasets.describe(pictures[0].data)} has no '
                         'outer sweep.')
    return [path.with_name(path.name.replace(PLACEHOLDER,
                                             str(picture.number)))
            for picture in pictures]


def _settle(temporary: Path, file: Path):
    """
    Checks that COMSOL drew into `temporary`, and gives it the permissions
    of the `file` it replaces, if any (on POSIX systems; Windows has only
    a read-only flag, which would keep it from replacing the file).
    """
    if not temporary.stat().st_size:
        raise OSError(f'COMSOL wrote nothing to the picture "{file}".')
    if os.name == 'nt':
        return
    try:
        mode = file.stat().st_mode & 0o777
    except OSError:
        return
    os.chmod(temporary, mode)


def _temporary(file: Path) -> Path:
    """
    Creates an empty file next to `file`, of the same type, to draw into
    before replacing `file`; makes the folder if needed. It gets the
    permissions of a new file.
    """
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            name = file.with_name(f'.{file.stem}.{secrets.token_hex(4)}'
                                  f'{file.suffix}')
            try:
                os.close(os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                                 0o666))
                return name
            except FileExistsError:
                continue
        raise FileExistsError(f'no free temporary name next to "{file}"')
    except OSError as error:
        raise OSError(f'Could not write the picture "{file}": '
                      f'{error.strerror or error}') from error


def _remove(temporary: Path):
    """Removes a temporary picture, also a read-only one; errors pass."""
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        try:
            os.chmod(temporary, stat.S_IWRITE)
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _group(create, model, geom: Node, picture, view: str, expr: str,
           unit: str | None, slices, level: int | None,
           entities: list | None):
    """
    Creates a plot group with one surface or slice plot of `picture` and
    returns both. The dataset is set first: a new plot group uses the
    first dataset of the model, whatever its geometry, and a Deform added
    later takes its displacement from the dataset set by then. Of a sweep,
    the outer value comes next: it fills in the levels of the loop, of
    which only the first, the step, is changed.
    """
    sdim = _comsol.sdim(geom)
    group = create(model.result(), f'PlotGroup{sdim}D')
    _comsol.set_property(group, 'data', str(picture.data.tag()))
    if picture.outer is not None:
        _comsol.set_property(group, 'outersolnum', str(picture.outer))
    if picture.step is not None:
        levels = ([str(level) for level in group.getStringArray('looplevel')]
                  if picture.outer is not None else [''])
        levels[0] = str(picture.step)
        _comsol.set_property(group, 'looplevel', levels)
    for key, value in (('view', view), ('edges', 'on'),
                       ('titletype', 'auto')):
        _comsol.set_property(group, key, value)
    if slices is None:
        feature = group.create('plot1', 'Surface')
    else:
        axis, positions = slices
        feature = group.create('plot1', 'Slice')
        for key, value in (('quickplane', PLANES[axis]),
                           (f'quick{axis}method', 'coord'),
                           (f'quick{axis}',
                            ' '.join(repr(p) for p in positions))):
            _comsol.set_property(feature, key, value)
    _comsol.set_property(feature, 'expr', expr)
    if unit is not None:
        _comsol.set_property(feature, 'unit', unit)
    if entities is not None and level is not None:
        chosen = feature.feature().create('sel1', 'Selection')
        chosen.selection().geom(geom.tag(), level)
        chosen.selection().set(entities)
    return group, feature


def _draw(create, model, group, path: Path, pixels, sdim: int, expr: str,
          slices):
    """
    Writes a plot group to a file, explaining COMSOL's errors. Exporting
    draws the group; running it first (`group.run()`) would draw it in a
    window of the COMSOL server too, which shows on Windows. COMSOL fills
    in the unit it applied (`rangeunit`) either way.
    """
    try:
        _image.export(create, model, group, path, pixels, sdim)
    except OSError as failure:
        error = failure.__cause__
        if not isinstance(error, Exception) or not _image.drawing_failed(error):
            raise
        if slices is not None and 'Undefined variable' in \
                _comsol.reason(error):
            raise RuntimeError(
                f'{_results.failed(expr, error)} If "{expr}" exists on '
                'boundaries only, as ht.ntflux does, plot it without '
                'x/y/z.') from error
        raise _results.failed(expr, error) from error


def _deform(feature, scale: float | str, geom: Node):
    """Draws `feature` on the deformed shape, checking the displacement."""
    deformed = feature.feature().create('def1', 'Deform')
    if not any(str(e) for e in deformed.getStringArray('expr')):
        sdim = _comsol.sdim(geom)
        kind = ('axisymmetric' if sdim == 2 and _results.axisymmetric(geom)
                else sdim)
        raise ValueError(f'deform=True needs a displacement field '
                         f'{DISPLACEMENT[kind]}, e.g. from solid '
                         'mechanics.')
    if scale != 'auto':
        _comsol.set_property(deformed, 'scaleactive', True)
        _comsol.set_property(deformed, 'scale', repr(scale))


def _temporary_view(create, geom: Node, facts: dict, name: str) -> str:
    """
    Creates a view that looks at the geometry from direction `name`,
    without perspective, and returns its tag. `facts` is the geometry's
    `summary()`.
    """
    box = facts['bounding_box']
    centre = [(low + high) / 2 for low, high in box.values()]
    diagonal = math.dist(*zip(*box.values()))
    direction, up = VIEWS[name]
    norm = math.hypot(*direction)
    position = [c + 3 * diagonal * d / norm
                for c, d in zip(centre, direction)]
    view = create(_comsol.component_of(geom).view(), geom.tag())
    camera = view.camera()
    for key, value in (('position', [repr(p) for p in position]),
                       ('target', [repr(c) for c in centre]),
                       ('up', [str(u) for u in up]),
                       ('projection', 'orthographic')):
        _comsol.set_property(camera, key, value)
    return str(view.tag())


############
# Checking #
############

def _slices(x, y, z) -> tuple[str, list[float]] | None:
    """Returns the axis and the positions of slices, or `None`."""
    given = {axis: value for axis, value in (('x', x), ('y', y), ('z', z))
             if value is not None}
    if not given:
        return None
    if len(given) > 1:
        raise ValueError('Give slice positions along one axis (x, y or z) '
                         'per picture.')
    [(axis, value)] = given.items()
    wrong = (f"{axis} must be a number or a list of numbers in the "
             f"geometry's length unit, not {value!r}.")
    try:
        dimension = numpy.ndim(value)
    except ValueError:                        # ragged lists
        raise TypeError(wrong) from None
    if isinstance(value, str) or dimension > 1:
        raise TypeError(wrong)
    items = list(value) if dimension == 1 else [value]
    if not items:
        raise ValueError(wrong)
    for item in items:
        if (isinstance(item, (bool, numpy.bool_))
                or not isinstance(item, numbers.Real)):
            raise TypeError(wrong)
        if not math.isfinite(float(item)):
            raise ValueError(wrong)
    return axis, [float(item) for item in items]


def _check_slices(geom: Node, facts: dict, axis: str, positions: list[float],
                  domains: list | None) -> tuple[str, list[float]]:
    """
    Checks that each slice crosses the extent of a domain (one of `domains`
    if given): COMSOL draws nothing, without an error, elsewhere. Positions
    within rounding of the geometry's ends are moved onto them. `facts` is
    the geometry's `summary()`.
    """
    box = facts['bounding_box']
    low, high = box[axis]
    diagonal = math.dist(*zip(*box.values()))
    margin = 1e-9 * diagonal
    # bounding_box of domains is single precision
    tolerance = 1e-6 * max(diagonal, abs(low), abs(high))
    count = facts['domains']
    ranges = []
    for number in domains or range(1, count + 1):
        extent = bounding_box(geom, 'domain', number)
        if extent is not None:
            ranges.append(extent[axis])
    checked = []
    for position in positions:
        if not low - margin <= position <= high + margin:
            raise ValueError(f'{axis}={position:g} is outside the geometry '
                             f'({axis} from {low:g} to {high:g}).')
        position = min(max(position, low), high)
        if not any(start - tolerance <= position <= end + tolerance
                   for start, end in ranges):
            where = ' of the selection' if domains else ''
            raise ValueError(f'{axis}={position:g} passes through no '
                             f'domain{where}.')
        checked.append(position)
    return axis, checked


def _check_level(selection: Node, level: int, sdim: int, slices: bool):
    """Raises for a selection whose level the plot cannot draw."""
    allowed: tuple[int, ...]
    if slices:
        allowed, text = (3,), 'Slices take a domain selection'
    elif sdim == 3:
        allowed = (3, 2)
        text = 'A surface plot in 3D takes a boundary or domain selection'
    else:
        allowed, text = (2,), 'A plot in 2D takes a domain selection'
    if level not in allowed:
        kind = _comsol.entity_level_name(level, sdim) or level
        raise ValueError(f'{text}; "{selection}" selects {kind} entities.')


def _scale(deform) -> float | str | None:
    """Returns `'auto'`, a scale factor, or `None` for no deformation."""
    if deform is False or deform is None:
        return None
    if deform is True:
        return 'auto'
    wrong = (f'deform must be True, False or a positive scale factor, not '
             f'{deform!r}.')
    if (isinstance(deform, (bool, numpy.bool_))
            or not isinstance(deform, numbers.Real)):
        raise TypeError(wrong)
    scale = float(deform)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(wrong)
    return scale


def _limits(color_range) -> tuple[float, float] | None:
    """Returns the colour range as two floats, checked."""
    if color_range is None:
        return None
    wrong = (f'color_range must be (min, max), two finite numbers with '
             f'min < max, not {color_range!r}.')
    try:
        dimension = numpy.ndim(color_range)
    except ValueError:                        # ragged lists
        raise ValueError(wrong) from None
    if isinstance(color_range, str) or dimension != 1 \
            or len(color_range) != 2:
        raise ValueError(wrong)
    if any(isinstance(v, (bool, numpy.bool_))
           or not isinstance(v, numbers.Real) for v in color_range):
        raise ValueError(wrong)
    low, high = (float(v) for v in color_range)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(wrong)
    return low, high


def _colortable(feature, name: str) -> str:
    """
    Returns the colour table `name` as COMSOL spells it. COMSOL accepts
    any name and silently draws with its default, so unknown names raise.
    """
    known = [str(v) for v in feature.getAllowedPropertyValues('colortable')]
    if name in known:
        return name
    lower = {k.lower(): k for k in known}
    close = get_close_matches(name.lower(), list(lower), n=3, cutoff=0.6)
    hint = (f'did you mean {" or ".join(repr(lower[c]) for c in close)}?'
            if close else "e.g. 'Rainbow', 'HeatCamera' or 'GrayScale'.")
    raise ValueError(f'No colortable {name!r}; {hint}')
