"""
Results of a solved model. Public as `mk.integral`, `mk.average`,
`mk.maximum`, `mk.minimum` and `mk.value`.

They evaluate with temporary numerical features, tables and cut points
that are removed afterwards, with the model history switched off
meanwhile: the model is left as it was. Where COMSOL would silently give a
wrong number, they raise instead: COMSOL evaluates the first dataset when
there are several, the old solution after a geometry change, and ignores a
unit that does not fit the expression.
"""
from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy
from mph.node import Node
from numpy.typing import NDArray

from . import _comsol, _datasets, _sweep
from ._measure import numbers_of

# COMSOL feature types: the kind of result, then the level of the entities
KINDS = {'integral': 'Int', 'average': 'Av', 'maximum': 'Max',
         'minimum': 'Min'}
LEVELS = {3: 'Volume', 2: 'Surface', 1: 'Line'}

CALLS = {name: f'mk.{name}(geom, entity, expr, selection)' for name in KINDS}
CALLS['value'] = 'mk.value(geom, expr, points)'
CALLS['plot'] = 'mk.plot(geom, expr, filename, selection)'

# Properties the helpers set themselves, and what to use instead.
RESERVED = {'expr': 'the expression argument', 'data': 'dataset=',
            'table': 'nothing (the table is temporary)',
            'includepos': 'position= of mk.maximum or mk.minimum',
            't': 'step='}
RESERVED_PREFIXES = {
    ('innerinput', 'solnum', 'looplevel', 'interp'): 'step=',
    ('outer',): 'outer=',
    ('dataseries',): "step='all' and combine the values in Python",
}

STEPS = ('all', 'first', 'last')

Array = NDArray[Any]
One = int | numpy.integer | Literal['first', 'last'] | None
Many = Literal['all'] | Sequence[int] | NDArray[numpy.integer]
Step = (int | numpy.integer | str | Sequence[int] | NDArray[numpy.integer]
        | None)
# outer values of a sweep: the same forms
OuterOne = One
OuterMany = Many
Outer = Step


###########
# Helpers #
###########

@overload
def integral(geom: Node, entity: str, expr: str, /, selection=None, *,
             unit: str | None = None, dataset=None, step: One = None,
             outer: OuterOne = None,
             **properties: Any) -> float: ...
@overload
def integral(geom: Node, entity: str, expr: str, /, selection=None, *,
             unit: str | None = None, dataset=None, step: Many,
             outer: Outer = None,
             **properties: Any) -> Array: ...
@overload
def integral(geom: Node, entity: str, expr: str, /, selection=None, *,
             unit: str | None = None, dataset=None, step: Step = None,
             outer: OuterMany,
             **properties: Any) -> Array: ...
@overload
def integral(geom: Node, entity: str, expr: str, /, selection=None, *,
             unit: str | None = None, dataset=None, step: Step = None,
             outer: Outer = None,
             **properties: Any) -> float | Array: ...
def integral(geom: Node, entity: str, expr: str, /, selection=None, *,
             unit: str | None = None, dataset=None, step: Step = None,
             outer: Outer = None,
             **properties: Any) -> Any:
    """
    Returns the integral of an expression over entities of a solved model,
    e.g. the heat flowing in through a face:

    ```python
    heat = mk.integral(geom, 'boundary', 'ht.ntflux', hot, unit='W')
    ```

    `entity` and `selection` work as in `measure()`: `'domain'`,
    `'boundary'` or `'edge'`, and an entity number, a list of them, a
    selection node, or `None` for all entities of that kind. In 2D the
    integral is per unit thickness (e.g. W/m); in an axisymmetric geometry
    it is over the revolved body (pass `intvolume=False` or
    `intsurface=False` for the plain one). Signs follow COMSOL:
    `ht.ntflux` is the flux out of the domain, so heat flowing in is
    negative.

    `unit` is the unit of the result (the integral, e.g. `'W'`), checked
    against the expression: COMSOL ignores a unit that does not fit, this
    raises. Without it, values are in SI units, also in a geometry drawn
    in mm (a volume in m³, while `measure()` gives mm³).

    `dataset` is the solution to evaluate, by name as in
    `model.datasets()`, by tag or node, or the study that made it; it is
    needed when the geometry has several solutions. A study whose last
    solve failed raises, until it is solved again.

    `step` picks steps of a time-dependent study, sweep or frequency list,
    counted from 1: `'first'`, `'last'`, a number, a list of numbers or
    `'all'`; `mk.step_values(geom)` gives their times or parameter values.
    Unlike MPh's `model.evaluate`, `None` means the only step and raises
    when there are several. `outer` picks, in the same forms, values of a
    parametric sweep that COMSOL stores as an outer loop: around a
    time-dependent, frequency-domain or eigenvalue study, or over a
    geometry, mesh, material or function; `mk.outer_values(geom)` gives
    them. `step` then counts the steps of each value; several values at
    once need the same steps (when a solver picks the time steps, each
    value has its own: take `'first'`, `'last'` or one value at a time).
    `outer=1` raises for a dataset without such a sweep, unlike `step=1`
    for one with a single step.

    One step (and value) gives a float; a list or `'all'` gives an array,
    with an axis for the outer values before the one for the steps, e.g.
    shape `(3, 4)` for `outer='all', step='all'` over 3 values of 4 steps.
    Complex results (e.g. frequency domain) come back as complex numbers.

    Other keyword arguments are COMSOL properties of the numerical feature
    (e.g. `intorder`). The geometry must be built as it was solved:
    changing it after the solve raises, unless it was built and meshed
    again without solving. Changing physics, materials or parameters
    without solving again goes unnoticed too. Each value of a sweep is
    read from its own solution and checked against the geometry; sweeps
    that change the geometry, or whose mesh leaves part of it out, raise
    for now. An empty selection raises, while `measure()` gives 0.
    """
    return _over('integral', geom, entity, expr, selection, unit, dataset,
                 step, outer, False, properties)


@overload
def average(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: One = None,
            outer: OuterOne = None,
            **properties: Any) -> float: ...
@overload
def average(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Many,
            outer: Outer = None,
            **properties: Any) -> Array: ...
@overload
def average(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: OuterMany,
            **properties: Any) -> Array: ...
@overload
def average(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: Outer = None,
            **properties: Any) -> float | Array: ...
def average(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: Outer = None,
            **properties: Any) -> Any:
    """
    Returns the average of an expression over entities of a solved model,
    e.g. the mean temperature of a face:

    ```python
    mk.average(geom, 'boundary', 'T', top, unit='degC')
    ```

    The average weighs by volume, area or length (in an axisymmetric
    geometry by the revolved one). Arguments and results work as in
    `integral()`; `unit` is the unit of the expression.
    """
    return _over('average', geom, entity, expr, selection, unit, dataset,
                 step, outer, False, properties)


@overload
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: One = None,
            outer: OuterOne = None,
            position: Literal[False] = False,
            **properties: Any) -> float: ...
@overload
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Many,
            outer: Outer = None,
            position: Literal[False] = False,
            **properties: Any) -> Array: ...
@overload
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: OuterMany,
            position: Literal[False] = False,
            **properties: Any) -> Array: ...
@overload
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: One = None,
            outer: OuterOne = None,
            position: Literal[True],
            **properties: Any) -> tuple[float, Array]: ...
@overload
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Many,
            outer: Outer = None,
            position: Literal[True],
            **properties: Any) -> tuple[Array, Array]: ...
@overload
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: OuterMany,
            position: Literal[True],
            **properties: Any) -> tuple[Array, Array]: ...
@overload
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: Outer = None,
            position: bool = False,
            **properties: Any) -> Any: ...
def maximum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: Outer = None,
            position: bool = False,
            **properties: Any) -> Any:
    """
    Returns the maximum of an expression over entities of a solved model,
    e.g. the hottest temperature in all domains:

    ```python
    mk.maximum(geom, 'domain', 'T', unit='degC')
    value, (x, y, z) = mk.maximum(geom, 'domain', 'T', position=True)
    ```

    COMSOL takes it at the mesh's evaluation points, so it is close to,
    not exactly, the true maximum. `position=True` also returns where it
    is, in the geometry's length unit; with several steps, an array of
    values and one row of coordinates per step (with several outer values
    too, shape `(values, steps, 3)` in 3D). For a complex expression
    the maximum of the real part; use `'abs(p)'` for the largest
    magnitude. Other arguments work as in `integral()`; `unit` is the unit
    of the expression.
    """
    return _over('maximum', geom, entity, expr, selection, unit, dataset,
                 step, outer, position, properties)


@overload
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: One = None,
            outer: OuterOne = None,
            position: Literal[False] = False,
            **properties: Any) -> float: ...
@overload
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Many,
            outer: Outer = None,
            position: Literal[False] = False,
            **properties: Any) -> Array: ...
@overload
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: OuterMany,
            position: Literal[False] = False,
            **properties: Any) -> Array: ...
@overload
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: One = None,
            outer: OuterOne = None,
            position: Literal[True],
            **properties: Any) -> tuple[float, Array]: ...
@overload
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Many,
            outer: Outer = None,
            position: Literal[True],
            **properties: Any) -> tuple[Array, Array]: ...
@overload
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: OuterMany,
            position: Literal[True],
            **properties: Any) -> tuple[Array, Array]: ...
@overload
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: Outer = None,
            position: bool = False,
            **properties: Any) -> Any: ...
def minimum(geom: Node, entity: str, expr: str, /, selection=None, *,
            unit: str | None = None, dataset=None, step: Step = None,
            outer: Outer = None,
            position: bool = False,
            **properties: Any) -> Any:
    """
    Returns the minimum of an expression over entities of a solved model,
    e.g. `mk.minimum(geom, 'domain', 'T', unit='degC')`. Works as
    `maximum()`.
    """
    return _over('minimum', geom, entity, expr, selection, unit, dataset,
                 step, outer, position, properties)


def value(geom: Node, expr: str, points, /, *, unit: str | None = None,
          dataset=None, step: Step = None, outer: Outer = None,
          outside: Literal['error', 'nan'] = 'error') -> float | Array:
    """
    Returns the values of an expression at points of a solved model:

    ```python
    mk.value(geom, 'T', (0.05, 0.025, 0.005), unit='degC')
    mk.value(geom, 'T', [(0, 0, 0), (0.1, 0, 0)], unit='degC')
    ```

    `points` is one point, `(x, y, z)` (`(x, y)` or `(r, z)` in 2D), or
    rows of them, one point per row: a list of tuples or an array of shape
    `(n, 3)`. **Coordinates by rows as in MATLAB, `[xs, ys, zs]`, are
    refused, but with as many points as coordinates the two look alike:
    give one row per point.** Coordinates are in the geometry's length
    unit. One point gives a float, several an array with one value per
    point; with several steps, one more axis for the steps.

    It evaluates domain variables, interpolated from the mesh. Variables
    that exist on boundaries only, such as `ht.ntflux`, raise: evaluate
    those with `average()` or `maximum()` over a boundary selection. A
    point outside the geometry, or where nothing was solved (no physics
    there), raises; `outside='nan'` gives nan there instead. `unit`,
    `dataset`, `step` and `outer` work as in `integral()`; the axes are
    points, outer values, steps.
    """
    name = 'value'
    check_expr(name, expr)
    if outside not in ('error', 'nan'):
        raise ValueError(f"outside must be 'error' or 'nan', not "
                         f'{outside!r}.')
    check_geometry(name, geom)
    sdim = _comsol.sdim(geom)
    coordinates, single = _points(geom, points, sdim)
    steps(step, None, '')
    steps(outer, None, '', what='outer')
    model = geom.model.java
    with _datasets.scratch(model) as create:
        request = _sweep.resolve(create, geom, dataset, step, outer)
        results: list[tuple[Array, Array]] = []
        for target in request.targets:
            found, missing = _interpolate(create, model, target, expr, unit,
                                          coordinates)
            if missing.any():
                if outside == 'error':
                    raise ValueError(_outside_message(missing, target.where))
                found[missing] = numpy.nan
            results.append((found, missing))
        if unit is not None:
            checked = next(((target, numpy.flatnonzero(~missing)[0])
                            for target, (_, missing)
                            in zip(request.targets, results)
                            if not missing.all()), None)
            if checked is None:
                raise ValueError(f'No point has a value to check the unit '
                                 f'{unit!r} at.')
            target, inside = checked
            _check_point_unit(create, model, target.data, expr, unit,
                              coordinates[inside])
    # points, outer values, steps
    every: Array = numpy.stack([found for found, _ in results],
                               axis=1)[:, request.order]
    if not request.many_step:
        every = every[:, :, 0]
    if not request.many_outer:
        every = every[:, 0]
    if single:
        every = every[0]
    return _scalar(every)


def _interpolate(create, model, target, expr: str, unit: str | None,
                 coordinates: Array) -> tuple[Array, Array]:
    """
    Returns the values of `expr` at the points on one target, a row per
    point and a column per step, and which points have no value.
    """
    unique, order = _unique(target.solnums)
    feature = create(model.result().numerical(), 'Interp')
    _comsol.set_properties(feature, {
        'data': target.data, 'expr': [expr, '1'],
        'unit': None if unit is None else [unit, '1'],
        'coorderr': 'off', 'solnum': unique})
    feature.setInterpolationCoordinates(coordinates.T.tolist())
    try:
        real = numpy.array(feature.getReal(), dtype=float)
    except Exception as error:
        if 'Undefined variable' in _comsol.reason(error):
            raise RuntimeError(
                f'{failed(expr, error)} value() evaluates domain '
                f'variables; if "{expr}" exists on boundaries only, as '
                'ht.ntflux does, use mk.average or mk.maximum over a '
                'boundary selection.') from error
        raise failed(expr, error) from error
    count = len(coordinates)
    shape = (count, 2 * len(unique))
    if real.shape != shape:
        raise RuntimeError(f'COMSOL returned values of shape '
                           f'{real.shape}, expected {shape}.')
    # a row per point: for each step, the expression and '1'
    real = real.reshape(count, len(unique), 2)[:, order]
    found: Array = real[:, :, 0]
    if feature.isComplex():
        imag = numpy.array(feature.getImag(), dtype=float)
        found = found + 1j*imag.reshape(count, len(unique), 2)[:, order, 0]
    # '1' is nan only outside the geometry and where nothing was solved
    return found, numpy.isnan(real[:, :, 1]).any(axis=1)


#################
# Over entities #
#################

def _over(name: str, geom: Node, entity: str, expr: str, selection,
          unit: str | None, dataset, step, outer, position: bool,
          properties: dict) -> Any:
    """Evaluates integrals, averages, maxima and minima."""
    check_expr(name, expr)
    _check_reserved(name, properties)
    check_geometry(name, geom)
    level = _comsol.entity_dim(geom, entity)
    if level == 0:
        raise ValueError(f'{name}() works on domains, boundaries and edges; '
                         'use mk.value(geom, expr, points) for values at '
                         'points.')
    steps(step, None, '')
    steps(outer, None, '', what='outer')
    model = geom.model.java
    sdim = _comsol.sdim(geom)
    settings: dict[str, Any] = {'expr': expr, 'unit': unit,
                                'innerinput': 'manual'}
    if position:
        settings['includepos'] = True
    if name in ('integral', 'average') and axisymmetric(geom):
        settings['intvolume' if level == 2 else 'intsurface'] = True
    ftype = KINDS[name] + LEVELS[level]
    column = -(sdim + 1) if position else -1
    with _datasets.scratch(model) as create:
        request = _sweep.resolve(create, geom, dataset, step, outer)
        found = numbers_of(geom, entity, selection)
        if not found:
            raise ValueError(f'Selection "{selection}" is empty.'
                             if isinstance(selection, Node)
                             else 'The selection is empty.')

        def evaluate(target, settings: dict
                     ) -> tuple[str, Array, Array | None]:
            unique, order = _unique(target.solnums)
            feature = create(model.result().numerical(), ftype)
            table = create(model.result().table(), 'Table')
            feature.set('data', target.data)
            feature.selection().geom(geom.tag(), level)
            feature.selection().set(found)
            # defaults first, so that `properties` can override them
            _comsol.set_properties(feature, {**settings, 'solnum': unique,
                                             **properties})
            feature.set('table', str(table.tag()))
            try:
                feature.setResult()
            except Exception as error:
                if 'not meshed' in _comsol.reason(error):
                    at = f' at {target.where}' if target.where else ''
                    raise RuntimeError(
                        f'Some of these entities have no solution{at} (no '
                        'physics or mesh there); pass a selection of the '
                        'solved ones.') from error
                raise failed(expr, error) from error
            real = numpy.array(table.getReal(), dtype=float)
            imag = (numpy.array(table.getImag(), dtype=float)
                    if table.isComplex() else None)
            if real.ndim != 2 or len(real) != len(unique):
                raise RuntimeError(f'COMSOL returned {len(real)} rows, '
                                   f'expected {len(unique)}.')
            headers = [str(h) for h in table.getColumnHeaders()]
            return headers[column], real[order], \
                None if imag is None else imag[order]

        results: list[tuple[Array, Array | None]] = []
        for target in request.targets:
            header, real, imag = evaluate(target, settings)
            if (not results and unit is not None
                    and not header.endswith(f'({unit})')):
                plain, *_ = evaluate(target, {**settings, 'unit': None})
                _check_unit(expr, unit, header, plain)
            results.append((real, imag))
    # outer values, steps
    values: Array = numpy.stack([
        real[:, column] if imag is None
        else real[:, column] + 1j*imag[:, column]
        for real, imag in results])[request.order]
    where: Array = numpy.stack([real[:, -sdim:]
                                for real, _ in results])[request.order]
    if not request.many_step:
        values, where = values[:, 0], where[:, 0]
    if not request.many_outer:
        values, where = values[0], where[0]
    if not position:
        return _scalar(values)
    return _scalar(values), where


############
# Checking #
############

def check_expr(name: str, expr):
    """Raises for anything but one expression, e.g. swapped arguments."""
    call = CALLS[name]
    if isinstance(expr, str):
        if expr in _comsol.ENTITIES:
            raise ValueError(f'{expr!r} is an entity kind; the order is '
                             f'{call}.')
        return
    if isinstance(expr, Node):
        raise TypeError(f'The expression must be a string, not the node '
                        f'"{expr}"; the order is {call}.')
    if isinstance(expr, (list, tuple, numpy.ndarray)) and not all(
            isinstance(item, str) for item in expr):
        raise TypeError(f'The expression must be a string, not {expr!r}; '
                        f'the order is {call}.')
    if isinstance(expr, (list, tuple)):
        raise TypeError(f'{name}() takes one expression at a time, not '
                        f'{expr!r}; call it once per expression.')
    raise TypeError(f'The expression must be a string such as "T", not '
                    f'{expr!r}.')


def _check_reserved(name: str, properties: dict):
    """Raises for properties the helper sets itself."""
    for key in properties:
        if key == 'position':
            raise ValueError(f'{name}() gives no position; mk.maximum and '
                             'mk.minimum do.')
        use = RESERVED.get(key)
        for prefixes, hint in RESERVED_PREFIXES.items():
            if key.startswith(prefixes):
                use = hint
        if use is not None:
            raise ValueError(f'{name}() sets "{key}" itself; use {use}.')


def check_geometry(name: str, geom: Node):
    """Raises for work planes and 1D geometries."""
    _comsol.check_not_workplane(geom, name)
    if _comsol.sdim(geom) < 2:
        raise ValueError(f'{name}() needs a 2D or 3D geometry; "{geom}" is '
                         '1D.')


def check_current(geom: Node):
    """
    Raises if the geometry changed since the solve.

    COMSOL then evaluates the old solution without a warning, but the
    geometry is no longer built, or it was rebuilt and its mesh is empty.
    A rebuild without changes keeps the mesh.
    """
    message = (f'Geometry "{geom}" changed or its mesh was cleared since '
               'the solve; run model.build(geom), model.mesh() and '
               'model.solve().')
    try:
        _comsol.check_built(geom)
    except RuntimeError:
        raise RuntimeError(message) from None
    meshes = _comsol.component_of(geom).mesh()
    tags = list(meshes.tags())
    if tags and all(meshes.get(tag).isEmpty() for tag in tags):
        raise RuntimeError(message)


def _check_unit(expr: str, unit: str, header: str, plain: str):
    """
    Raises if COMSOL ignored `unit`: `header` is the column header with the
    unit, `plain` the one without. The header ends in the unit applied,
    spelled as requested, e.g. "Temperature (degC)".
    """
    if header == plain:
        raise unit_error(expr, _unit_of(header), unit)


def unit_error(expr: str, applied: str, unit: str) -> ValueError:
    """Returns the error for a unit COMSOL ignored."""
    return ValueError(f'COMSOL evaluated "{expr}" in {applied}, not '
                      f'{unit!r}; give a unit of the same kind, or none for '
                      'SI units.')


def _check_point_unit(create, model, data: str, expr: str, unit: str,
                      point: Array):
    """
    Checks `unit` on a cut point at `point` of dataset `data` (a tag):
    unlike Interp, EvalPoint writes a table, whose header shows the unit
    applied.
    """
    cut = create(model.result().dataset(), f'CutPoint{len(point)}D')
    cut.set('data', data)
    for axis, coordinate in zip('xyz', point):
        cut.set(f'point{axis}', repr(float(coordinate)))

    def header(unit: str | None) -> str:
        feature = create(model.result().numerical(), 'EvalPoint')
        table = create(model.result().table(), 'Table')
        feature.set('data', str(cut.tag()))
        feature.set('expr', expr)
        if unit is not None:
            feature.set('unit', unit)
        feature.set('table', str(table.tag()))
        try:
            feature.setResult()
        except Exception as error:
            raise failed(expr, error) from error
        # e.g. "Temperature (K), Point: (0.05, 0.01, 0.005)"
        return str(table.getColumnHeaders()[-1]).split(', Point:')[0]

    applied = header(unit)
    if not applied.endswith(f'({unit})'):
        _check_unit(expr, unit, applied, header(None))


#########
# Steps #
#########

def steps(step, count: int | None, where: str, *,
          single: str | None = None, what: str = 'step', hint: str = ''
          ) -> tuple[list[int], bool]:
    """
    Returns the step numbers `step` stands for and whether it asks for
    several (an array). With `count=None`, only checks its form.
    `where` names what has the steps in messages, e.g. `Dataset "s//Solution
    1"`, and `hint` follows the error for several steps, e.g. where to look
    them up. `single` names a caller that takes one step only, such as
    `'plot'`. `what` is the argument's name, `'step'` or `'outer'`.
    """
    nouns = 'steps' if what == 'step' else 'outer values'
    noun = 'step' if what == 'step' else 'outer value'
    forms = ("'first', 'last' or a number" if single else
             "'first', 'last', 'all', a number or a list of numbers")
    if single and (step == 'all' if isinstance(step, str)
                   else numpy.ndim(step) > 0):
        raise ValueError(f"{single}() draws one {noun}; pass {what}='last' "
                         f'or a number, not {step!r}.')
    if isinstance(step, str):
        if step not in STEPS:
            raise ValueError(f'{what} must be {forms}, not {step!r}.')
        if count is None:
            return [], step == 'all'
        return {'all': list(range(1, count + 1)), 'first': [1],
                'last': [count]}[step], step == 'all'
    if step is None:
        if count is None or count == 1:
            return [1], False
        choices = (f' or a number from 1 to {count}' if single else
                   f", a number from 1 to {count}, a list of them or 'all'")
        raise ValueError(f"{where} has {count} {nouns}; pass {what}='last'"
                         f'{choices}{hint}.')
    many = numpy.ndim(step) > 0
    items = list(step) if many else [step]
    if many and not items:
        raise ValueError(f'{what} is an empty list.')
    for item in items:
        if (isinstance(item, (bool, numpy.bool_))
                or not isinstance(item, numbers.Integral)):
            raise TypeError(f'{what} must be {forms}, not {step!r}.')
    found = [int(item) for item in items]
    if any(n < 1 for n in found):
        raise ValueError(f"{what} counts from 1 as in COMSOL: {what}=1 for "
                         f"the first, {what}='last' for the last.")
    if count is not None:
        wrong = [n for n in found if n > count]
        if wrong:
            raise ValueError(f'{where} has {count} '
                             f'{noun if count == 1 else nouns}, not '
                             f'{wrong[0]}{hint}.')
    return found, many


def _unique(steps: list[int]) -> tuple[list[int], list[int]]:
    """
    Returns the steps once each, in increasing order, and for each
    requested step its position among those, so that a step asked twice
    is evaluated once.
    """
    unique = sorted(set(steps))
    return unique, [unique.index(n) for n in steps]


##########
# Points #
##########

def _points(geom: Node, points, sdim: int) -> tuple[Array, bool]:
    """Returns the points as rows of an array, and whether it was one."""
    names = ('(r, z)' if axisymmetric(geom)
             else '(x, y)' if sdim == 2 else '(x, y, z)')
    try:
        array = numpy.asarray(points)
    except Exception:
        array = numpy.asarray(None)
    if array.dtype.kind not in 'iuf':
        raise TypeError(f'points must be numbers, one point {names} or '
                        f'rows of them, not {points!r}.')
    if array.ndim == 1 and len(array) == sdim:
        return array.reshape(1, sdim).astype(float), True
    if array.ndim == 2 and array.shape[1] == sdim and len(array):
        return array.astype(float), False
    message = (f'points must be one point {names} or rows of them (shape '
               f'(n, {sdim})); got shape {array.shape}')
    if array.ndim == 2 and array.shape[0] == sdim:
        message += (f', which looks transposed if you meant '
                    f'{array.shape[1]} points')
    raise ValueError(message + '.')


def _outside_message(missing: Array, where: str = '') -> str:
    """
    Names the points (counted from 1) that have no value, at `where` (an
    outer value) if given.
    """
    points = [str(n + 1) for n in numpy.flatnonzero(missing)]
    shown = (' and '.join(points) if len(points) <= 2
             else f'{", ".join(points[:-1])} and {points[-1]}')
    noun = 'Point' if len(points) == 1 else 'Points'
    verb = 'is' if len(points) == 1 else 'are'
    at = f' at {where}' if where else ''
    return (f'{noun} {shown} of {len(missing)} {verb} outside the geometry '
            f"or where nothing was solved{at}; pass outside='nan' to get "
            'nan there.')


##########
# Common #
##########

def axisymmetric(geom: Node) -> bool:
    """Tells whether a 2D geometry is axisymmetric."""
    try:
        return bool(_comsol.java_of(geom).isAxisymmetric())
    except Exception:
        return False


def _unit_of(header: str) -> str:
    """Returns the unit in parentheses at the end of a column header."""
    if not header.endswith(')'):
        return header
    depth = 0
    for i in range(len(header) - 1, -1, -1):
        depth += {')': 1, '(': -1}.get(header[i], 0)
        if depth == 0:
            return header[i + 1:-1]
    return header


def failed(expr: str, error: Exception) -> RuntimeError:
    """Returns the error for an expression COMSOL could not evaluate."""
    return RuntimeError(f'COMSOL could not evaluate "{expr}": '
                        f'{_comsol.reason(error)}')


def _scalar(values: Array) -> Any:
    """Returns a 0-d array as a float or complex, others as they are."""
    if numpy.ndim(values):
        return values
    item = values.item() if isinstance(values, numpy.ndarray) else values
    return complex(item) if numpy.iscomplexobj(values) else float(item)
