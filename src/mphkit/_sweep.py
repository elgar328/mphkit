"""
Parametric sweeps that COMSOL stores as an outer loop. Public as
`mk.outer_values`.

COMSOL stores a sweep as an outer loop around a time-dependent,
frequency-domain or eigenvalue study, over geometry or mesh parameters,
materials and functions: one solution per value (a child), named by the
sweep's solution, which `Solutions` in `_datasets` sorts out. Each child
carries its parameter names and values in SI units. A stationary study
swept over other parameters keeps them as steps instead.
"""
from __future__ import annotations

import math
import numbers
import re
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy
from mph.node import Node
from numpy.typing import NDArray

from . import _comsol, _datasets, _results

# Inner step names that are no model parameters: stationary, time,
# frequency and eigenvalue steps
STEP_NAMES = ('', 't', 'freq', 'lambda')
# Prefixes of the outer names of material and function sweeps
SWITCHES = ('matsw.', 'funsw.')
# Values shown in messages: the first ones, then the last ones
SHOWN = (5, 2)
# Relative tolerances: step values compared between outer values, and
# vertex coordinates
TOLERANCE = 1e-9
COORDINATES = 1e-12

# a number as COMSOL prints it in plot titles, e.g. 133.33 or 5.5511E-17,
# with this many significant digits
SIGNIFICANT = 5
NUMBER = r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'

Array = NDArray[Any]


def outer_values(geom: Node, /, *, dataset=None) -> list[dict[str, float]]:
    """
    Returns the parameter values of a sweep stored as an outer loop, one
    dictionary per value, in SI units; the first one is `outer=1`:

    ```python
    mk.outer_values(geom)   # [{'Th': 373.15}, {'Th': 473.15}, {'Th': 573.15}]
    ```

    A sweep over `Th` from 100 to 300 degC gives kelvin, one over a length
    in mm gives meters, as the results helpers do without `unit`. Material
    and function sweeps give the number of the case
    (`{'matsw.comp1.sw1': 2.0}`); errors that list the values show their
    names. A sweep that keeps the last solution only gives one dictionary.

    `[]` means that the dataset has no outer sweep. A stationary study
    swept over parameters that change no geometry stores them as steps:
    that raises, pointing to `step=`. `dataset` works as in
    `mk.integral()`, also by study. Leaves nothing in the model.
    """
    _results.check_geometry('outer_values', geom)
    model = geom.model.java
    chosen = _datasets.select(geom, dataset)
    data = chosen.java
    if chosen.kind == 'plain':
        solution = _datasets.solution_of(model, data)
        names = stepped(model, solution)
        if names:
            name = _datasets.describe(data)
            raise ValueError(
                f'Dataset {name} holds its sweep over {", ".join(names)} as '
                'steps, not as outer values: pass step= to the results '
                f'helpers (model.inner("{_name(data)}") gives the values).')
        return []
    return values(model, str(data.getString('solution')))


def values(model, sweep: str) -> list[dict[str, float]]:
    """
    Returns the outer parameter values of each child of the sweep in
    solution `sweep`, in SI units and in the children's order of names.
    """
    outer = set(_outer_names(model.sol(sweep)))
    found = []
    for child in _datasets.stored(model.sol(sweep)):
        solution = model.sol(child)
        names = _datasets.param_names(solution)
        missing = sorted(outer - set(names))
        if missing:
            raise RuntimeError(f'Solution {child} of the sweep in solution '
                               f'{sweep} has no value of '
                               f'{", ".join(missing)}.')
        numbers = [float(value) for value in solution.getParamVals()]
        found.append({name: number for name, number in zip(names, numbers)
                      if name in outer})
    return found


def labels(model, sweep: str) -> list[str]:
    """
    Returns, for each child of the sweep in solution `sweep`, its values
    for messages: in SI units as given back, then as swept, e.g.
    `Th=473.15 (200 degC)`. Material and function sweeps add the child's
    label, e.g. `[Material Switch 1=Material 2]`.
    """
    solution = model.sol(sweep)
    info = solution.getSolutioninfo()
    units = dict(zip(_outer_names(solution), _chars(info.getPUnitsOuter())))
    found = []
    for number, child, row in zip(
            _datasets.outer_numbers(solution),
            _datasets.stored(solution), values(model, sweep)):
        try:
            swept = [float(value) for value in info.getPvals([[number, 1]])[0]]
        except Exception:
            swept = []
        names = [name for name in _datasets.param_names(model.sol(child))
                 if name in row]
        parts = []
        for i, name in enumerate(names):
            part = f'{name}={row[name]!r}'
            unit = units.get(name)
            if unit and i < len(swept):
                part += f' ({swept[i]:g} {unit})'
            parts.append(part)
        text = ', '.join(parts)
        if any(name.startswith(SWITCHES) for name in names):
            text += f' [{model.sol(child).label()}]'
        found.append(text)
    return found


def label_of(model, solution: str) -> str:
    """Returns the parameter values a solution was made for, for messages."""
    java = model.sol(solution)
    names = _datasets.param_names(java)
    numbers = [float(value) for value in java.getParamVals()]
    return ', '.join(f'{name}={number!r}'
                     for name, number in zip(names, numbers))


def listed(labels: list[str]) -> str:
    """Lists values for messages, counted from 1, shortened when long."""
    lines = [f'{n}: {label}' for n, label in enumerate(labels, 1)]
    first, last = SHOWN
    if len(lines) > first + last + 1:
        lines = lines[:first] + ['...'] + lines[-last:]
    return '; '.join(lines)


def stepped(model, solution) -> list[str]:
    """
    Returns the names of the model parameters a plain solution is swept
    over as steps: a stationary study's parametric or auxiliary sweep.
    """
    try:
        names = [str(name) for name in solution.getPNames()]
        parameters = {str(name) for name in model.param().varnames()}
    except Exception:
        return []
    return [name for name in names
            if name not in STEP_NAMES and name in parameters]


def _outer_names(solution) -> list[str]:
    """Returns the names of the outer parameters of a sweep's solution."""
    return [name or '' for name in
            _chars(solution.getSolutioninfo().getPNamesOuter())]


def _chars(arrays) -> list[str | None]:
    """Joins Java char arrays into strings; `None` stays."""
    return [None if array is None else ''.join(str(c) for c in array)
            for array in arrays]


def _name(dataset) -> str:
    """Returns the MPh name of a Java dataset."""
    return _comsol.name_of(dataset)


def step_values(geom: Node, /, *, dataset=None, outer=None
                ) -> dict[str, Array]:
    """
    Returns the values of the steps of a solved study by name, in SI
    units: element k-1 of each array belongs to `step=k`.

    ```python
    mk.step_values(geom)              # {'t': array([0., 10., 20.])}
    mk.step_values(geom, outer=2)     # the steps of outer value 2
    mk.step_values(geom, outer='all') # {'t': array of (values, steps)}
    ```

    One outer value (or none) gives one-dimensional arrays, `'all'` or a
    list two-dimensional ones, a row per outer value. Times are in
    seconds (also from a study in minutes), frequencies in Hz (also from a
    list in kHz), swept parameters in SI units (degC gives K). Several
    swept parameters come in COMSOL's order, not necessarily the study's.
    Eigenvalues (`'lambda'`) are complex unless all imaginary parts are 0;
    an eigenfrequency study adds `'freq'` in Hz, iλ/(2π), as COMSOL's
    `solid.freq` and the acoustics `freq` (checked for these two; complex
    with damping, take `.real`). A stationary study without sweep gives
    `{}`: one step, without a value.

    `dataset` works as in `mk.integral()`, `outer` as there. Without
    `outer`, a sweep's outer values must all have the same steps; when a
    solver chose the time steps they differ, so go through them one by
    one:

    ```python
    for k in range(1, len(mk.outer_values(geom)) + 1):
        times = mk.step_values(geom, outer=k)['t']
    ```

    MPh's `model.inner()` reads the same values, but interleaved for
    several parameters, real parts only, and for the last outer value.
    Leaves nothing in the model.
    """
    _results.check_geometry('step_values', geom)
    check_outer(outer)
    model = geom.model.java
    chosen = _datasets.select(geom, dataset)
    data = chosen.java
    where = f'Dataset {_datasets.describe(data)}'
    if chosen.kind == 'plain':
        if outer is not None:
            raise no_outer(model, data, outer, 'step_values', dataset)
        return named_steps(model, _datasets.solution_of(model, data))
    sweep = Sweep(model, str(data.getString('solution')))
    if outer is None:
        if len(sweep.children) == 1 or same_steps(model, sweep.children,
                                                  strict=True):
            return named_steps(model, model.sol(sweep.children[0]))
        counts = {sweep.count(k) for k in range(1, len(sweep.children) + 1)}
        rows = ("outer=k for one value, or outer='all' for a row per value"
                if len(counts) == 1 else 'outer=k for one value at a time')
        raise ValueError(f'The outer values of {where} have different steps '
                         f'({_differences(model, sweep)}); pass {rows}.')
    positions, many = pick(outer, sweep, data, None)
    tables = [named_steps(model, model.sol(sweep.children[k - 1]))
              for k in positions]
    if not many:
        return tables[0]
    names = list(tables[0])
    lengths = {len(next(iter(table.values()), ())) for table in tables}
    if any(list(table) != names for table in tables) or len(lengths) > 1:
        raise ValueError(f'The outer values of {where} have different steps '
                         f'({_differences(model, sweep)}); pass outer=k for '
                         'one value at a time.')
    return {name: numpy.array([table[name] for table in tables])
            for name in names}


#########
# Sweep #
#########

class Sweep:
    """An outer sweep's solution: its children, values and steps."""

    def __init__(self, model, tag: str) -> None:
        self.model = model
        self.tag = tag
        solution = model.sol(tag)
        self.numbers = _datasets.outer_numbers(solution)
        self.children = _datasets.stored(solution)
        self.names = _outer_names(solution)
        self._labels: list[str] | None = None

    @property
    def labels(self) -> list[str]:
        """The values of each child for messages (see `labels()`)."""
        if self._labels is None:
            self._labels = labels(self.model, self.tag)
        return self._labels

    def count(self, k: int) -> int:
        """Returns the number of steps of outer value `k` (from 1)."""
        info = self.model.sol(self.children[k - 1]).getSolutioninfo()
        return len(info.getSolnum(1, True))

    def where(self, data, k: int) -> str:
        """Names outer value `k` of dataset `data` in messages."""
        return (f'Dataset {_datasets.describe(data)} at outer={k} '
                f'({self.labels[k - 1]})')

    def materials(self) -> bool:
        """Tells whether it sweeps materials only (no geometry change)."""
        return all(name.startswith('matsw.') for name in self.names)

    def swept(self, k: int) -> dict[str, float]:
        """
        Returns the outer values of `k` (from 1) as swept, e.g. in degC,
        by name; COMSOL gives them in the order of the child's names.
        """
        info = self.model.sol(self.tag).getSolutioninfo()
        row = [float(v) for v in info.getPvals([[self.numbers[k - 1], 1]])[0]]
        names = _datasets.param_names(self.model.sol(self.children[k - 1]))
        return {name: number for name, number in zip(names, row)
                if name in self.names}

    def title(self, k: int, data, step: int | None) -> Title:
        """
        What the title of a picture of outer value `k` at `step` must
        show (`None` with one step).
        """
        every = [self.swept(n) for n in range(1, len(self.children) + 1)]
        plain = [name for name in every[k - 1]
                 if not name.startswith(SWITCHES)]
        label = str(self.model.sol(self.children[k - 1]).label())
        switches = []
        for part in label.split(', '):
            left, _, right = part.partition('=')
            if left not in plain and right:
                switches.append((left, right))
        wanted = sum(name.startswith(SWITCHES) for name in self.names)
        steps = {}
        if step is not None:
            table = named_steps(self.model, self.model.sol(
                self.children[k - 1]))
            uniform = same_steps(self.model, self.children)
            for name, item in STEP_ITEMS.items():
                if name in table and numpy.isrealobj(table[name]):
                    steps[item] = (float(table[name][step - 1]),
                                   not uniform)
        return Title(self.numbers[k - 1],
                     {name: every[k - 1][name] for name in plain},
                     {name: [row[name] for row in every] for name in plain},
                     switches, self.where(data, k), len(switches) >= wanted,
                     steps)


class Target(NamedTuple):
    """A dataset to evaluate and its steps, in the order asked."""
    data: str
    solnums: list[int]
    where: str  # names the outer value in messages, or ''


class Restriction(NamedTuple):
    """
    Why a sweep is read with selection nodes or all entities only:
    `'sweep'`, it changes the geometry, or `'unknown'`, its mesh leaves
    part of the geometry out. `matching()` lists the values solved on the
    geometry as built, for messages.
    """
    kind: str
    where: str
    matching: Callable[[], list[str]]


class Request(NamedTuple):
    """
    What to evaluate: a target per outer value (each once), the position
    of each requested value among them, whether `outer` and `step` ask
    for several (an axis each), and the restriction of a sweep whose
    geometry cannot be the one built, if any.
    """
    targets: list[Target]
    order: list[int]
    many_outer: bool
    many_step: bool
    restricted: Restriction | None = None


def resolve(create, geom: Node, dataset, step, outer) -> Request:
    """
    Returns the datasets and steps to evaluate for `dataset`, `step` and
    `outer`: the dataset itself, or a temporary dataset per value of an
    outer sweep, made with `create`, after checking the geometry.
    """
    model = geom.model.java
    chosen = _datasets.select(geom, dataset)
    data = chosen.java
    if chosen.kind == 'plain':
        if outer is not None:
            raise no_outer(model, data, outer, None, dataset)
        _results.check_current(geom)
        count = _datasets.step_count(model, data)
        solnums, many = _results.steps(
            step, count, f'Dataset {_datasets.describe(data)}',
            hint=f' ({_call("step_values", dataset)} gives their times or '
                 'parameter values)')
        return Request([Target(str(data.tag()), solnums, '')], [0], False,
                       many)
    sweep = Sweep(model, str(data.getString('solution')))
    positions, many_outer = pick(outer, sweep, data, step)
    unique, order = _results._unique(positions)
    found = _steps(sweep, data, unique, step, dataset, None)
    children = Children(create, geom, sweep, data)
    decision = check(geom, sweep, children, unique, data)
    restricted = None
    if decision in ('sweep', 'unknown'):
        restricted = Restriction(
            decision, f'dataset {_datasets.describe(data)}',
            lambda: _matching(geom, sweep, children, data))
    targets = [Target(str(children.dataset(k).tag()), found[k][0],
                      f'outer={k} ({sweep.labels[k - 1]})') for k in unique]
    return Request(targets, order, many_outer, found[unique[0]][1],
                   restricted)


def _matching(geom: Node, sweep: Sweep, children: Children, data
              ) -> list[str]:
    """Lists the outer values solved on the geometry as built."""
    try:
        _comsol.check_built(geom)
    except RuntimeError:
        return []
    current = vertices(geom)
    found = []
    for k in range(1, len(sweep.children) + 1):
        points = children.points(k)
        if points is not None and same_points(points, current):
            found.append(f'outer={k} ({sweep.labels[k - 1]})')
    return found


def _steps(sweep: Sweep, data, unique: list[int], step, dataset,
           single: str | None) -> dict[int, tuple[list[int], bool]]:
    """
    Returns, for each requested outer value, the steps `step` stands for
    and whether it asks for several. Several values need the same steps,
    unless `step` is `'first'` or `'last'`.
    """
    model = sweep.model
    several = len(unique) > 1 and not (isinstance(step, str)
                                       and step in ('first', 'last'))
    if several and not same_steps(model, [sweep.children[k - 1]
                                          for k in unique]):
        where = f'dataset {_datasets.describe(data)}'
        raise ValueError(
            f'The outer values {_numbers(unique)} of {where} have different '
            f"steps ({_differences(model, sweep, unique)}); pass step="
            "'first' or 'last', or one outer value at a time, e.g. "
            f'outer={unique[0]} ({_call("step_values", dataset)[:-1]}, '
            f'outer={unique[0]}) gives its steps).')
    return {k: _results.steps(
        step, sweep.count(k), sweep.where(data, k), single=single,
        hint=f' ({_call("step_values", dataset)[:-1]}, outer={k}) gives '
             f'them: {_preview(model, sweep.children[k - 1])})')
            for k in unique}


class Picture(NamedTuple):
    """One picture to draw of a dataset."""
    data: Any
    outer: int | None   # COMSOL's outer solution number (`outersolnum`)
    step: int | None    # the step (`looplevel`), None if there is one
    number: int | None  # the outer value, counted from 1 as in `outer=`
    title: Title | None  # what its title must show


class Title(NamedTuple):
    """What the title of a picture of one outer value must show."""
    number: int  # COMSOL's outer solution number
    expected: dict[str, float]  # values as swept, by name
    candidates: dict[str, list[float]]  # those of all values, by name
    switches: list[tuple[str, str]]  # e.g. ('Material Switch 1', 'Steel')
    where: str
    # whether every outer parameter has an item above to check
    complete: bool = True
    # the step drawn, by title item ('Time', 'freq'), in SI units, and
    # whether it must show: when the values have different steps
    steps: dict[str, tuple[float, bool]] = {}


# Units COMSOL prints the step items of titles in, to SI
STEP_UNITS = {
    'Time': {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, '\u00b5s': 1e-6,
             'min': 60.0, 'h': 3600.0, 'd': 86400.0},
    'freq': {'Hz': 1.0, 'mHz': 1e-3, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9,
             'THz': 1e12}}
# Step names and the title items that show them
STEP_ITEMS = {'t': 'Time', 'freq': 'freq'}


def pictures(create, geom: Node, dataset, step, outer
             ) -> tuple[list[Picture], bool]:
    """
    Returns what to draw for `dataset`, `step` and `outer`, a picture per
    outer value, and whether `outer` asks for several. A sweep's pictures
    are drawn from its dataset, so every value must have been solved on
    the geometry as built.
    """
    model = geom.model.java
    chosen = _datasets.select(geom, dataset)
    data = chosen.java
    if chosen.kind == 'plain':
        if outer is not None:
            raise no_outer(model, data, outer, None, dataset)
        _results.check_current(geom)
        count = _datasets.step_count(model, data)
        solnums, _ = _results.steps(
            step, count, f'Dataset {_datasets.describe(data)}',
            single='plot',
            hint=f' ({_call("step_values", dataset)} gives their times or '
                 'parameter values)')
        return [Picture(data, None, solnums[0] if count > 1 else None,
                        None, None)], False
    sweep = Sweep(model, str(data.getString('solution')))
    positions, many = pick(outer, sweep, data, step)
    twice = sorted({k for k in positions if positions.count(k) > 1})
    if twice:
        raise ValueError(f'outer asks for value {twice[0]} more than once; '
                         'mk.plot draws one picture per value.')
    found = _steps(sweep, data, positions, step, dataset, 'plot')
    children = Children(create, geom, sweep, data)
    decision = check(geom, sweep, children, positions, data)
    if decision == 'full' and not sweep.materials():
        current = vertices(geom)
        for k in range(1, len(sweep.children) + 1):
            points = children.points(k)
            if points is None:
                decision = 'unknown'
                break
            if not same_points(points, current):
                decision = 'sweep'
                break
    where = f'dataset {_datasets.describe(data)}'
    if decision == 'unknown':
        raise RuntimeError(
            f'The solution of {where} covers part of the geometry only (no '
            'physics on the rest), so it cannot be checked against the '
            'geometry as built; for pictures, mesh all domains with a mesh '
            'of your own (not physics-controlled) and solve again.')
    if decision == 'sweep':
        raise ValueError(
            f'The sweep of {where} changes the geometry, and mk.plot draws '
            'on the geometry as built. To draw one value, set its '
            'parameters (mk.outer_values(geom)), run model.build(geom) and '
            'model.mesh(), and solve a study without the sweep.')
    drawn = {k: found[k][0][0] if sweep.count(k) > 1 else None
             for k in positions}
    return [Picture(data, sweep.numbers[k - 1], drawn[k], k,
                    sweep.title(k, data, drawn[k])) for k in positions], many


def title_problem(indicator: str, title: Title) -> str | None:
    """
    Returns `'missing'` if a picture's title (COMSOL's
    `evaluatedparamindicator`) does not show the outer values, `'wrong'`
    if it shows others than `title` expects, else `None`. COMSOL prints
    the values as swept, rounded, e.g. "Th(2)=200 degC Time=10 s"; the
    number in parentheses is its outer solution number.
    """
    if not title.complete or not (title.expected or title.switches):
        return 'missing'
    for name, value in title.expected.items():
        found = list(re.finditer(rf'(?<![\w.]){re.escape(name)}'
                                 rf'(?:\((\d+)\))?=\s*{NUMBER}', indicator))
        if not found:
            return 'missing'
        for match in found:
            if (match.group(1) is not None
                    and int(match.group(1)) != title.number):
                return 'wrong'
            if not _printed_as(match.group(2), value,
                               title.candidates[name]):
                return 'wrong'
    for left, right in title.switches:
        found = list(re.finditer(rf'(?<![\w.]){re.escape(left)}'
                                 r'(?:\((\d+)\))?=', indicator))
        if not found:
            return 'missing'
        for match in found:
            if (match.group(1) is not None
                    and int(match.group(1)) != title.number):
                return 'wrong'
            # the name ends the title, an item, or comes before the next
            if not re.match(rf'{re.escape(right)}(?=$|,|\s+[^\s=,]+=)',
                            indicator[match.end():]):
                return 'wrong'
    for item, (value, required) in title.steps.items():
        shown = re.search(rf'(?<![\w.]){item}(?:\(\d+\))?=\s*{NUMBER}'
                          r'\s*([^\s,]*)', indicator)
        factor = None if shown is None else \
            STEP_UNITS[item].get(shown.group(2))
        if shown is None or factor is None:
            if required:
                return 'missing'
            continue
        if not _printed_as(shown.group(1), value/factor, [value/factor]):
            return 'wrong'
    return None


def _printed_as(text: str, value: float, candidates: list[float]) -> bool:
    """
    Tells whether `value` printed with the digits of `text` gives it, or
    is the closest to it among the values of all outer values.
    """
    printed = float(text)
    mantissa, _, exponent = text.lower().partition('e')
    decimals = len(mantissa.partition('.')[2])
    half = 0.5 * 10.0**(int(exponent or 0) - decimals)
    if printed:
        # COMSOL prints 5 significant digits and drops trailing zeros:
        # "0.1" is 0.10000, not anything from 0.05 to 0.15
        half = min(half, 0.5 * 10.0**(math.floor(math.log10(abs(printed)))
                                      - SIGNIFICANT + 1))
    if abs(printed - value) <= half*(1 + 1e-9):
        return True
    if len(candidates) < 2:
        return abs(printed - value) <= max(1e-3*abs(value), 1e-12)
    distances = [abs(c - printed) for c in candidates]
    best = min(distances)
    return any(abs(c - value) <= 1e-12*abs(value)
               for c, d in zip(candidates, distances)
               if d <= best*(1 + 1e-9))


def pick(outer, sweep: Sweep, data, step) -> tuple[list[int], bool]:
    """
    Returns the outer values (positions from 1) `outer` stands for and
    whether it asks for several.
    """
    count = len(sweep.children)
    where = f'Dataset {_datasets.describe(data)}'
    if outer is None:
        if count == 1:
            return [1], False
        message = (f'{where} holds a parametric sweep over {count} values '
                   f'({listed(sweep.labels)}); pass outer= a number, a list '
                   "of them, 'all', 'first' or 'last' (mk.outer_values(geom) "
                   'gives the values).')
        if step is not None and all(sweep.count(k) == 1
                                    for k in range(1, count + 1)):
            message += (' Each value has one step: the sweep is stored as '
                        f'an outer loop, so pass outer={step!r} instead of '
                        'step=.')
        raise ValueError(message)
    if isinstance(outer, Mapping):
        return [by_value(sweep, outer, where)], False
    if _by_name(outer):
        return [by_value(sweep, wanted, where) for wanted in outer], True
    return _results.steps(outer, count, where, what='outer')


#################
# Picking value #
#################

def check_outer(outer):
    """Checks the form of `outer`, before COMSOL is asked anything."""
    if isinstance(outer, Mapping):
        _check_values(outer)
    elif _by_name(outer):
        for wanted in outer:
            _check_values(wanted)
    else:
        _results.steps(outer, None, '', what='outer')


def _by_name(outer) -> bool:
    """
    Tells whether `outer` is a list of values by name; raises for a list
    that mixes them with numbers.
    """
    if not isinstance(outer, (list, tuple)) or not outer:
        return False
    named = [isinstance(item, Mapping) for item in outer]
    if any(named) and not all(named):
        raise TypeError(f'outer mixes numbers and values by name: '
                        f'{outer!r}; give either.')
    return all(named)


def _check_values(wanted):
    """Checks one outer value by name, e.g. {'Th': '200[degC]'}."""
    if not wanted:
        raise TypeError("outer={} names no parameter; e.g. outer="
                        "{'Th': '200[degC]'} or {'Th': 473.15} (SI).")
    for name, given in wanted.items():
        if not isinstance(name, str):
            raise TypeError(f'outer takes parameters by name, not '
                            f'{name!r}.')
        if (isinstance(given, (bool, numpy.bool_))
                or not isinstance(given, (numbers.Real, str))):
            raise TypeError(f'outer={{{name!r}: ...}} takes a number in SI '
                            f"units or a value with its unit such as "
                            f"'200[degC]', not {given!r}.")


def by_value(sweep: Sweep, wanted: Mapping, where: str) -> int:
    """
    Returns the outer value (counted from 1) that has the values
    `wanted`, by name: numbers in SI units, or strings with a unit that
    COMSOL converts, e.g. '200[degC]'. Some parameters are enough if they
    pick one value.
    """
    model = sweep.model
    rows = values(model, sweep.tag)
    swept = [sweep.swept(k) for k in range(1, len(rows) + 1)]
    for name in wanted:
        if name not in sweep.names:
            steps = _names(model.sol(sweep.children[0]))
            if name in steps:
                raise ValueError(
                    f'{where} holds {name} as steps of each outer value, not '
                    f"as an outer value: pass step= (mk.step_values(geom, "
                    'outer=k) gives them).')
            raise ValueError(f'{where} has no outer parameter {name!r}; its '
                             f'outer parameters are '
                             f'{", ".join(sweep.names)}.')
    targets = {name: _si(model, name, given)
               for name, given in wanted.items()}
    found = [k for k, row in enumerate(rows, 1)
             if all(_close(row[name], target, [r[name] for r in rows])
                    for name, target in targets.items())]
    asked = ', '.join(f'{name}={given!r}' for name, given in wanted.items())
    if len(found) == 1:
        return found[0]
    if not found:
        message = (f'{where} has no value with {asked}; it has '
                   f'{listed(sweep.labels)}.')
        numbers_given = {name: float(given) for name, given in wanted.items()
                         if not isinstance(given, str)}
        if numbers_given and any(
                all(_close(row.get(name, math.nan), number,
                           [r.get(name, math.nan) for r in swept])
                    for name, number in numbers_given.items())
                for row in swept):
            message += (' Numbers are in SI units; give the unit as a '
                        "string instead, e.g. '200[degC]'.")
        raise ValueError(message)
    if all(rows[k - 1] == rows[found[0] - 1] for k in found):
        raise ValueError(f'{where} has {asked} more than once (outer='
                         f'{_numbers(found)}); pass outer= one of these '
                         'numbers.')
    raise ValueError(f'{asked} fits several values of {where}: '
                     f'{listed(sweep.labels)}; give more parameters, or '
                     'the number.')


def _step_of(model, solution, wanted: Mapping) -> int | None:
    """
    Returns the step (from 1) of a sweep stored as steps that has the
    values `wanted` by name, or `None` unless exactly one has.
    """
    table = named_steps(model, solution)
    targets = {name: _si(model, name, given)
               for name, given in wanted.items()}
    count = len(next(iter(table.values())))
    found = [k for k in range(1, count + 1)
             if all(_close(float(table[name][k - 1].real), target,
                           [float(v) for v in table[name].real])
                    for name, target in targets.items())]
    return found[0] if len(found) == 1 else None


def _si(model, name: str, given) -> float:
    """
    Returns a value given for parameter `name` in SI units: a number as
    it is, a string converted by COMSOL with the parameter's unit.
    """
    if not isinstance(given, str):
        return float(given)
    if name.startswith(SWITCHES):
        raise ValueError(f'{name} takes the number of the case, not '
                         f'{given!r}.')
    parameters = model.param()
    unit = parameters.evaluateUnit(name)
    if unit is None:
        raise ValueError(f'Parameter {name} has no unit, or the model no '
                         f'longer has it: give its value as a number (in SI '
                         f'units), not {given!r}.')
    try:
        with _comsol.history_off(model):
            return float(parameters.evaluate(given, str(unit)))
    except Exception as error:
        raise ValueError(f'COMSOL cannot read {given!r} as a value of {name} '
                         f'in {unit}: {_comsol.reason(error)} Give the unit '
                         "in brackets, e.g. '200[degC]', or a number in SI "
                         'units.') from error


def _close(value: float, target: float, every: list[float]) -> bool:
    """Tells whether two values agree within 1e-9 of the largest swept."""
    scale = max([abs(v) for v in every if not math.isnan(v)] or [0.0])
    return abs(value - target) <= TOLERANCE * scale


def no_outer(model, data, outer, caller: str | None, dataset
             ) -> ValueError:
    """
    The error for `outer` given with a dataset without outer sweep;
    `caller` is `'step_values'` for that one.
    """
    where = f'Dataset {_datasets.describe(data)}'
    names = stepped(model, _datasets.solution_of(model, data))
    if names and caller == 'step_values':
        return ValueError(
            f'{where} holds its sweep over {", ".join(names)} as steps, not '
            'as outer values; leave out outer= to get them.')
    if names:
        instead = (f'step={outer!r}' if not isinstance(outer, (Mapping, list,
                                                              tuple))
                   else 'step=')
        if isinstance(outer, Mapping) and set(outer) <= set(names):
            found = _step_of(model, _datasets.solution_of(model, data),
                             outer)
            if found is not None:
                instead = f'step={found}'
        return ValueError(
            f'{where} holds its sweep over {", ".join(names)} as steps, not '
            f'as outer values: pass {instead} instead of outer= '
            f'({_call("step_values", dataset)} gives the values).')
    return ValueError(f'{where} has no outer sweep; leave out outer=.')


##############
# Selections #
##############

# Component selections that COMSOL evaluates on each value's geometry,
# alone or from their inputs
POSITIONS = ('Box', 'Ball', 'Cylinder', 'Disk')
COMBINED = ('Union', 'Intersection', 'Difference', 'Complement', 'Adjacent')
# The endings of the selections COMSOL derives from a geometry feature: by
# level (`sel.result`, `sel.cumulative`) or layer (`sel.layer`)
DERIVED = r'dom|bnd|edg|pnt|core|layer\d+'


def restricted_selection(geom: Node, entity: str, level: int, selection,
                         restriction: Restriction) -> str | None:
    """
    Returns the tag of the selection node to evaluate on each value's
    geometry, or `None` for all entities; raises for entity numbers and
    selections that may pick other entities per value.
    """
    if selection is None:
        return None
    if isinstance(selection, Node):
        java = _comsol.check_selection(geom, selection)
        if [int(d) for d in java.dimension()] != [level]:
            raise ValueError(f'Selection "{selection}" is not a {entity} '
                             'selection.')
        tag = str(java.tag())
        why = blocked(geom, tag)
        if why:
            raise ValueError(
                f'Selection "{selection}" may pick other entities for some '
                f'values of {restriction.where}: {why}. {_instead(entity)}')
        return tag
    if restriction.kind == 'unknown':
        raise ValueError(
            f'The solution of {restriction.where} covers part of the '
            'geometry only (no physics on the rest), so whether it was '
            'solved on the geometry as built cannot be checked, nor what '
            'entity numbers stand for: pass a selection node, or None for '
            'all. For numbers and pictures, mesh all domains with a mesh '
            'of your own (not physics-controlled) and solve again.')
    matching = restriction.matching()
    numbered = (f'for {", ".join(matching)}, solved on the geometry as '
                'built' if matching else
                'for a value solved on the geometry as built (none is now)')
    raise ValueError(
        f'The sweep of {restriction.where} changes the geometry, so entity '
        f'numbers stand for other entities in some values. '
        f'{_instead(entity)} Numbers work one value at a time, {numbered}. '
        'For pictures of one value, set its parameters, run '
        'model.build(geom) and model.mesh(), and solve a study without the '
        'sweep.')


def _instead(entity: str) -> str:
    """Says which selections follow each value of a geometry sweep."""
    return (f'Pass a selection node or None: e.g. mk.sel.box(geom, '
            f"{entity!r}, x='W') follows each value of W when its range is "
            'given by parameters; boxes, balls, cylinders and disks, their '
            'unions, intersections, differences, complements and adjacent '
            'selections, and sel.result, sel.layer and sel.cumulative of '
            'features that are no selections do too.')


def blocked(geom: Node, tag: str, seen: tuple[str, ...] = ()) -> str | None:
    """
    Returns why the selection `tag` may pick other entities for values of
    a sweep that changes the geometry, or `None` if COMSOL evaluates it on
    each value's geometry.
    """
    if tag in seen:
        return None
    seen = (*seen, tag)
    component = _comsol.component_of(geom)
    if tag in [str(t) for t in component.selection().tags()]:
        java = component.selection(tag)
        kind = str(java.getType())
        name = f'"{java.label()}"'
        if kind == 'FromSequence':
            return _derived(geom, tag)
        if kind == 'Explicit':
            return (f'{name} is an explicit selection, a list of entity '
                    'numbers')
        if kind in POSITIONS:
            if _string(java, 'inputent') != 'selections':
                return None
            inputs = _strings(java, 'input')
        elif kind == 'Difference':
            inputs = _strings(java, 'add') + _strings(java, 'subtract')
        elif kind in COMBINED:
            inputs = _strings(java, 'input')
        else:
            return f'{name} is a {kind} selection, which mphkit cannot check'
        for inner in inputs:
            why = blocked(geom, inner, seen)
            if why:
                return (f'{name} takes "{_label(geom, inner)}" as input; '
                        f'{why}')
        return None
    return _derived(geom, tag)


def _derived(geom: Node, tag: str) -> str | None:
    """
    Works as `blocked()` for a selection that COMSOL derives from the
    geometry sequence: of a feature's result or layers, or of a
    cumulative selection, fine unless a selection feature makes them.
    """
    gtag = _comsol.tag_of(geom)
    name = f'"{_label(geom, tag)}"'
    features = _comsol.java_of(geom).feature()
    ftags = [str(t) for t in features.tags()]
    rest = tag[len(gtag) + 1:] if tag.startswith(f'{gtag}_') else ''
    unknown = f'{name} is a selection mphkit cannot check'
    geometry_side = (f'{name} is made in the geometry sequence; make it in '
                     'the component instead (where=None)')
    if rest in ftags:
        # a selection feature of the sequence, e.g. geom1_boxsel1
        return (geometry_side if _comsol.is_selection_feature(
            features.get(rest)) else unknown)
    owner, _, suffix = rest.rpartition('_')
    if not owner or not re.fullmatch(DERIVED, suffix):
        return unknown
    if owner in ftags:
        return (geometry_side if _comsol.is_selection_feature(
            features.get(owner)) else None)
    cumulative = _comsol.cumulative_tags(geom).values()
    if owner in cumulative:
        for ftag in ftags:
            feature = features.get(ftag)
            if (_string(feature, 'contributeto') == owner
                    and _comsol.is_selection_feature(feature)):
                return (f'{name} collects the selection feature "'
                        f'{feature.label()}" of the geometry sequence; '
                        'make that one in the component instead '
                        '(where=None)')
        return None
    return unknown


def check_not_empty(create, geom: Node, level: int, tag: str | None,
                    request: Request, selection):
    """
    Raises if the selection `tag` is empty in the geometry of a requested
    value: COMSOL would give 0 or nan.
    """
    if tag is None:
        return
    model = geom.model.java
    for target in request.targets:
        feature = create(model.result().numerical(),
                         'Int' + _results.LEVELS[level])
        table = create(model.result().table(), 'Table')
        feature.set('data', target.data)
        feature.selection().geom(geom.tag(), level)
        feature.selection().named(tag)
        _comsol.set_properties(feature, {
            'expr': '1', 'innerinput': 'manual',
            'solnum': [target.solnums[0]]})
        feature.set('table', str(table.tag()))
        try:
            feature.setResult()
        except Exception as error:
            if _results.unmeshed(error):
                raise RuntimeError(
                    f'Some entities of "{selection}" have no solution at '
                    f'{target.where} (no physics or mesh there); pass a '
                    'selection of the solved ones.') from error
            raise
        if not numpy.array(table.getReal(), dtype=float)[0, -1]:
            raise ValueError(
                f'Selection "{selection}" is empty at {target.where}: it '
                "picks nothing in that value's geometry. A box at fixed "
                'coordinates can miss a face that moves; one whose range '
                "is given by parameters, e.g. x='W', follows each value. "
                'Selections the geometry makes (sel.result, sel.layer, '
                'sel.cumulative) are empty in values solved before they '
                'were made: solve again.')


def _label(geom: Node, tag: str) -> str:
    """Returns the label of a selection by tag, or the tag."""
    selections = geom.model.java.selection()
    if tag in [str(t) for t in selections.tags()]:
        return str(selections.get(tag).label())
    return tag


def _strings(java, name: str) -> list[str]:
    """Returns a string array property of a Java object, or []."""
    try:
        return [str(value) for value in java.getStringArray(name)]
    except Exception:
        return []


############
# Geometry #
############

SAME, UNKNOWN, DIFFERENT = 'same', 'unknown', 'different'


class Children:
    """
    Temporary datasets of a sweep's children, made with `create`, and the
    vertices of the geometry each was solved on.
    """

    def __init__(self, create, geom: Node, sweep: Sweep, data) -> None:
        self.create = create
        self.geom = geom
        self.sweep = sweep
        self.frame = _string(data, 'frametype')
        self._datasets: dict[int, Any] = {}
        self._points: dict[int, Array | None] = {}

    def dataset(self, k: int):
        """Returns the temporary dataset of outer value `k`."""
        if k not in self._datasets:
            model = self.sweep.model
            java = self.create(model.result().dataset(), 'Solution')
            java.set('solution', self.sweep.children[k - 1])
            java.set('geom', self.geom.tag())
            if self.frame:
                java.set('frametype', self.frame)
            self._datasets[k] = java
        return self._datasets[k]

    def points(self, k: int) -> Array | None:
        """
        Returns the geometry vertices of outer value `k`, a row each, or
        `None` where its mesh leaves parts of the geometry out.
        """
        if k not in self._points:
            self._points[k] = fingerprint(
                self.create, self.geom, self.dataset(k), self.sweep.count(k),
                self.sweep.labels[k - 1])
        return self._points[k]

    def others(self) -> str:
        """
        Compares all outer values' vertices: `'single'` for one value,
        `'same'` if all are the same, `'differ'` otherwise (also if some
        cannot be read).
        """
        count = len(self.sweep.children)
        if count == 1:
            return 'single'
        try:
            first = self.points(1)
            if first is None:
                return 'differ'
            for k in range(2, count + 1):
                points = self.points(k)
                if points is None or not same_points(first, points):
                    return 'differ'
        except RuntimeError:
            # COMSOL could not read a value's geometry: not the same
            return 'differ'
        return 'same'


def fingerprint(create, geom: Node, dataset, count: int, label: str
                ) -> Array | None:
    """
    Returns the vertices of the geometry a solution was solved on, a row
    each in the geometry's length unit, evaluated at all points of its
    mesh in the geometry frame (which a deformation does not move).
    `None` if the mesh does not cover the whole geometry.
    """
    model = geom.model.java
    java = _comsol.java_of(geom)
    sdim = _comsol.sdim(geom)
    names = [str(name) for name in
             _comsol.component_of(geom).geometryCoord()]
    exprs = ([names[0], names[2]] if sdim == 2 and _results.axisymmetric(geom)
             else names[:sdim])
    feature = create(model.result().numerical(), 'EvalPoint')
    table = create(model.result().table(), 'Table')
    feature.set('data', str(dataset.tag()))
    feature.selection().geom(geom.tag(), 0)
    feature.selection().all()
    _comsol.set_property(feature, 'expr', exprs)
    _comsol.set_property(feature, 'unit', [str(java.lengthUnit())]
                         * len(exprs))
    if count > 1:
        _comsol.set_property(feature, 'innerinput', 'first')
    feature.set('table', str(table.tag()))
    try:
        feature.setResult()
    except Exception as error:
        if _results.unmeshed(error):
            return None
        raise RuntimeError(f'COMSOL could not read the geometry of the '
                           f'solution at {label}: '
                           f'{_comsol.reason(error)}') from error
    headers = [str(header) for header in table.getColumnHeaders()]
    columns = [i for i, header in enumerate(headers) if 'Point:' in header]
    real = numpy.array(table.getReal(), dtype=float)
    values = real[0, columns].reshape(len(exprs), -1).T
    return None if numpy.isnan(values).any() else values


def vertices(geom: Node) -> Array:
    """Returns the vertices of the built geometry, a row each."""
    java = _comsol.java_of(geom)
    found = numpy.array(java.getVertexCoord(), dtype=float)
    return found.reshape(_comsol.sdim(geom), -1).T


def same_points(a: Array, b: Array) -> bool:
    """
    Tells whether two sets of points are the same, in any order, within
    1e-12 of their largest coordinate.
    """
    if a.shape != b.shape:
        return False
    if not len(a):
        return True
    tolerance = COORDINATES * max(float(numpy.abs(a).max()),
                                  float(numpy.abs(b).max()), 1e-300)
    # sorted rows first; points within the tolerance can sort apart
    first = a[numpy.lexsort(a.T[::-1])]
    second = b[numpy.lexsort(b.T[::-1])]
    if numpy.all(numpy.abs(first - second) <= tolerance):
        return True
    # one to one: a point matched once cannot match again
    free = numpy.ones(len(b), dtype=bool)
    for row in a:
        near = numpy.flatnonzero(free & numpy.all(numpy.abs(b - row)
                                                  <= tolerance, axis=1))
        if not len(near):
            return False
        free[near[0]] = False
    return True


def decide(built: bool, requested: list[str], others: Callable[[], str]
           ) -> str:
    """
    Decides what can be read of a sweep from the requested values'
    verdicts (`SAME`, `UNKNOWN` or `DIFFERENT` from the built geometry;
    without it, `UNKNOWN` or `DIFFERENT`) and `others()`, all values
    compared with each other (`'single'`, `'same'`, `'differ'`):

    - `'full'`: the geometry is the one solved; numbers and pictures.
    - `'unknown'`: a mesh leaves parts out; named selections only.
    - `'sweep'`: the sweep changes the geometry; named selections only.
    - `'stale'`, `'stale-single'`: the geometry changed since the solve.
    - `'unbuilt'`, `'unbuilt-single'`: the geometry is not built.
    """
    if built and requested and all(v == SAME for v in requested):
        return 'full'
    if UNKNOWN in requested:
        return 'unknown'
    kind = others()
    if kind == 'differ':
        return 'sweep'
    if built:
        return 'stale' if kind == 'same' else 'stale-single'
    return 'unbuilt' if kind == 'same' else 'unbuilt-single'


def check(geom: Node, sweep: Sweep, children: Children, unique: list[int],
          data) -> str:
    """
    Checks that the requested outer values were solved on the geometry
    as built, and returns the decision (see `decide()`): `'full'`,
    `'sweep'` or `'unknown'`; raises for the rest.
    """
    if sweep.materials():
        _results.check_current(geom)
        return 'full'
    try:
        _comsol.check_built(geom)
        built = True
    except RuntimeError:
        built = False
    current = vertices(geom) if built else None
    verdicts = []
    for k in unique:
        points = children.points(k)
        if points is None:
            verdicts.append(UNKNOWN)
        elif current is not None and same_points(points, current):
            verdicts.append(SAME)
        else:
            verdicts.append(DIFFERENT)
    decision = decide(built, verdicts, children.others)
    if decision == 'full':
        return decision
    where = f'dataset {_datasets.describe(data)}'
    if decision in ('stale', 'stale-single'):
        k = unique[verdicts.index(DIFFERENT)]
        points = children.points(k)
        counts = ('' if points is None or current is None
                  or len(points) == len(current) else
                  f' ({len(current)} vertices now, {len(points)} in the '
                  'solution)')
        if decision == 'stale':
            raise RuntimeError(f'Geometry "{geom}" changed since the solve'
                               f'{counts}; run model.build(geom), '
                               'model.mesh() and model.solve().')
        raise RuntimeError(
            f'Geometry "{geom}" differs from the one {where} was solved '
            f'on{counts}: it changed since the solve, or the sweep changes '
            'the geometry and keeps the last value only. Run '
            'model.build(geom), model.mesh() and model.solve(); a sweep '
            'that changes the geometry must keep all solutions to be read.')
    if decision == 'unbuilt':
        raise RuntimeError(f'Geometry "{geom}" is not built; run '
                           'model.build(geom), and if it changed since the '
                           'solve, model.mesh() and model.solve() too.')
    if decision == 'unbuilt-single':
        raise RuntimeError(
            f'Geometry "{geom}" is not built; set the parameters to the '
            f"sweep's value ({sweep.labels[-1]}, see mk.outer_values(geom)) "
            'and run model.build(geom).')
    return decision


#########
# Steps #
#########

def named_steps(model, solution) -> dict[str, Array]:
    """
    Returns the step values of a solution by name, in SI units; adds
    `'freq'` in Hz to the eigenvalues of an eigenfrequency study.
    """
    found = _step_table(solution)
    if list(found) == ['lambda'] and _eigenfrequency(model, solution):
        freq = 1j*found['lambda']/(2*math.pi)
        found['freq'] = _real_if_exact(freq)
    return {name: _real_if_exact(values) for name, values in found.items()}


def same_steps(model, children: list[str], strict: bool = False) -> bool:
    """
    Tells whether the given children have the same steps: names, number
    and values within 1e-9 of each name's largest. Eigenvalues differ
    between values by nature: their number alone counts, unless `strict`.
    """
    tables = []
    for child in children:
        solution = model.sol(child)
        tables.append((_names(solution), _count(solution), solution))
    names, count, _ = tables[0]
    if any(other[:2] != (names, count) for other in tables[1:]):
        return False
    if names == ['lambda']:
        return not strict or len(tables) == 1
    if names == ['']:
        return True
    values = [_step_table(solution) for *_, solution in tables]
    for name in names:
        stack = numpy.array([table[name] for table in values])
        tolerance = TOLERANCE * float(numpy.abs(stack).max())
        if numpy.any(numpy.abs(stack - stack[0]) > tolerance):
            return False
    return True


def _step_table(solution) -> dict[str, Array]:
    """
    Returns the step values of a solution by name, complex: COMSOL gives
    them interleaved, step by step.
    """
    names = _names(solution)
    if names == ['']:
        return {}
    count = _count(solution)
    real = numpy.array(solution.getPVals() or [], dtype=float)
    try:
        imag = numpy.array(solution.getPValsImag() or [], dtype=float)
    except Exception:
        imag = numpy.zeros(0)
    if not imag.size:
        imag = numpy.zeros_like(real)
    if real.size != len(names)*count or imag.size != real.size:
        raise RuntimeError(f'COMSOL gave {real.size} step values for '
                           f'{len(names)} names and {count} steps.')
    values = (real + 1j*imag).reshape(count, len(names))
    return {name: values[:, i] for i, name in enumerate(names)}


def _real_if_exact(values: Array) -> Array:
    """Returns real values if all imaginary parts are exactly 0."""
    return values.real.copy() if not numpy.any(values.imag) else values


def _eigenfrequency(model, solution) -> bool:
    """
    Tells whether a solution's study finds eigenfrequencies (an active
    Eigenfrequency step and no Eigenvalue step), whose eigenvalue is
    λ = -iω.
    """
    try:
        study = model.study(_datasets.study_of(solution))
        steps = [study.feature(tag) for tag in study.feature().tags()]
        active = [str(step.getType()) for step in steps if step.isActive()]
    except Exception:
        return False
    return 'Eigenfrequency' in active and 'Eigenvalue' not in active


def _names(solution) -> list[str]:
    """Returns the names of a solution's steps; `['']` if stationary."""
    return [str(name) for name in solution.getPNames()]


def _count(solution) -> int:
    """Returns the number of steps of a solution."""
    return len(solution.getSolutioninfo().getSolnum(1, True))


def _differences(model, sweep: Sweep, unique: list[int] | None = None
                 ) -> str:
    """Says how the steps of the outer values differ, for messages."""
    unique = unique or list(range(1, len(sweep.children) + 1))
    counts = [sweep.count(k) for k in unique]
    if len(set(counts)) > 1:
        shown = [f'outer={k}: {count} steps'
                 for k, count in zip(unique, counts)]
        first, last = SHOWN
        if len(shown) > first + last + 1:
            shown = shown[:first] + ['...'] + shown[-last:]
        return ', '.join(shown)
    return f'{counts[0]} steps each, at other times or values'


def _preview(model, child: str) -> str:
    """The step values of a child for messages, shortened."""
    table = named_steps(model, model.sol(child))
    if not table:
        return 'one stationary step'
    parts = []
    for name, values in table.items():
        shown = [f'{value:g}' for value in values]
        first, last = SHOWN
        if len(shown) > first + last + 1:
            shown = shown[:first] + ['...'] + shown[-last:]
        parts.append(f'{name} = {", ".join(shown)}')
    return '; '.join(parts)


def _numbers(values: list[int]) -> str:
    """Lists numbers for messages: `1, 2 and 3`."""
    shown = [str(n) for n in values]
    return shown[0] if len(shown) == 1 else \
        f'{", ".join(shown[:-1])} and {shown[-1]}'


def _call(name: str, dataset) -> str:
    """Spells a call of `mk.<name>` with the dataset passed, if any."""
    if dataset is None:
        return f'mk.{name}(geom)'
    if isinstance(dataset, Node):
        return f'mk.{name}(geom, dataset={_comsol.tag_of(dataset)!r})'
    return f'mk.{name}(geom, dataset={dataset!r})'


def _string(java, name: str) -> str | None:
    """Returns a string property of a Java object, or `None`."""
    try:
        return str(java.getString(name))
    except Exception:
        return None
