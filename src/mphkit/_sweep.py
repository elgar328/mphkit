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

from mph.node import Node

from . import _comsol, _datasets, _results

# Inner step names that are no model parameters: stationary, time,
# frequency and eigenvalue steps
STEP_NAMES = ('', 't', 'freq', 'lambda')
# Prefixes of the outer names of material and function sweeps
SWITCHES = ('matsw.', 'funsw.')
# Values shown in messages: the first ones, then the last ones
SHOWN = (5, 2)


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

