"""
Datasets of solved models: finding the one to evaluate, and temporary
features for evaluating it.

COMSOL evaluates the first dataset of the model when none is given,
whatever its geometry, so the helpers pick the solved dataset of the
geometry themselves and raise when there are several.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple

from mph.node import Node

from . import _comsol, _sweep


class Chosen(NamedTuple):
    """A dataset to evaluate and its kind."""
    java: Any
    kind: str  # 'sweep' (an outer loop) or 'plain'


def select(geom: Node, dataset) -> Chosen:
    """
    Returns the solution dataset to evaluate: `dataset` (a dataset, or a
    study with one dataset for the geometry; by node, MPh name, label or
    tag) or, without it, the only solved dataset of the geometry. COMSOL
    would take the first dataset of the model, whatever its geometry.

    Refuses the datasets of a study whose last solve failed, and those
    holding one value of an outer sweep: the sweep's solution of a value,
    or the copy of the last value that COMSOL makes.
    """
    model = geom.model.java
    solutions = Solutions(model)
    if dataset is None:
        solved = _solved_datasets(model)
        own = [java for java in solved if geometry_tag(java) == geom.tag()]
        if not own:
            if solved:
                raise RuntimeError(f'No solved dataset for geometry '
                                   f'"{geom}"; solve a study that includes '
                                   'it.')
            raise RuntimeError('No solved dataset; run model.solve() first.')
        return _only(geom, solutions, own, None)
    java, what = _lookup(geom, dataset)
    if what == 'study':
        own = [found for found in _solved_datasets(model)
               if geometry_tag(found) == geom.tag()
               and study_of(solution_of(model, found)) == str(java.tag())]
        if not own:
            raise ValueError(f'Study "{_comsol.name_of(java)}" has no solved '
                             f'dataset for geometry "{geom}"; solve it, or '
                             'pass a study that includes the geometry.')
        return _only(geom, solutions, own, java)
    name = _comsol.name_of(java)
    kind = str(java.getType())
    if kind != 'Solution':
        raise ValueError(f'Dataset "{name}" is a {kind} dataset; pass a '
                         'solution dataset.')
    owner = geometry_tag(java)
    if owner != geom.tag():
        other = f', but to {owner}' if owner else ''
        raise ValueError(f'Dataset "{name}" does not belong to geometry '
                         f'"{geom}" ({geom.tag()}){other}.')
    if not is_solved(model, java):
        raise RuntimeError(f'Dataset "{name}" has no solution; run '
                           'model.solve() first.')
    return _usable(geom, solutions, java, ValueError)


def _solved_datasets(model) -> list:
    """Returns the model's solution datasets that hold values."""
    datasets = model.result().dataset()
    return [datasets.get(tag) for tag in datasets.tags()
            if str(datasets.get(tag).getType()) == 'Solution'
            and is_solved(model, datasets.get(tag))]


def _lookup(geom: Node, dataset) -> tuple[Any, str]:
    """
    Returns the Java dataset or study given by node, MPh name, label or
    tag, and `'dataset'` or `'study'`; datasets come first.
    """
    model = geom.model.java
    wrong = (f'dataset must be a name, tag or node of a dataset or study, '
             f'not {dataset!r}.')
    if (isinstance(dataset, Node) and len(dataset.path) == 2
            and dataset.path[0] == 'studies'):
        return find(model.study(), dataset, 'study', 'studies', wrong), \
            'study'
    try:
        return find(model.result().dataset(), dataset, 'dataset',
                    'datasets', wrong), 'dataset'
    except LookupError as error:
        if not isinstance(dataset, str):
            raise
        try:
            return find(model.study(), dataset, 'study', 'studies', wrong), \
                'study'
        except LookupError:
            studies = [repr(_comsol.name_of(model.study(tag)))
                       for tag in model.study().tags()]
            raise LookupError(f'{error} No study "{dataset}" either; the '
                              f'model has {", ".join(studies) or "none"}.'
                              ) from None


def _only(geom: Node, solutions: Solutions, own: list, study) -> Chosen:
    """
    Returns the only dataset among `own` (the solved datasets of the
    geometry, of `study` if given) that holds a whole solution.
    """
    model = solutions.model
    for java in own:
        failure = solutions.failure(str(java.getString('solution')))
        if failure:
            others = [describe(other) for other in own
                      if not solutions.failure(
                          str(other.getString('solution')))]
            if others:
                failure += (f' Other solved datasets: {", ".join(others)}; '
                            'pass dataset= one of these.')
            raise RuntimeError(failure)
    usable = [java for java in own
              if solutions.kind(str(java.getString('solution')))
              in ('sweep', 'plain')]
    if not usable:
        return _usable(geom, solutions, own[0], RuntimeError)
    if len(usable) == 1:
        return _usable(geom, solutions, usable[0], RuntimeError)
    names = ', '.join(describe(java) for java in usable)
    where = (f'Study "{_comsol.name_of(study)}"' if study is not None
             else f'Geometry "{geom}"')
    studies = [study_of(solution_of(model, java)) for java in usable]
    message = (f'{where} has several solved datasets: {names}; pass '
               'dataset= one of these')
    if study is None and len(set(studies)) == len(studies):
        message += (", or the study, e.g. dataset="
                    f'{_study_name(model, studies[0])!r}')
    message += '.'
    tags = [str(java.getString('solution')) for java in usable]
    if len(set(tags)) < len(tags):
        message += (' Some of them show the same solution (a dataset '
                    'copied, e.g. for a selection of its own); pass the one '
                    'you mean.')
    elif len(set(studies)) < len(studies):
        sweeps = [java for java, tag in zip(usable, tags)
                  if solutions.kind(tag) == 'sweep']
        message += (' Some come from the same study, solved again after '
                    'its steps changed')
        if sweeps and len(sweeps) < len(usable):
            old = sweeps[0]
            message += (f'; {describe(old)} holds a sweep of an earlier '
                        'solve: to drop it, remove its solution, '
                        "model.java.sol().remove("
                        f"{str(old.getString('solution'))!r})")
        message += '.'
    raise ValueError(message)


def _usable(geom: Node, solutions: Solutions, java, error: type) -> Chosen:
    """
    Returns `java` as the dataset to evaluate, or raises if it holds a
    failed solve, one value of a sweep or an unfinished sweep. `error` is
    the exception for a dataset of the wrong kind.
    """
    model = solutions.model
    tag = str(java.getString('solution'))
    failure = solutions.failure(tag)
    if failure:
        raise RuntimeError(failure)
    kind = solutions.kind(tag)
    if kind == 'child':
        raise error(_child_message(geom, solutions, java, tag))
    if kind == 'copy':
        raise error(_copy_message(geom, solutions, java, tag))
    if kind == 'sweep':
        values, held = solutions.sweeps[tag], solutions.children[tag]
        if len(values) != len(held):
            study = _study_name(model, study_of(model.sol(tag)))
            raise RuntimeError(f'Dataset {describe(java)} holds an '
                               f'unfinished sweep ({len(held)} solutions '
                               f'for {len(values)} values); run '
                               f'model.solve({study!r}) again.')
    return Chosen(java, kind)


def _child_message(geom: Node, solutions: Solutions, java, tag: str) -> str:
    """The error for a dataset of one value of a sweep."""
    sweep = solutions.parent[tag]
    number = solutions.children[sweep].index(tag) + 1
    label = _sweep.labels(solutions.model, sweep)[number - 1]
    return (f'Dataset {describe(java)} holds one value of a parametric '
            f'sweep ({label}); {_pass(geom, solutions, sweep)} with '
            f'outer={number}.')


def _copy_message(geom: Node, solutions: Solutions, java, tag: str) -> str:
    """The error for a dataset of the copy of a sweep's last value."""
    model = solutions.model
    copy = solutions.copies[tag]
    study = study_of(model.sol(copy))
    sweeps = [sweep for sweep in solutions.sweeps
              if study_of(model.sol(sweep)) == study]
    if not sweeps:
        return (f'Dataset {describe(java)} holds one value of a parametric '
                f'sweep whose other values are gone ('
                f'{_sweep.label_of(model, copy)}); run model.solve('
                f'{_study_name(model, study)!r}) again for all of them.')
    sweep = sweeps[-1]
    label = _sweep.labels(model, sweep)[-1]
    advice = f"{_pass(geom, solutions, sweep)} with outer='last' ({label})"
    if copy != tag:
        return (f'Dataset {describe(java)} holds the first study step of '
                f"the sweep's last value only: the sweep keeps the last "
                'study step of each value, not this one. For the last '
                f'step, {advice}.')
    return (f'Dataset {describe(java)} holds only the last value of a '
            f'parametric sweep; {advice}.')


def _pass(geom: Node, solutions: Solutions, sweep: str) -> str:
    """Says which dataset to pass for the sweep in solution `sweep`."""
    found = [java for java in _solved_datasets(solutions.model)
             if geometry_tag(java) == geom.tag()
             and str(java.getString('solution')) == sweep]
    if found:
        return (f'pass dataset={str(found[0].tag())!r}, '
                f'{describe(found[0])},')
    # by Java: COMSOL renames the dataset, which loses an MPh node
    return (f'its solution {sweep} has no dataset for geometry "{geom}": '
            f"make one, ds = mk.set((model/'datasets').create('Solution')"
            f'.java, solution={sweep!r}, geom={geom.tag()!r}), and pass '
            'dataset=str(ds.tag())')

def find(container, value, what: str, group: str, wrong: str):
    """
    Returns the Java object in `container` given by a node of the MPh
    group `group`, an MPh name, a label or a tag. `wrong` is the message
    for a value of another type; `what` names the object in messages.
    """
    if isinstance(value, Node):
        if len(value.path) != 2 or value.path[0] != group:
            raise TypeError(f'"{value}" is not a {what} node.')
        key = _comsol.tag_of(value)
    elif isinstance(value, str):
        key = value
    else:
        raise TypeError(wrong)
    for tag in container.tags():
        java = container.get(tag)
        if key in (str(tag), _comsol.name_of(java), str(java.label())):
            return java
    known = [_comsol.name_of(container.get(tag)) for tag in container.tags()]
    raise LookupError(f'No {what} "{key}"; the model has '
                      f'{", ".join(repr(n) for n in known) or "none"}.')


def geometry_tag(dataset) -> str | None:
    """Returns the tag of the geometry a Java dataset belongs to."""
    try:
        return str(dataset.getString('geom'))
    except Exception:
        return None


def solution_of(model, dataset):
    """Returns the Java solution of a dataset, or `None`."""
    try:
        tag = str(dataset.getString('solution'))
    except Exception:
        return None
    if tag not in [str(t) for t in model.sol().tags()]:
        return None
    return model.sol(tag)


def is_solved(model, dataset) -> bool:
    """Tells whether a dataset refers to a solution that holds values."""
    solution = solution_of(model, dataset)
    return solution is not None and not solution.isEmpty()


class Solutions:
    """
    The model's solutions, sorted: outer sweeps, their solutions per value
    (children), copies of a sweep's last value, and failed studies.

    An outer sweep's solution has outer solution numbers, and a
    StoreSolution feature per value that names the child solution. The
    copy, "Solution N", holds the last value and carries its parameter
    names, as the sweep and its children do; a plain solution carries
    none. A study whose last solve failed keeps the error on one of its
    solutions, until it is solved again.
    """

    def __init__(self, model) -> None:
        self.model = model
        tags = [str(tag) for tag in model.sol().tags()]
        self.sweeps: dict[str, list[int]] = {}
        self.children: dict[str, list[str]] = {}
        self.parent: dict[str, str] = {}
        for tag in tags:
            numbers = outer_numbers(model.sol(tag))
            if numbers:
                self.sweeps[tag] = numbers
                self.children[tag] = stored(model.sol(tag))
                for child in self.children[tag]:
                    self.parent[child] = tag
        # copy, or the solution of an earlier study step it stores → copy
        self.copies: dict[str, str] = {}
        for tag in tags:
            if (tag not in self.sweeps and tag not in self.parent
                    and param_names(model.sol(tag))):
                self.copies[tag] = tag
        for copy in list(self.copies):
            for store in stored(model.sol(copy)):
                if (store in tags and store not in self.sweeps
                        and store not in self.parent):
                    self.copies.setdefault(store, copy)
        # study → the solution with the error, and the error
        self.failed: dict[str, tuple[str, str]] = {}
        for tag in tags:
            message = error_of(model.sol(tag))
            if message is not None:
                self.failed.setdefault(study_of(model.sol(tag)),
                                       (tag, message))

    def kind(self, tag: str) -> str:
        """Returns `'sweep'`, `'child'`, `'copy'` or `'plain'`."""
        if tag in self.sweeps:
            return 'sweep'
        if tag in self.parent:
            return 'child'
        if tag in self.copies:
            return 'copy'
        return 'plain'

    def failure(self, tag: str) -> str | None:
        """The error if the study of solution `tag` failed, else `None`."""
        study = study_of(self.model.sol(tag))
        if study not in self.failed:
            return None
        where, message = self.failed[study]
        first = next((line.strip() for line in message.splitlines()
                      if line.strip()), message)
        name = _study_name(self.model, study)
        return (f'The last solve of study "{name}" failed: {first} '
                f'(solution {where}); fix it and run model.solve({name!r}). '
                "MPh's model.evaluate reads the values it left, unchecked.")


def outer_numbers(solution) -> list[int]:
    """Returns the outer solution numbers of a sweep's solution, else []."""
    try:
        return [int(n) for n in solution.getSolutioninfo().getOuterSolnum()]
    except Exception:
        return []


def stored(solution) -> list[str]:
    """
    Returns the solutions that the StoreSolution features of a solution
    name, in order: a sweep's children, or a study step's solution.
    """
    found = []
    for tag in solution.feature().tags():
        feature = solution.feature(tag)
        try:
            if str(feature.getType()) == 'StoreSolution':
                found.append(str(feature.getString('sol')))
        except Exception:
            continue
    return found


def param_names(solution) -> list[str]:
    """Returns the names of the sweep parameters a solution was made for."""
    try:
        return [str(name) for name in solution.getParamNames()]
    except Exception:
        return []


def error_of(solution) -> str | None:
    """Returns the error a solution's last solve ended with, or `None`."""
    try:
        message = solution.getErrorMessage()
    except Exception:
        return None
    return str(message) if message is not None and str(message).strip() \
        else None


def study_of(solution) -> str:
    """Returns the tag of the study a solution belongs to, or `''`."""
    try:
        return str(solution.study())
    except Exception:
        return ''


def _study_name(model, study: str) -> str:
    """Returns the MPh name of a study by tag, or the tag."""
    if study in [str(tag) for tag in model.study().tags()]:
        return _comsol.name_of(model.study(study))
    return study


def describe(dataset) -> str:
    """Names a Java dataset in messages: `"name" (tag)`."""
    return f'"{_comsol.name_of(dataset)}" ({dataset.tag()})'


def step_count(model, dataset) -> int:
    """Returns the number of steps (inner solutions) of a dataset."""
    solution = solution_of(model, dataset)
    if solution is None:
        raise LookupError(f'Dataset "{_comsol.name_of(dataset)}" has no '
                          'solution.')
    return len(solution.getSolutioninfo().getSolnum(1, True))


@contextmanager
def scratch(model) -> Iterator[Callable[[Any, str], Any]]:
    """
    Yields a function that creates temporary features, removed afterwards
    together with everything else created, with the history switched off.
    """
    made: list[tuple[Any, str]] = []

    def create(container, kind: str):
        tag = str(container.uniquetag('mk'))
        container.create(tag, kind)
        made.append((container, tag))
        return container.get(tag)

    with _comsol.history_off(model):
        try:
            yield create
        finally:
            for container, tag in reversed(made):
                try:
                    if tag in [str(t) for t in container.tags()]:
                        container.remove(tag)
                except Exception:
                    pass
