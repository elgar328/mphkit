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
from typing import Any

from mph.node import Node

from . import _comsol



def solved_dataset(geom: Node, dataset):
    """
    Returns the Java solution dataset to evaluate.

    Without `dataset`, the only solved dataset of the geometry: COMSOL
    would take the first dataset of the model, whatever its geometry.
    """
    model = geom.model.java
    datasets = model.result().dataset()
    if dataset is not None:
        java = find_dataset(geom, dataset)
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
        _check_outer(model, [java])
        return java
    solved = [datasets.get(tag) for tag in datasets.tags()
              if str(datasets.get(tag).getType()) == 'Solution'
              and is_solved(model, datasets.get(tag))]
    own = [java for java in solved if geometry_tag(java) == geom.tag()]
    if not own:
        if solved:
            raise RuntimeError(f'No solved dataset for geometry "{geom}"; '
                               'solve a study that includes it.')
        raise RuntimeError('No solved dataset; run model.solve() first.')
    _check_outer(model, own)
    if len(own) > 1:
        names = ', '.join(f'"{_comsol.name_of(java)}" ({java.tag()})'
                          for java in own)
        raise ValueError(f'Geometry "{geom}" has several solved datasets: '
                         f'{names}; pass dataset= one of these.')
    return own[0]


def find_dataset(geom: Node, dataset):
    """Returns the Java dataset given by node, MPh name, label or tag."""
    return find(geom.model.java.result().dataset(), dataset, 'dataset',
                'datasets', f'dataset must be a name, tag or node, not '
                f'{dataset!r}.')


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


def _check_outer(model, datasets: list):
    """
    Raises for studies with an outer loop, such as a parametric sweep of a
    time-dependent study: one of their datasets holds only the last
    parameter, the other all of them in a way `step` cannot address.
    """
    studies = set()
    for dataset in datasets:
        solution = solution_of(model, dataset)
        if solution is None:
            continue
        try:
            studies.add(str(solution.study()))
        except Exception:
            pass
    for tag in model.sol().tags():
        solution = model.sol(tag)
        try:
            study = str(solution.study())
            outer = len(solution.getSolutioninfo().getOuterSolnum())
        except Exception:
            continue
        if study in studies and outer > 1:
            label = str(model.study(study).label())
            raise NotImplementedError(f'Parametric sweeps with an outer loop '
                                      f'are not supported yet ("{label}").')


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
