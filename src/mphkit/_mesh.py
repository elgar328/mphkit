"""
The quality of a mesh, in numbers. Public as `mk.mesh_quality`.

COMSOL's mesh statistics object would give some of these numbers, but its
quality measure defaults to another one than the COMSOL Desktop's, and
changing it is recorded in the model. Instead, the quality of each element
is evaluated at the element's centroid on a temporary mesh dataset, which
gives the same numbers for every element type, prisms and pyramids
included, and leaves nothing in the model.
"""
from __future__ import annotations

from typing import Any

import numpy
from mph.node import Node

from . import _comsol, _measure

# Quality measures, by COMSOL's names, and their mesh variables
MEASURES = {'skewness': 'qualskewness', 'maxangle': 'qualmaxangle',
            'volcircum': 'qual', 'vollength': 'qualvollength',
            'condition': 'qualcondition', 'growth': 'qualgrowth'}
# Element types of volumes, and of faces (3D boundaries, 2D domains)
VOLUMES = ('tet', 'pyr', 'prism', 'hex')
FACES = ('tri', 'quad')
WORST = 5
BINS = 10


def mesh_quality(geom: Node, entity: str = 'domain', /, selection=None, *,
                 mesh: Node | str | None = None,
                 quality: str = 'skewness') -> dict:
    """
    Returns the quality of a mesh in numbers, for all domains, some
    domains or (in 3D) boundaries:

    ```python
    mk.mesh_quality(geom)
    # {'mesh': 'mesh', 'quality': 'skewness', 'level': 'domain',
    #  'elements': {'tet': 6267, 'pyr': 5, 'prism': 1093, 'hex': 35},
    #  'min': 0.045, 'mean': 0.62,
    #  'worst': [{'quality': 0.045, 'position': (0.98, 1.0, 0.94),
    #             'entity': 1}, ...],
    #  'histogram': [6, 60, 208, 403, 708, 1047, 1619, 1906, 1114, 329],
    #  'size': {'min': 0.116, 'max': 0.394},
    #  'by_entity': {1: {'elements': 6804, 'min': 0.045, 'mean': 0.61},
    #                2: {'elements': 596, 'min': 0.327, 'mean': 0.66}},
    #  'unmeshed': [],
    #  'messages': [{'node': 'meshes/mesh/Boundary Layers 1',
    #                'severity': 'info', 'message': 'To avoid low quality '
    #                'elements the thickness was reduced locally. ...',
    #                'level': 'boundary', 'entities': [6, 7]}]}
    mk.mesh_quality(geom, 'domain', 1)        # one domain
    mk.mesh_quality(geom, 'boundary', [3, 4]) # surface elements in 3D
    ```

    `entity` is `'domain'` or, in 3D, `'boundary'`; `selection` is an
    entity number, a list of them, a selection node at that level, or
    `None` for all. `mesh` is needed when the geometry's component has
    several non-empty meshes (name as in `model.meshes()`, tag or node;
    `None` or `True` for the only one, as in `mk.image`). Quality
    runs from 0 to 1, 1 being best. `quality` names COMSOL's measure:
    `'skewness'`
    (the default, also in the COMSOL Desktop and its mesh messages),
    `'maxangle'`, `'volcircum'`, `'vollength'`, `'condition'` or
    `'growth'` (the size change to neighbouring elements, not the shape).
    The shape measures differ, e.g. 0.065 to 0.097 for the worst
    element of the example's plate. Long thin boundary-layer elements rate
    high by skewness and maximum angle, low by the volume measures.

    The result has the measure (`'quality'`), the element count by type,
    the lowest and the mean quality (each element counted once), the
    `worst` elements (up to five, lowest first, each with its number
    under `'quality'`, the centroid in the geometry's length unit and the
    entity number), a `histogram` of the counts with quality in
    0-0.1, 0.1-0.2, ..., 0.9-1 (0.3 counts in 0.3-0.4, 1 in the last),
    the element `size` (COMSOL's `h`, the longest edge of an element, in
    the geometry's length unit; not a boundary-layer thickness), each
    entity's count, lowest and mean quality (`by_entity`), the selected
    entities without elements (`unmeshed`), and the information, warnings
    and errors COMSOL gave when building the mesh (`messages`, with the
    level and entities they point to, for the whole mesh). With no
    elements, `min`, `mean` and `size` are `None`.

    Low-quality elements can keep a solver from converging. Their cause
    is often what `messages` points to: short edges, small faces, thin
    regions. `worst` and `by_entity` show where they are; `mk.image(geom,
    'mesh.png', selection, mesh=True)` draws them. `unmeshed` entities
    may be on purpose; with errors in `messages`, building the mesh
    failed there, and solving stops with that error (errors whose
    entities a later feature meshed do not stop it). Only straight
    elements are rated: the solver reports inverted curved elements
    itself (`model.problems()`). Not tried: warnings, imported meshes,
    virtual operations. Judges nothing; returns plain values and leaves
    nothing in the model.
    """
    _comsol.check_not_workplane(geom, 'mk.mesh_quality')
    if not isinstance(entity, str):
        example = 'selection' if isinstance(entity, Node) else repr(entity)
        raise TypeError(f"entity must be 'domain' or 'boundary', not "
                        f"{entity!r}; the selection comes third, e.g. "
                        f"mk.mesh_quality(geom, 'domain', {example}).")
    if quality not in MEASURES:
        raise ValueError(f'quality must be one of {list(MEASURES)}, not '
                         f'{quality!r}.')
    if mesh is True:
        mesh = None
    if mesh is not None and not isinstance(mesh, (Node, str)):
        raise TypeError(f'mesh must be a mesh name, tag or node, or None, '
                        f'not {mesh!r}.')
    sdim = _comsol.sdim(geom)
    if sdim < 2:
        raise ValueError(f'Mesh quality is for 2D and 3D geometries; '
                         f'"{geom}" is 1D.')
    level = _comsol.entity_dim(geom, entity)
    if level not in ((3, 2) if sdim == 3 else (2,)):
        raise ValueError("Mesh quality is for 'domain' or, in 3D, "
                         f"'boundary' entities, not {entity!r}: edge and "
                         'point elements have no quality.')
    entities = sorted(set(_measure.numbers_of(geom, entity, selection)))
    if level == sdim and _comsol.entity_count(geom, level) == 0:
        message = f'Geometry "{geom}" has no domains'
        if sdim == 3:
            message += "; pass 'boundary' for its surface elements"
        raise ValueError(message + '.')
    sequence = _sequence(geom, mesh)
    kinds, points, owners, numbers = _elements(sequence, level, entities)
    variable = MEASURES[quality]
    if len(owners):
        values = _evaluate(geom, sequence, level, variable, points, owners,
                           numbers, entities)
    else:
        values = numpy.zeros((0, 2))
    rated, size = values[:, 0], values[:, 1]
    result: dict[str, Any] = {
        'mesh': _comsol.name_of(sequence), 'quality': quality,
        'level': 'domain' if level == sdim else 'boundary',
        'elements': {kind: int(numpy.count_nonzero(kinds == kind))
                     for kind in dict.fromkeys(kinds.tolist())}}
    if len(rated):
        result['min'] = float(rated.min())
        result['mean'] = float(rated.mean())
    else:
        result['min'] = result['mean'] = None
    lowest = numpy.argsort(rated, kind='stable')[:WORST]
    result['worst'] = [
        {'quality': float(rated[i]),
         'position': tuple(float(x) for x in points[:, i]),
         'entity': int(owners[i])} for i in lowest]
    # bins of exactly 0.1 width: 0.3 counts in 0.3-0.4, 1 in the last
    bins = numpy.clip(numpy.floor(rated * BINS), 0, BINS - 1).astype(int)
    result['histogram'] = [int(n) for n in numpy.bincount(bins,
                                                          minlength=BINS)]
    result['size'] = ({'min': float(size.min()), 'max': float(size.max())}
                      if len(size) else None)
    result['by_entity'] = _by_entity(rated, owners)
    result['unmeshed'] = sorted(set(entities) - set(owners.tolist()))
    result['messages'] = _messages(geom, sequence)
    return result


def find_mesh(geom: Node, mesh, check: bool = True) -> Any:
    """
    Returns the Java mesh sequence to draw: the one given, or the only
    non-empty one of the geometry's component. Raises for an empty mesh
    and for one whose settings changed since it was built, where COMSOL
    would draw nothing or the old mesh; `check=False` leaves that to the
    caller.
    """
    meshes = _comsol.component_of(geom).mesh()
    own = [str(t) for t in meshes.tags()]
    if mesh is True:
        if not own:
            raise RuntimeError(f'Geometry "{geom}" has no mesh; create one '
                               "with (model/'meshes').create(geom) and run "
                               'model.mesh().')
        full = [meshes.get(t) for t in own if not meshes.get(t).isEmpty()]
        if not full:
            raise RuntimeError(f'Geometry "{geom}" has no mesh yet; run '
                               'model.mesh() first.')
        if len(full) > 1:
            names = ', '.join(f'"{_comsol.name_of(m)}" ({m.tag()})'
                              for m in full)
            raise ValueError(f'Geometry "{geom}" has several meshes: '
                             f'{names}; pass mesh= one of these.')
        sequence = full[0]
    else:
        wrong = (f'mesh must be True, False, or a mesh name, tag or node, '
                 f'not {mesh!r}.')
        try:
            # the geometry's own meshes first: labels repeat across
            # components
            sequence = _comsol.find_node(meshes, mesh, 'mesh', 'meshes',
                                         wrong)
        except LookupError:
            sequence = _comsol.find_node(geom.model.java.mesh(), mesh,
                                         'mesh', 'meshes', wrong)
            raise ValueError(f'Mesh "{_comsol.name_of(sequence)}" does not '
                             f'belong to geometry "{geom}".') from None
    if check:
        _comsol.check_mesh_built(sequence)
    return sequence


def _sequence(geom: Node, mesh) -> Any:
    """
    Returns the Java mesh sequence. If it is empty or changed since it
    was built, the error names what COMSOL reported when building it last:
    a mesh whose only feature failed is empty, with the reason in an
    error node. If no mesh of the component has elements, the errors of
    all of them are named.
    """
    try:
        sequence = find_mesh(geom, True if mesh is None else mesh,
                             check=False)
    except RuntimeError as error:
        meshes = _comsol.component_of(geom).mesh()
        reasons = _failures(geom, [meshes.get(t) for t in meshes.tags()])
        if not reasons:
            raise
        raise RuntimeError(f'Geometry "{geom}" has no mesh elements; the '
                           f'last build failed: {reasons}') from error
    try:
        _comsol.check_mesh_built(sequence)
    except RuntimeError as error:
        reasons = _failures(geom, [sequence])
        if not reasons:
            raise
        name = _comsol.name_of(sequence)
        if sequence.isEmpty():
            raise RuntimeError(f'Mesh "{name}" has no elements; its last '
                               f'build failed: {reasons}') from None
        raise RuntimeError(f'{error} Its last build failed: {reasons}'
                           ) from None
    return sequence


def _failures(geom: Node, sequences) -> str:
    """The errors of the last builds of mesh sequences, as text."""
    try:
        errors = [item for sequence in sequences
                  for item in _messages(geom, sequence)
                  if item['severity'] == 'error']
    except Exception:
        return ''
    return '; '.join(_describe(item) for item in errors)


def _elements(sequence, level: int, entities: list[int]):
    """
    Returns the elements at a level in the given entities: their types,
    centroids (one column each), entities and numbers (1 for the first
    element of each type, as COMSOL's `meshelement`).
    """
    present = {str(kind) for kind in sequence.getTypes()}
    vertices = numpy.asarray(sequence.getVertex(), dtype=float)
    wanted = numpy.asarray(entities, dtype=int)
    kinds, points, owners, numbers = [], [], [], []
    for kind in (VOLUMES if level == 3 else FACES):
        if kind not in present:
            continue
        corners = numpy.asarray(sequence.getElem(kind), dtype=int)
        owner = numpy.asarray(sequence.getElemEntity(kind), dtype=int)
        keep = numpy.isin(owner, wanted)
        if not keep.any():
            continue
        kinds.append(numpy.full(int(keep.sum()), kind))
        points.append(vertices[:, corners[:, keep]].mean(axis=1))
        owners.append(owner[keep])
        numbers.append(numpy.nonzero(keep)[0] + 1)
    if not owners:
        return (numpy.array([], dtype=str),
                numpy.zeros((vertices.shape[0], 0)),
                numpy.array([], dtype=int), numpy.array([], dtype=int))
    return (numpy.concatenate(kinds), numpy.concatenate(points, axis=1),
            numpy.concatenate(owners), numpy.concatenate(numbers))


def _evaluate(geom: Node, sequence, level: int, variable: str, points,
              owners, numbers, entities: list[int]):
    """
    Returns the quality and size of each element, read at its centroid.
    A point is checked to have hit its own element; one on a boundary that
    coincides with another (an identity pair) may hit the other's, and is
    read again with only its own boundary selected.
    """
    model = geom.model.java
    with _comsol.scratch(model) as create:
        data = create(model.result().dataset(), 'Mesh')
        data.set('mesh', str(sequence.tag()))
        data.set('sorder', 'linear')
        interp = create(model.result().numerical(), 'Interp')
        interp.set('data', str(data.tag()))
        interp.set('expr', [variable, 'h', 'dom', 'meshelement'])
        interp.set('edim', str(level))
        selection = interp.selection()
        selection.geom(geom.tag(), level)

        def read(where, which):
            if which is None:
                selection.all()
            else:
                selection.set(which)
            interp.setInterpolationCoordinates(where)
            # a copy: the array over Java's is read-only
            return numpy.array(interp.getReal(), dtype=float).reshape(
                where.shape[1], 4)

        everything = len(entities) == _comsol.entity_count(geom, level)
        values = read(points, None if everything else entities)
        wrong = _missed(values, owners, numbers)
        for entity in numpy.unique(owners[wrong]):
            again = numpy.nonzero(wrong & (owners == entity))[0]
            values[again] = read(points[:, again], [int(entity)])
        wrong = _missed(values, owners, numbers)
    if wrong.any():
        raise RuntimeError(f'COMSOL could not read the quality of '
                           f'{int(wrong.sum())} elements of mesh '
                           f'"{_comsol.name_of(sequence)}" at their '
                           'centroids.')
    return values[:, :2]


def _missed(values, owners, numbers):
    """Tells which points did not hit their own element."""
    found = numpy.isfinite(values).all(axis=1)
    hit = numpy.zeros(len(values), dtype=bool)
    hit[found] = ((numpy.rint(values[found, 2]) == owners[found])
                  & (numpy.rint(values[found, 3]) == numbers[found]))
    return ~hit


def _by_entity(quality, owners) -> dict[int, dict]:
    """Returns the count, lowest and mean quality of each entity."""
    if not len(owners):
        return {}
    entities, index, counts = numpy.unique(owners, return_inverse=True,
                                           return_counts=True)
    lowest = numpy.full(len(entities), numpy.inf)
    numpy.minimum.at(lowest, index, quality)
    sums = numpy.bincount(index, weights=quality)
    return {int(e): {'elements': int(n), 'min': float(low),
                     'mean': float(total / n)}
            for e, n, low, total in zip(entities, counts, lowest, sums)}


############
# Messages #
############

def _messages(geom: Node, sequence) -> list[dict]:
    """
    Returns the information, warnings and errors on the active features
    of a Java mesh sequence (also on their subfeatures), with the messages
    of nested problems joined to their parent's.
    """
    found: list[dict] = []
    sdim = _comsol.sdim(geom)

    def walk(container, path: str):
        for tag in container.tags():
            feature = container.get(tag)
            try:
                if not feature.isActive():
                    continue
            except Exception:
                pass
            node = f'{path}/{_comsol.name_of(feature)}'
            try:
                problems = feature.problem()
                tags = list(problems.tags())
            except Exception:
                tags = []
            for problem in tags:
                found.append(_problem(feature.problem(problem), node, sdim))
            try:
                children = feature.feature()
            except Exception:
                continue
            walk(children, node)

    walk(sequence.feature(), f'meshes/{_comsol.name_of(sequence)}')
    return found


def _problem(problem, node: str, sdim: int) -> dict:
    """Returns one problem node, its nested ones joined in."""
    texts, kinds, level, entities = [], [], None, None
    stack = [problem]
    while stack:
        item = stack.pop(0)
        text = str(item.message()).strip() if hasattr(item, 'message') else ''
        if text:
            texts.append(text)
        try:
            kinds.append(str(item.getType()))
        except Exception:
            pass
        if level is None:
            try:
                if item.hasSelection():
                    chosen = item.selection()
                    dims = _comsol.selection_dims(chosen)
                    if len(dims) == 1:
                        level = _comsol.entity_level_name(dims[0], sdim)
                        entities = [int(e) for e in chosen.entities()]
            except Exception:
                pass
        try:
            stack.extend(item.problem(tag) for tag in item.problem().tags())
        except Exception:
            pass
    severity = ('error' if any('error' in k.lower() for k in kinds) else
                'warning' if any('warning' in k.lower() for k in kinds) else
                'info')
    return {'node': node, 'severity': severity, 'message': ' '.join(texts),
            'level': level, 'entities': entities}


def _describe(item: dict) -> str:
    """One problem as text, e.g. for an error message."""
    where = (f' ({item["level"]} {", ".join(map(str, item["entities"]))})'
             if item['entities'] else '')
    return f'"{item["node"]}": {item["message"]}{where}'
