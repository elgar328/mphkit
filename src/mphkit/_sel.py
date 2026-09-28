"""Geometry-based selections; documented in `mphkit.sel`."""
from __future__ import annotations

import math
import numbers

from mph.node import Node
from mph.node import escape

from . import _comsol
from .geometry import feature

WHERE = ('component', 'geometry')
MARGIN = 1e-6  # relative margin of box bounds, see box()


def _where(parent: Node, where: str | None) -> str:
    """
    Resolves `where`: by default the component for a geometry and the
    plane's own sequence for a work plane, which has no component
    selections.
    """
    workplane = _comsol.is_workplane(parent.java)
    if where is None:
        return 'geometry' if workplane else 'component'
    if where not in WHERE:
        raise ValueError(f'where must be one of {WHERE}, not {where!r}.')
    if workplane and where == 'component':
        raise ValueError('A work plane has no component selections; its '
                         'selections are made in the plane (leave out '
                         'where).')
    return where


def _level(geom: Node, entity: str, where: str | None) -> int:
    """Maps an entity name to `entitydim`; `'object'` is -1 (geometry only)."""
    where = _where(geom, where)
    if entity != 'object' and str(entity).lower() in ('object', 'objects'):
        message = (f"Entity must be one of {_comsol.ENTITIES + ('object',)}, "
                   f"not {entity!r}. Did you mean 'object'?")
        if where != 'geometry':
            message += " The 'object' level needs where='geometry'."
        raise ValueError(f'{message} {_comsol.ENTITY_GLOSSARY}')
    if entity == 'object':
        if where != 'geometry':
            raise ValueError("The 'object' level is for inputs of geometry "
                             "operations; use it with where='geometry'.")
        return -1
    return _comsol.entity_dim(geom, entity)


def _create(geom: Node, type: str, where: str | None, name: str | None,
            properties: dict, default: str | None = None) -> Node:
    """Creates a selection of `type` (component names, e.g. 'Box')."""
    where = _where(geom, where)
    model = geom.model
    if _comsol.is_workplane(geom.java):
        # A work plane derives no model-level selections: the feature in
        # the plane is the input for later operations there.
        return feature(geom, f'{type}Selection', name=name, **properties)
    if where == 'geometry':
        node = feature(geom, f'{type}Selection', name=name, **properties)
        if properties.get('entitydim') == -1:
            # Objects: COMSOL derives four same-labeled selections, none of
            # them for physics. The feature itself is the input to use.
            return node
        derived = model/'selections'/escape(node.name())
        _comsol.check_tag(derived, f'{geom.tag()}_{node.tag()}')
        return derived
    container = _comsol.component_of(geom).selection()
    taken = _comsol.selection_labels(model)
    tag, label = _comsol.create_java(container, 'selections', type, name,
                                     taken, tags=model.java.selection(),
                                     default=default)
    node = model/'selections'/escape(label)
    try:
        _comsol.check_tag(node, tag)
        java = container.get(tag)
        _comsol.set_properties(java, properties)
    except Exception:
        container.remove(tag)
        raise
    return node


def _inputs(geom: Node, where: str | None, values) -> list[str]:
    """Returns selection tags for the inputs of a set operation."""
    where = _where(geom, where)
    values = [values] if isinstance(values, (Node, str)) else list(values)
    if where == 'component':
        for value in values:
            if isinstance(value, Node) and value.path[0] == 'geometries':
                raise TypeError(f'"{value}" is a selection in the geometry '
                                "sequence; use where='geometry' with it.")
        return _comsol.names(values)
    tags = []
    for value in values:
        if isinstance(value, Node):
            source = _comsol.selection_source(geom, value)
            if source is None:
                raise TypeError(f'"{value}" is not a selection.')
            if source[2] == 'cumulative':
                raise TypeError(
                    f'Selection "{value}" is a cumulative selection; set '
                    "operations with where='geometry' take selections made "
                    "with where='geometry'.")
            tags.append(source[0])
        else:
            tags.append(str(value))
    return tags


def _bounds(geom: Node, **ranges) -> dict:
    """Turns x/y/z ranges into Box properties; a scalar is a plane."""
    dim = _comsol.parent_dim(geom)
    properties = {}
    for axis, value in ranges.items():
        if value is None:
            continue
        if 'xyz'.index(axis) >= dim:
            raise ValueError(f'A {dim}D geometry has no {axis} coordinate.')
        if isinstance(value, (list, tuple)):
            low, high = value
        else:
            low = high = value
        if (all(isinstance(v, numbers.Real) and not isinstance(v, bool)
                for v in (low, high)) and high < low):
            raise ValueError(
                f'{axis}=({low}, {high}) is reversed: give (min, max), e.g. '
                f'{axis}=({high}, {low}). COMSOL would select what lies '
                'outside the range; use sel.complement for that.')
        properties[f'{axis}min'] = _widen(low, -1)
        properties[f'{axis}max'] = _widen(high, +1)
    return properties


def _widen(value, sign: int):
    """Widens a bound by a millionth of its value, away from the range."""
    if isinstance(value, str):
        op = '-' if sign < 0 else '+'
        return f'({value}){op}1e-6*abs({value})'  # same text as box()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return value
    try:
        number = float(value)
    except OverflowError:
        return value  # passed on as text, see _comsol.convert
    if not math.isfinite(number):
        return number  # inf - inf would be NaN
    return number + sign*MARGIN*abs(number)


def box(geom: Node, entity: str, /, x=None, y=None, z=None, *,
        condition: str = 'inside', where: str | None = None,
        name: str | None = None, **properties) -> Node:
    """
    Selects the entities inside a box.

    `x`, `y`, `z` are `(min, max)` ranges; a single value such as `z=0.3`
    selects on that plane (in 2D or a work plane, `x=10, y=10` is a
    corner). Omitted coordinates are unbounded. Each bound gets a margin of
    a millionth of its value, since COMSOL compares faces, edges and
    domains in single precision; features thinner than that may be picked
    too. So `x=1.1` is stored as 1.0999989 to 1.1000011, and `z='L'` as
    `(L)-1e-6*abs(L)` to `(L)+1e-6*abs(L)`; pass `xmin=`, `xmax=`, ... to
    set a bound exactly.

    With `where='geometry'` (or in a work plane), `entity` may be
    `'object'` to select whole objects as input of a geometry operation.
    `geom` may also be a work plane, in the plane's coordinates, e.g. to
    fillet one corner there (see `mphkit.sel` for what such selections can
    be used for).

    `condition` defaults to `'inside'` (entity entirely inside the box).
    This differs from COMSOL's default `'intersects'`, which also picks
    entities that only touch the box, e.g. every face touching a plane.
    Other values: `'intersects'`, `'allvertices'`, `'somevertex'`.
    """
    properties = {'entitydim': _level(geom, entity, where),
                  'condition': condition,
                  **_bounds(geom, x=x, y=y, z=z), **properties}
    return _create(geom, 'Box', where, name, properties)


def ball(geom: Node, entity: str, /, center, r, *,
         condition: str = 'inside', where: str | None = None,
         name: str | None = None, **properties) -> Node:
    """
    Selects the entities inside a ball of radius `r` around `center`.

    `condition` defaults to `'inside'`, see `box()`. The radius gets no
    automatic margin: give it a little room, e.g. `1.01*r`.
    """
    dim = _comsol.parent_dim(geom)
    if len(center) != dim:
        raise ValueError(f'center needs {dim} coordinates.')
    position = {f'pos{a}': c for a, c in zip('xyz', center)}
    properties = {'entitydim': _level(geom, entity, where),
                  'condition': condition, 'r': r, **position, **properties}
    return _create(geom, 'Ball', where, name, properties)


def cylinder(geom: Node, entity: str, /, pos, r, *, axis=None, top=None,
             bottom=None, rin=None, condition: str = 'inside',
             where: str | None = None, name: str | None = None,
             **properties) -> Node:
    """
    Selects the entities inside a cylinder (3D).

    `pos` is the center of the base and `axis` the direction, `'x'`, `'y'`,
    `'z'` (default) or a vector. `top` and `bottom` are measured from `pos`
    along the axis; left out, the cylinder is unbounded. With `rin` it is a
    shell, e.g. `r=2.1, rin=1.9` picks the side faces of an r=2 cylinder.
    Bound the shell with `top` and `bottom` slightly beyond the cylinder,
    e.g. `bottom=-0.01*h, top=1.01*h`; unbounded, it also picks side faces
    of other objects on the same axis.
    `condition` defaults to `'inside'`, see `box()`. In 2D use `disk()`.
    Radii and `top`/`bottom` get no automatic margin: give them a little
    room, e.g. `1.01*r`.
    """
    _comsol.check_not_workplane(geom, 'sel.cylinder')
    if _comsol.sdim(geom) != 3:
        raise ValueError('sel.cylinder needs a 3D geometry; use sel.disk in '
                         '2D.')
    _comsol.check_vector(geom, 'pos', pos)
    if axis is not None and not isinstance(axis, str):
        _comsol.check_vector(geom, 'axis', axis)
    properties = {'entitydim': _level(geom, entity, where),
                  'condition': condition, 'pos': pos, 'r': r, 'rin': rin,
                  'top': top, 'bottom': bottom,
                  **_comsol.axis_properties(axis), **properties}
    return _create(geom, 'Cylinder', where, name, properties)


def disk(geom: Node, entity: str, /, center, r, *, rin=None,
         condition: str = 'inside', where: str | None = None,
         name: str | None = None, **properties) -> Node:
    """
    Selects the entities inside a disk of radius `r` around `center` (2D).

    With `rin` it is a ring. `condition` defaults to `'inside'`, see
    `box()`. In 3D use `cylinder()` or `ball()`. Radii get no automatic
    margin: give them a little room, e.g. `1.01*r`.
    """
    if _comsol.parent_dim(geom) != 2:
        raise ValueError('sel.disk needs a 2D geometry; use sel.cylinder or '
                         'sel.ball in 3D.')
    _comsol.check_vector(geom, 'center', center)
    properties = {'entitydim': _level(geom, entity, where),
                  'condition': condition, 'posx': center[0],
                  'posy': center[1], 'r': r, 'rin': rin, **properties}
    return _create(geom, 'Disk', where, name, properties)


def all_(geom: Node, entity: str, /, *, where: str | None = None,
         name: str | None = None) -> Node:
    """
    Selects all entities of one kind, e.g. all boundaries.

    Use it as `mk.sel.all`: `from mphkit.sel import *` would replace
    Python's built-in `all`.
    """
    properties = {'entitydim': _level(geom, entity, where),
                  'condition': 'intersects'}
    return _create(geom, 'Box', where, name, properties)


def union(geom: Node, entity: str, /, input, *, where: str | None = None,
          name: str | None = None) -> Node:
    """Selects the union of the `input` selections."""
    properties = {'entitydim': _level(geom, entity, where),
                  'input': _inputs(geom, where, input)}
    return _create(geom, 'Union', where, name, properties)


def intersection(geom: Node, entity: str, /, input, *,
                 where: str | None = None, name: str | None = None) -> Node:
    """Selects the entities that all `input` selections have in common."""
    properties = {'entitydim': _level(geom, entity, where),
                  'input': _inputs(geom, where, input)}
    return _create(geom, 'Intersection', where, name, properties)


def difference(geom: Node, entity: str, /, add, subtract, *,
               where: str | None = None, name: str | None = None) -> Node:
    """Selects the entities in `add` that are not in `subtract`."""
    properties = {'entitydim': _level(geom, entity, where),
                  'add': _inputs(geom, where, add),
                  'subtract': _inputs(geom, where, subtract)}
    return _create(geom, 'Difference', where, name, properties)


def complement(geom: Node, entity: str, /, input, *,
               where: str | None = None, name: str | None = None) -> Node:
    """Selects all entities that are not in the `input` selections."""
    properties = {'entitydim': _level(geom, entity, where),
                  'input': _inputs(geom, where, input)}
    return _create(geom, 'Complement', where, name, properties)


def adjacent(geom: Node, /, input, entity: str = 'boundary', *,
             input_entity: str = 'domain', exterior: bool = True,
             interior: bool = False, where: str | None = None,
             name: str | None = None) -> Node:
    """
    Selects the entities of kind `entity` adjacent to the `input` selections.

    For example the exterior boundaries of a domain selection. The inputs
    are of kind `input_entity`.
    """
    properties = {'entitydim': _comsol.entity_dim(geom, input_entity),
                  'outputdim': _comsol.entity_dim(geom, entity),
                  'input': _inputs(geom, where, input),
                  'exterior': exterior, 'interior': interior}
    return _create(geom, 'Adjacent', where, name, properties)


def result(geom: Node, feature: Node, entity: str, /, *,
           name: str | None = None) -> Node:
    """
    Selects the entities created by a geometry feature.

    Turns on the feature's result selection (`selresult`) and returns a
    named component selection that physics can use, without relying on
    COMSOL's tag convention. Calling it again returns the same selection.
    Because it switches on `selresult`, a Java export of the model shows
    that setting on the feature; the geometry is not affected.
    """
    if len(feature.path) != 3 or _comsol.geometry_of(feature) != geom:
        raise ValueError(f'"{feature}" is not a top-level feature of '
                         f'"{geom}".')
    entity = _comsol.entity_name(geom, entity)
    suffix = _comsol.RESULT_SUFFIX[entity]
    java = _comsol.java_of(feature)
    known = [str(p) for p in java.properties()]
    if 'selresult' not in known:
        raise ValueError(f'"{feature}" has no result selection.')
    own = f'{geom.tag()}_{feature.tag()}'
    clash = _comsol.selection_labels(geom.model, exclude_prefix=own)
    if feature.name() in clash:
        raise ValueError(f'A selection is already named "{feature.name()}"; '
                         'rename the feature, or give that selection '
                         'another name (`name=` when creating it).')
    was_on = str(java.getString('selresult')) == 'on'
    _comsol.set_property(java, 'selresult', True)
    if 'selresultshow' in known:
        show = str(java.getString('selresultshow'))
        if not was_on:
            java.set('selresultshow', suffix)
        elif show not in (suffix, 'all'):
            java.set('selresultshow', 'all')
    derived = _comsol.result_tag(geom, feature, entity)
    return _wrap(geom, derived, entity, name,
                 default=f'{feature.name()} ({entity})')


def layer(geom: Node, feature: Node, layer, /, *, name: str | None = None) -> Node:
    """
    Selects the domains of one layer of a geometry feature.

    Layers are the shells a Block, Cylinder, Sphere or Rectangle can be
    given (`layername`, `layer`, `layertop`, ...), typically a perfectly
    matched layer. `layer` is a name from `layername`, its number counted
    from 1, or `'core'` for the part outside every layer.

    COMSOL derives these selections for domains only, and they survive
    boolean operations on the feature. They are empty when the feature is
    consumed, for instance as the subtracted object of a difference; a
    `sel.box` over the same region is the alternative then. Calling this
    again returns the same selection. Like `result()`, it switches a
    setting on the feature (`sellayer`), which shows in a Java export.
    """
    if len(feature.path) != 3 or _comsol.geometry_of(feature) != geom:
        raise ValueError(f'"{feature}" is not a top-level feature of '
                         f'"{geom}".')
    java = _comsol.java_of(feature)
    known = [str(p) for p in java.properties()]
    if 'sellayer' not in known:
        raise ValueError(f'"{feature}" does not support layers.')
    names = [str(n) for n in java.getStringArray('layername')] \
        if 'layername' in known else []
    expected: str | None
    if isinstance(layer, str) and layer.lower() == 'core':
        index, expected = None, 'Core'
    elif isinstance(layer, bool) or not isinstance(layer, (int, str)):
        raise TypeError(f'A layer is a name, a number or "core", '
                        f'not {layer!r}.')
    elif isinstance(layer, int):
        index = layer
        expected = names[index - 1] if 0 < index <= len(names) else None
    elif layer in names:
        index = names.index(layer) + 1
        expected = layer
    else:
        raise ValueError(f'"{feature}" has no layer named "{layer}". '
                         f'Known names: {names or "none"} (or a number, '
                         f'or "core").')
    _comsol.set_property(java, 'sellayer', True)
    derived = _comsol.layer_tag(geom, feature, index)
    selections = geom.model.java.selection()
    if derived not in [str(t) for t in selections.tags()]:
        prefix = f'{geom.tag()}_{feature.tag()}_'
        found = [str(t) for t in selections.tags()
                 if str(t).startswith(prefix) and 'layer' in str(t)
                 or str(t) == f'{prefix}core']
        raise ValueError(f'"{feature}" has no layer selection "{derived}". '
                         f'Found: {found or "none"}.')
    label = str(selections.get(derived).label())
    if expected and not label.startswith(expected):
        raise RuntimeError(f'Selection "{derived}" is labeled "{label}", '
                           f'expected it to start with "{expected}". The '
                           'COMSOL naming convention may have changed.')
    return _wrap(geom, derived, 'domain', name,
                 default=f'{feature.name()} ({expected or index})')


def entities(geom: Node, selection: Node, /) -> list[int]:
    """
    Returns the entity numbers of a selection, in ascending order.

    The numbers are at the selection's own level: an adjacent selection of
    domains gives boundary numbers. Entity numbers change when the geometry
    changes, so use them to check or inspect a model, not to set up physics.
    The geometry must be built; a selection made with `where='geometry'`
    changes the geometry sequence, so build again after creating one.
    """
    _comsol.check_not_workplane(geom, 'sel.entities')
    _comsol.check_built(geom)
    java = _comsol.check_selection(geom, selection)
    return sorted(int(e) for e in java.entities())


def find(geom: Node, entity: str, /, x=None, y=None, z=None, *,
         condition: str = 'inside') -> list[int]:
    """
    Returns the numbers of the entities inside a box, without a selection.

    Takes the coordinates and `condition` of `box()`, but removes the
    selection again and returns the entity numbers. Meant for checks and
    lookups: entity numbers change with the geometry, so physics should use
    `box()`. For a lookup of another kind, create the selection, read it
    with `entities()` and `remove()` it; leave out `name` to avoid label
    clashes.
    """
    _comsol.check_not_workplane(geom, 'sel.find')
    node = box(geom, entity, x, y, z, condition=condition)
    try:
        return entities(geom, node)
    finally:
        _comsol.component_of(geom).selection().remove(node.tag())


def cumulative(geom: Node, group, entity: str, /, *, create: bool = False,
               name: str | None = None) -> Node:
    """
    Selects the entities of a cumulative selection, COMSOL's way of
    collecting what several features create.

    In the COMSOL Desktop a feature's "Contribute to" list has a New
    button that creates a cumulative selection; other features pick it
    from the list, and physics offers it per level, e.g.
    "holes (Boundary)". The same steps here:

    ```python
    plate = mk.block(geom, (60, 40, 5))
    holes = mk.sel.cumulative(geom, 'holes', 'domain', create=True)  # New
    # Contribute to
    hole = mk.cylinder(geom, 3, 7, (15, 10, -1), contributeto=holes)
    row = mk.array(geom, hole, size=(3, 2, 1), displ=(15, 20, 0))
    mk.difference(geom, plate, [row])
    walls = mk.sel.cumulative(geom, 'holes', 'boundary')  # holes (Boundary)
    ```

    `group` is the cumulative selection's name, or a node returned by this
    function. `create=True` creates it and fails if it exists; otherwise it
    must exist, so a typo raises instead of giving an empty selection.
    Features take the node or the name as `contributeto`. It holds every
    entity of the contributing objects, including copies made by `array`,
    `move`, `rotate` or `mirror` (also with `keep=True`), and survives
    later Boolean operations: above, `walls` are the walls of the six
    holes. The copying feature may also take `contributeto=` itself. Only
    top-level features of the geometry can contribute, not features inside
    a work plane. Calling this again returns the same selection.
    """
    _comsol.check_not_workplane(geom, 'sel.cumulative')
    entity = _comsol.entity_name(geom, entity)
    model = geom.model
    selections = _comsol.java_of(geom).selection()
    known = _comsol.cumulative_tags(geom)
    if create:
        if not isinstance(group, str):
            raise TypeError('create=True takes the name of a new cumulative '
                            f'selection, not {group!r}.')
        if group == 'none':
            raise ValueError('"none" means "no cumulative selection" in '
                             'COMSOL; choose another name.')
        if group in known:
            raise ValueError(f'Cumulative selection "{group}" already exists '
                             f'in "{geom}".')
        taken = _comsol.selection_labels(model)
        if group in taken:
            raise ValueError(f'A selection is already named "{group}".')
        if name is not None:
            _comsol.pick_label(name, name, taken)
        tag = str(selections.uniquetag('csel'))
        selections.create(tag, 'CumulativeSelection')
        selections.get(tag).label(group)
    else:
        tag = _comsol.cumulative_tag(geom, group)
    try:
        derived = f'{geom.tag()}_{tag}_{_comsol.RESULT_SUFFIX[entity]}'
        tags = [str(t) for t in model.java.selection().tags()]
        if derived not in tags:
            prefix = f'{geom.tag()}_{tag}_'
            found = [t for t in tags if t.startswith(prefix)]
            raise ValueError(f'Cumulative selection "{group}" has no '
                             f'selection "{derived}". Found: {found}.')
        label = str(selections.get(tag).label())
        return _wrap(geom, derived, entity, name,
                     default=f'{label} ({entity})')
    except Exception:
        if create:
            selections.remove(tag)
        raise


def _wrap(geom: Node, derived: str, entity: str, name: str | None,
          default: str) -> Node:
    """
    Returns a component selection that stands for a derived selection.

    MPh lists selections COMSOL derives from a geometry feature under that
    feature's label, so several can share one name and none of them can be
    addressed by path. A Union with a unique label and the derived tag as
    its input can. An existing wrapper is found by its input, so renaming
    the feature does not create a second one.
    """
    container = _comsol.component_of(geom).selection()
    for tag in container.tags():
        member = container.get(tag)
        if (str(member.getType()) == 'Union'
                and [str(t) for t in member.getStringArray('input')]
                == [derived]):
            return geom.model/'selections'/escape(member.label())
    properties = {'entitydim': _comsol.entity_dim(geom, entity),
                  'input': [derived]}
    return _create(geom, 'Union', 'component', name, properties,
                   default=default)
