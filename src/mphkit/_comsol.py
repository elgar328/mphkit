"""
Internals that depend on COMSOL or MPh behavior.

Everything that relies on COMSOL tag conventions or on MPh internals lives
here, so that a change in either needs a fix in one place only.
"""
from __future__ import annotations

import numbers
from collections.abc import Iterable
from difflib import get_close_matches

import numpy
from mph import Node
from mph.node import cast, escape, join, tag_pattern

from ._expr import vector

ENTITIES = ('domain', 'boundary', 'edge', 'point')
RESULT_SUFFIX = {'domain': 'dom', 'boundary': 'bnd', 'edge': 'edg', 'point': 'pnt'}

# Everyday words for entities, suggested (not accepted) in error messages.
# Faces are boundaries in 3D but domains in 2D; see `entity_suggestion`.
ENTITY_ALIASES = {
    'boundaries': 'boundary',
    'domains': 'domain', 'volume': 'domain', 'volumes': 'domain',
    'region': 'domain', 'regions': 'domain', 'subdomain': 'domain',
    'subdomains': 'domain', 'body': 'domain', 'bodies': 'domain',
    'edges': 'edge', 'line': 'edge', 'lines': 'edge', 'curve': 'edge',
    'curves': 'edge',
    'points': 'point', 'vertex': 'point', 'vertices': 'point',
    'corner': 'point', 'corners': 'point', 'node': 'point', 'nodes': 'point',
}
FACES = ('face', 'faces', 'surface', 'surfaces')
ENTITY_GLOSSARY = ("COMSOL's names: 'domain' (volumes in 3D, areas in 2D), "
                   "'boundary' (faces in 3D, edges in 2D), 'edge', 'point' "
                   "(vertices).")

# Everyday words for COMSOL property names, suggested in error messages:
# the candidates the object has (e.g. `rmaj` and `rmin` on a torus).
PROPERTY_ALIASES = {
    'radius': ('r', 'rmaj', 'rmin'), 'height': ('h',),
    'position': ('pos',), 'origin': ('pos',), 'location': ('pos',),
    'center': ('pos',), 'centre': ('pos',),
    'angle': ('rot', 'angles', 'angle1', 'angle2'), 'rotation': ('rot',),
    'displacement': ('displ', 'displx', 'disply', 'displz'),
    'offset': ('displ', 'displx', 'disply', 'displz'), 'spacing': ('displ',),
    'normal': ('normalvector', 'axis'),
    'thickness': ('distance',), 'length': ('distance',),
    'depth': ('distance',),
    'dimensions': ('size',), 'dims': ('size',),
    'count': ('size',), 'copies': ('size',),
}
# Aliases that hold for some feature types only: an Array's `size` is the
# number of copies, and the `axis` of a Rotate or Revolve is the rotation
# axis, not a normal.
ALIAS_ONLY = {('count', 'size'): ('Array',), ('copies', 'size'): ('Array',),
              ('normal', 'axis'): ('Mirror',)}
ALIAS_NOT = {('dimensions', 'size'): ('Array',), ('dims', 'size'): ('Array',)}


#########################
# Geometry and entities #
#########################

def geometry_of(node: Node) -> Node:
    """Returns the geometry node that `node` belongs to."""
    if len(node.path) < 2 or node.path[0] != 'geometries':
        raise TypeError(f'Node "{node}" is not a geometry or geometry feature.')
    return Node(node.model, join(node.path[:2]))


def component_of(geom: Node):
    """Returns the Java component that contains the geometry `geom`."""
    tag = geom.tag()
    components = geom.model.java.component()
    for ctag in components.tags():
        if tag in [str(t) for t in components.get(ctag).geom().tags()]:
            return components.get(ctag)
    raise LookupError(f'No component contains geometry "{geom}".')


def sdim(geom: Node) -> int:
    """Returns the space dimension of the geometry."""
    return int(geom.java.getSDim())


def entity_suggestion(entity, dim: int = 3) -> str | None:
    """Returns the COMSOL entity name meant by `entity`, or `None`."""
    key = str(entity).lower()
    if key in ENTITIES:
        return key
    if key in FACES:
        return 'boundary' if dim == 3 else 'domain'
    return ENTITY_ALIASES.get(key)


def entity_level_name(level: int, dim: int) -> str | None:
    """Names the entity level `level` (0 to `dim`) of a geometry."""
    names = {0: 'point', dim: 'domain'}
    if dim >= 2:
        names[dim - 1] = 'boundary'
    if dim == 3:
        names[1] = 'edge'
    return names.get(level)


def entity_error(geom: Node, entity, allowed=ENTITIES) -> str:
    """Explains an unknown entity name, suggesting COMSOL's name for it."""
    try:
        dim = parent_dim(geom)
    except Exception:
        dim = 3
    message = f'Entity must be one of {allowed}, not {entity!r}.'
    if isinstance(entity, numbers.Integral) and not isinstance(entity, bool):
        suggestion = entity_level_name(int(entity), dim)
    else:
        suggestion = entity_suggestion(entity, dim)
    if suggestion:
        message += f' Did you mean {suggestion!r}?'
    return f'{message} {ENTITY_GLOSSARY}'


def entity_name(geom: Node, entity: str) -> str:
    """Validates an entity name; in 2D (and work planes) an edge is a boundary."""
    if entity not in ENTITIES:
        raise ValueError(entity_error(geom, entity))
    dim = parent_dim(geom)
    if entity == 'edge':
        if dim < 2:
            raise ValueError('A 1D geometry has no edges.')
        if dim == 2:
            return 'boundary'
    return entity


def entity_dim(geom: Node, entity: str) -> int:
    """Maps an entity name to the `entitydim` value for a geometry or plane."""
    entity = entity_name(geom, entity)
    dim = parent_dim(geom)
    return {'domain': dim, 'boundary': dim - 1, 'edge': 1, 'point': 0}[entity]


def entity_count(geom: Node, dim: int) -> int:
    """Returns the number of entities of level `dim` in the built geometry."""
    java = geom.java
    if dim == sdim(geom):
        return int(java.getNDomains())
    if dim == sdim(geom) - 1:
        return int(java.getNBoundaries())
    return int(java.getNEdges() if dim == 1 else java.getNVertices())


def parent_dim(parent: Node) -> int:
    """Returns the space dimension of a geometry, or 2 for a work plane."""
    return 2 if is_workplane(parent.java) else sdim(parent)


def check_not_workplane(parent: Node, what: str):
    """Raises for a work plane where only a geometry makes sense."""
    if is_workplane(parent.java):
        raise TypeError(f'{what} needs a geometry, not the work plane '
                        f'"{parent}".')


def check_vector(parent: Node, name: str, value):
    """Raises unless `value` has one item per space dimension of `parent`."""
    dim = parent_dim(parent)
    if (not isinstance(value, (list, tuple, numpy.ndarray))
            or len(value) != dim):
        raise ValueError(f'{name} needs {dim} values in "{parent}", '
                         f'not {value!r}.')


def axis_properties(axis) -> dict:
    """Maps an axis given as 'x', 'y', 'z' or a vector to COMSOL properties."""
    if axis is None:
        return {}
    if isinstance(axis, str) and axis in ('x', 'y', 'z'):
        return {'axistype': axis}
    return {'axis': axis}


def check_built(geom: Node):
    """
    Raises if the geometry is not built or changed since its last build.

    Queries on an unbuilt geometry silently return nothing, and after an
    edit they return the numbers of the last build. COMSOL marks features
    as not built after edits, parameter changes and disabling, but not when
    a feature is removed, so a removal goes unnoticed.
    """
    features = geom.java.feature()
    stale = [str(tag) for tag in features.tags()
             if not features.get(tag).isBuilt()]
    if stale:
        raise RuntimeError(f'Geometry "{geom}" is not built or has changed '
                           f'since the last build (features: '
                           f'{", ".join(stale)}); run model.build(geom) '
                           'first.')


def check_selection(geom: Node, selection: Node):
    """
    Returns the Java selection behind a selection node of `geom`.

    MPh finds selections by label, and the selections COMSOL derives from
    one feature (e.g. `geom1_blk1_dom` and `geom1_blk1_bnd`) share the
    feature's label, so such a node may resolve to the wrong one. Those are
    rejected; `sel.result` and `sel.layer` give uniquely named selections.
    """
    if (not isinstance(selection, Node) or len(selection.path) != 2
            or selection.path[0] != 'selections'):
        raise TypeError(f'{selection!r} is not a selection node.')
    java = selection.java_if_exists()
    tag = str(java.tag())
    gtag = geom.tag()
    own = [str(t) for t in component_of(geom).selection().tags()]
    if tag not in own and not tag.startswith(f'{gtag}_'):
        raise ValueError(f'Selection "{selection}" does not belong to '
                         f'geometry "{geom}".')
    selections = geom.model.java.selection()
    label = str(java.label())
    if sum(str(selections.get(t).label()) == label
           for t in selections.tags()) > 1:
        raise ValueError(f'Several selections are named "{label}", so '
                         f'"{selection}" is ambiguous. Use sel.result(), '
                         'sel.layer() or sel.cumulative() for selections '
                         'COMSOL derives.')
    return java


##################
# Feature access #
##################

def is_workplane(java) -> bool:
    """Tells whether a Java node is a geometry work plane."""
    return (java is not None and hasattr(java, 'getType')
            and str(java.getType()) == 'WorkPlane')


def feature_container(parent: Node):
    """Returns the Java feature list that new features of `parent` go into."""
    java = parent.java
    if java is None:
        raise LookupError(f'Node "{parent}" does not exist.')
    if is_workplane(java):
        return java.geom().feature()
    if len(parent.path) == 2 and parent.path[0] == 'geometries':
        return java.feature()
    raise TypeError(f'Node "{parent}" is neither a geometry nor a work plane.')


def labels(container) -> set[str]:
    """Returns the labels of all members of a Java feature list."""
    return {str(container.get(tag).label()) for tag in container.tags()}


def selection_labels(model, exclude_prefix: str = None) -> set[str]:
    """
    Returns the labels of all selections in the model.

    This includes selections derived from geometry features (result
    selections, geometry-sequence selections), which MPh lists under the
    feature's label. Tags starting with `exclude_prefix` are left out.
    """
    selections = model.java.selection()
    found = set()
    for tag in selections.tags():
        if exclude_prefix and str(tag).startswith(exclude_prefix):
            continue
        found.add(str(selections.get(tag).label()))
    return found


def pick_label(name: str | None, default: str, taken: set[str]) -> str:
    """
    Returns the label for a new node.

    A label given by the user must be unique, because MPh finds nodes by
    label. An automatic label that clashes gets a number appended.
    """
    if name is not None:
        if name in taken:
            raise ValueError(f'The name "{name}" is already in use.')
        return name
    label, n = default, 2
    while label in taken:
        label = f'{default} ({n})'
        n += 1
    return label


def create_java(container, group: str, type: str, name: str | None,
                taken: set[str], tags=None, default: str = None,
                args: tuple = ()):
    """
    Creates a Java feature in `container` with a unique tag and label.

    Does not use MPh's `Node.create()`: that function finds the new node by
    label and may retag a different, existing node when labels clash.
    `tags` is the list whose `uniquetag()` is used (default: `container`),
    `default` replaces COMSOL's automatic label, and `args` go between tag
    and type (e.g. the geometry tag of a coordinate system). Returns
    `(tag, label)`.
    """
    if name is not None:
        pick_label(name, name, taken)            # fail before creating
    pattern = tag_pattern([group, '?', type]).rstrip('*')
    tag = str((tags or container).uniquetag(pattern))
    container.create(tag, *args, type)
    java = container.get(tag)
    label = pick_label(name, default or str(java.label()), taken)
    java.label(label)
    return tag, label


class WorkPlaneNode(Node):
    """
    MPh node that also resolves features inside a geometry work plane.

    MPh looks up child nodes in `java.feature()`, but the 2D features of a
    work plane live in `java.geom().feature()`. This subclass resolves,
    lists, creates and removes those; everything else is plain MPh.
    """

    def _workplane_parent(self):
        if self.is_root() or self.is_group():
            return None
        java = self.parent().java
        return java if is_workplane(java) else None

    @property
    def java(self):
        workplane = self._workplane_parent()
        if workplane is None:
            return super().java
        container = workplane.geom().feature()
        for tag in container.tags():
            member = container.get(tag)
            if self.name() == escape(member.label()):
                return member
        return None

    def children(self) -> list[Node]:
        java = self.java
        if is_workplane(java):
            container = java.geom().feature()
            return [self/escape(container.get(tag).label())
                    for tag in container.tags()]
        return super().children()

    def create(self, *arguments, name: str = None) -> Node:
        if is_workplane(self.java) and arguments:
            from .geometry import feature
            return feature(self, *arguments[:1], name=name)
        return super().create(*arguments, name=name)

    def remove(self):
        workplane = self._workplane_parent()
        if workplane is None:
            return super().remove()
        if not self.exists():
            raise LookupError(f'Node "{self}" does not exist in model tree.')
        workplane.geom().feature().remove(self.tag())


def child_node(parent: Node, label: str, cls: type = Node) -> Node:
    """Returns the node for the child of `parent` with the given label."""
    return cls(parent.model, join((*parent.path, label)))


def check_tag(node: Node, tag: str):
    """Verifies that the node resolves to the Java object just created."""
    if node.tag() != tag:
        raise RuntimeError(f'Node "{node}" resolves to tag "{node.tag()}", '
                           f'expected "{tag}".')


##############
# Properties #
##############

def is_selection_input(java, name: str) -> bool:
    """Tells whether a feature property is an input selection."""
    try:
        return str(java.getValueType(name)) == 'Selection'
    except Exception:
        return False


def convert(value):
    """Converts a Python value into something MPh's `cast()` accepts."""
    if isinstance(value, Node):
        return value.tag()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        items = [v.tag() if isinstance(v, Node) else v for v in value]
        if any(isinstance(v, (list, tuple, numpy.ndarray)) for v in items):
            return [vector(row) for row in items]
        if any(isinstance(v, bool) for v in items):
            return list(items)
        return vector(items)
    return value


def value_type(java, name: str) -> str | None:
    """Returns COMSOL's data type for a property, if it has one."""
    try:
        return str(java.getValueType(name))
    except Exception:
        return None


def allowed_values(java, name: str) -> list[str]:
    """Returns the values COMSOL accepts for a property, if enumerated."""
    try:
        return [str(v) for v in java.getAllowedPropertyValues(name)]
    except Exception:
        return []


def type_name(java) -> str:
    """
    Names a Java object for error messages.

    Features have `getType()`; physics property groups do not, so they are
    named by class and tag, e.g. `PhysicsPropClient (ShapeProperty)`.
    """
    try:
        return str(java.getType())
    except Exception:
        pass
    name = str(java.getClass().getSimpleName())
    try:
        return f'{name} ({java.tag()})'
    except Exception:
        return name


def set_property(java, name: str, value):
    """
    Sets a property.

    Unknown names raise `ValueError` with a suggestion. Some switches are
    stored as the strings `'on'`/`'off'` and reject a boolean (`sellayer`),
    while others accept one (`selresult`), so a boolean that COMSOL refuses
    is retried as `'on'`/`'off'`. Objects without a property list, such as
    variables, or with an empty one, such as a new material property group
    (it lists only what was set and takes any name), raise COMSOL's own
    error.
    """
    try:
        java.set(name, cast(convert(value)))
        return
    except Exception as error:
        if not hasattr(java, 'properties'):
            raise
        known = [str(p) for p in java.properties()]
        if not known:
            raise
        if name not in known:
            message = f'"{type_name(java)}" has no property "{name}".'
            close = property_suggestions(java, name, known)
            if close:
                message += f' Did you mean {", ".join(close)}?'
            message += ' Extra keyword arguments are COMSOL property names.'
            raise ValueError(message) from error
        if isinstance(value, bool) and value_type(java, name) == 'String':
            try:
                java.set(name, 'on' if value else 'off')
                return
            except Exception:
                pass
        allowed = allowed_values(java, name)
        if allowed:
            raise ValueError(
                f'Property "{name}" of "{type_name(java)}" accepts '
                f'{", ".join(repr(v) for v in allowed)}, not {value!r}.'
            ) from error
        raise


def property_suggestions(java, name: str, known: list[str]) -> list[str]:
    """
    Suggests property names for the unknown `name`, quoted for a message.

    An everyday word such as `radius` maps to COMSOL's name (`'r'`) when
    the object has it; otherwise the closest names are suggested.
    """
    same = [k for k in known if k.lower() == name.lower()]
    if same:
        return [repr(same[0])]
    key = name.lower()
    kind = type_name(java)
    found = [target for target in PROPERTY_ALIASES.get(key, ())
             if target in known
             and kind in ALIAS_ONLY.get((key, target), (kind,))
             and kind not in ALIAS_NOT.get((key, target), ())]
    if found == ['pos'] and key in ('center', 'centre'):
        return [center_hint(java, kind)]
    if found:
        return [repr(target) for target in found[:3]]
    return [repr(c) for c in get_close_matches(name, known, n=3, cutoff=0.7)]


def center_hint(java, kind: str) -> str:
    """Suggests how to place an object by its center."""
    try:
        corner = str(java.getString('base')) == 'corner'
    except Exception:
        corner = False
    if corner:
        return "'pos' with base='center'"
    if kind in ('Cylinder', 'Cone'):
        return "'pos' (the center of the base)"
    return "'pos'"


def set_properties(java, properties: dict, owner: Node = None,
                   container=None):
    """
    Sets properties in the given order, skipping `None` values.

    With `owner` (the geometry or work plane of a geometry feature) and its
    Java feature list `container`, input selections such as `input2` are
    assigned geometry objects and `contributeto` takes a cumulative
    selection by node or name; otherwise every key is a plain property.
    """
    for key, value in properties.items():
        if value is None:
            continue
        if owner is not None and key == 'contributeto' and value != 'none':
            if is_workplane(owner.java):
                raise ValueError('Features inside a work plane cannot '
                                 'contribute to a cumulative selection.')
            set_property(java, key, cumulative_tag(geometry_of(owner), value))
        elif owner is not None and is_selection_input(java, key):
            set_input(owner, container, java, key, value)
        else:
            set_property(java, key, value)


####################
# Input selections #
####################

def sequence_selection_tag(geom: Node, node: Node) -> str | None:
    """
    Returns the feature tag of a geometry-sequence selection.

    A selection feature such as `boxsel1` in geometry `geom1` shows up among
    the model's selections as `geom1_boxsel1`. Returns `None` for any other
    selection node (component selections, result selections).
    """
    tag = node.tag()
    gtag = geom.tag()
    for ftag in geom.java.feature().tags():
        if f'{gtag}_{ftag}' == tag:
            return str(ftag)
    return None


def resolve_object(container, ref: str) -> str:
    """Resolves an object reference given as tag or label to a tag."""
    tags = [str(t) for t in container.tags()]
    if ref in tags:
        return ref
    for tag in tags:
        if str(container.get(tag).label()) == ref:
            return tag
    if '(' in ref and ref.split('(')[0] in tags:
        return ref                               # array object, e.g. arr1(1,1)
    raise LookupError(f'No geometry object with tag or name "{ref}".')


INPUT_MODES = {
    ('Delete', 'input'): 'objects',
    ('Fillet3D', 'edge'): 'all', ('Chamfer3D', 'edge'): 'all',
    ('Fillet', 'point'): 'all', ('Chamfer', 'point'): 'all',
}
"""
How inputs that start at an entity level take geometry objects:
`'objects'` switches the input to objects (a Delete removes whole
objects), `'all'` selects every entity of the objects (the edges or
points a fillet rounds). Other entity-level inputs refuse objects.
"""


def selection_source(owner: Node, ref: Node):
    """
    Returns `(tag, level, kind)` for a selection node used as an input.

    `owner` is the geometry or work plane of the feature taking the input.
    `kind` is `'cumulative'` for a node from `sel.cumulative()` and
    `'sequence'` for a selection in the geometry or work plane sequence,
    given by its derived node or by the feature node itself (object-level
    and work-plane selections). `level` is the entity dimension, -1 for
    objects. Returns `None` for a node that is no selection; raises for a
    selection that cannot be an input here.
    """
    if (ref.path[0] == 'geometries' and len(ref.path) >= 4
            and not isinstance(ref, WorkPlaneNode)):
        ref = WorkPlaneNode(ref.model, join(ref.path))  # resolves in a plane
    if is_workplane(owner.java):
        if ref.path[0] == 'geometries' and ref.parent() == owner:
            if is_selection_feature(ref.java):
                return ref.tag(), sequence_level(owner, ref.tag()), \
                    'sequence'
            return None
        if (ref.path[0] == 'geometries' and len(ref.path) >= 4
                and is_selection_feature(ref.java)):
            raise ValueError(f'Selection "{ref}" belongs to "{ref.parent()}", '
                             f'not to the work plane "{owner}".')
        geom = geometry_of(owner)
        try:
            found = selection_source(geom, ref)
        except TypeError:
            raise TypeError(f'Selection "{ref}" cannot be used in a work '
                            'plane; use a selection made in that plane or '
                            'geometry objects.') from None
        if found is not None:
            raise ValueError(f'Selection "{ref}" belongs to "{geom}", not to '
                             'the work plane.')
        return None
    geom = owner
    if ref.path[0] == 'selections':
        found = find_cumulative(geom, ref)
        if found is not None:
            return (*found, 'cumulative')
        ftag = sequence_selection_tag(geom, ref)
        if ftag is None:
            raise TypeError(
                f'Selection "{ref}" cannot be used as input of a geometry '
                "operation. Use a selection made with where='geometry', a "
                'cumulative selection of this geometry, or geometry objects.')
        return ftag, sequence_level(geom, ftag), 'sequence'
    if ref.path[0] == 'geometries' and is_selection_feature(ref.java):
        if len(ref.path) == 3 and geometry_of(ref) == geom:
            return ref.tag(), sequence_level(geom, ref.tag()), 'sequence'
        where = 'the work plane ' if len(ref.path) > 3 else ''
        raise ValueError(f'Selection "{ref}" belongs to {where}'
                         f'"{ref.parent()}", not to "{geom}".')
    return None


def is_selection_feature(java) -> bool:
    """Tells whether a Java geometry feature is a selection, e.g. a box."""
    return (java is not None and hasattr(java, 'getType')
            and str(java.getType()).endswith('Selection'))


def sequence_level(parent: Node, ftag: str) -> int:
    """
    Returns the entity level of a sequence selection, -1 for objects.

    In a geometry it is read from the derived selection, not `entitydim`:
    an Adjacent selection keeps its input level there and an Explicit one
    has none. Object-level selections derive one selection per level
    instead, so there is none under the plain tag. A work plane derives
    nothing; its own selection list reports the level instead.
    """
    if is_workplane(parent.java):
        selections = parent.java.geom().selection()
        if ftag not in [str(t) for t in selections.tags()]:
            raise LookupError(f'Work plane "{parent}" lists no selection '
                              f'"{ftag}".')
        levels = [int(d) for d in selections.get(ftag).dimension()]
        return levels[0] if levels else -1
    selections = parent.model.java.selection()
    derived = f'{parent.tag()}_{ftag}'
    if derived in [str(t) for t in selections.tags()]:
        levels = [int(d) for d in selections.get(derived).dimension()]
        if levels:
            return levels[0]
    return -1


def set_input(owner: Node, container, java, key: str, value):
    """
    Assigns geometry objects or one selection to the input `key` of `java`.

    `owner` is the geometry or work plane the feature belongs to,
    `container` its Java feature list. The value is geometry objects
    (nodes, tags, labels), or one selection: made with `where='geometry'`
    (any level, including objects) or a cumulative selection. Everything is
    checked before the input changes. See `INPUT_MODES` for inputs that
    start at an entity level.
    """
    selection = java.selection(key)
    mode = INPUT_MODES.get((type_name(java), key))
    levels = [int(d) for d in selection.dimension()]
    dim = levels[0] if levels else None
    refs = value if isinstance(value, (list, tuple)) else [value]
    sources, objects = [], []
    for ref in refs:
        source = selection_source(owner, ref) if isinstance(ref, Node) \
            else None
        if source is not None:
            sources.append(source)
        elif isinstance(ref, Node):
            if ref.parent() != owner:
                raise ValueError(f'Node "{ref}" does not belong to "{owner}".')
            objects.append(ref.tag())
        elif isinstance(ref, str):
            objects.append(resolve_object(container, ref))
        else:
            raise TypeError(f'Cannot use {ref!r} as a geometry input.')
    if sources:
        if len(sources) > 1 or objects:
            raise ValueError('An input takes either geometry objects or '
                             'exactly one selection.')
        tag, level, kind = sources[0]
        if mode == 'objects':
            if level < 0 or kind == 'cumulative':
                selection.init()             # removes whole objects
            else:
                selection.init(level)        # removes these entities only
        elif dim is None:
            if kind == 'sequence' and level >= 0:
                selection.init(level)
        else:
            if level != dim:
                raise ValueError(f'Input "{key}" takes a selection of level '
                                 f'{dim}, not {level} (-1: objects).')
            selection.init(dim)
        selection.named(tag)
    elif mode == 'objects':
        selection.init()
        selection.set(objects)
    elif dim is None:
        selection.set(objects)
    elif mode == 'all':
        selection.init(dim)
        for tag in objects:
            selection.all(tag)
    else:
        raise TypeError(f'Input "{key}" takes entities of level {dim}, not '
                        "objects; pass a selection made with "
                        "where='geometry'.")


#####################
# Result selections #
#####################

def result_tag(geom: Node, feature: Node, entity: str) -> str:
    """Returns the tag of a feature's result selection for an entity."""
    return f'{geom.tag()}_{feature.tag()}_{RESULT_SUFFIX[entity]}'


def layer_tag(geom: Node, feature: Node, index: int | None) -> str:
    """
    Returns the tag of the selection for one of a feature's layers.

    Layers are numbered from 1 in the order of the `layername` property;
    `index=None` is the core, the part that is not in any layer. COMSOL
    creates these selections for domains only.
    """
    part = 'core' if index is None else f'layer{index}'
    return f'{geom.tag()}_{feature.tag()}_{part}'


def cumulative_tags(geom: Node) -> dict[str, str]:
    """
    Returns the cumulative selections of a geometry, label → tag.

    They live in the geometry's own selection list, which lists each one
    also per level (`csel1.dom`, ...) and the object selections of the
    features contributing to them (`cyl1`, `cyl1.dom`, ...). Cumulative
    selections have no type, only their client class tells them apart.
    """
    selections = geom.java.selection()
    found = {}
    for tag in selections.tags():
        member = selections.get(tag)
        if ('.' not in str(tag) and str(member.getClass().getSimpleName())
                == 'CumulativeSelectionClient'):
            found[str(member.label())] = str(tag)
    return found


def find_cumulative(geom: Node, node: Node):
    """
    Returns `(tag, level)` if `node` came from `sel.cumulative()` for `geom`.

    Such a node is a component Union of one selection COMSOL derived from a
    cumulative selection. Returns `None` for any other node.
    """
    dim = sdim(geom)
    levels = {'dom': dim, 'bnd': dim - 1, 'edg': 1, 'pnt': 0}
    gtag = geom.tag()
    derived = {f'{gtag}_{tag}_{suffix}': (tag, level)
               for tag in cumulative_tags(geom).values()
               for suffix, level in levels.items()}
    java = node.java
    if (java is None or not hasattr(java, 'getType')
            or str(java.getType()) != 'Union'):
        return None
    inputs = [str(t) for t in java.getStringArray('input')]
    if len(inputs) == 1 and inputs[0] in derived:
        return derived[inputs[0]]
    return None


def cumulative_tag(geom: Node, value) -> str:
    """Returns the tag of a cumulative selection given by node, label or tag."""
    found = cumulative_tags(geom)
    if isinstance(value, Node):
        match = find_cumulative(geom, value)
        if match is None:
            raise ValueError(f'"{value}" is not a cumulative selection of '
                             f'"{geom}"; use one returned by '
                             'sel.cumulative().')
        return match[0]
    value = str(value)
    if value in found:
        return found[value]
    if value in found.values():
        return value
    raise ValueError(f'Geometry "{geom}" has no cumulative selection '
                     f'"{value}". Known: {sorted(found) or "none"}. Create '
                     'it first with sel.cumulative(..., create=True).')


def names(values: Iterable) -> list[str]:
    """Returns the tags of selection nodes, passing strings through."""
    return [v.tag() if isinstance(v, Node) else str(v) for v in values]
