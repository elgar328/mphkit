"""
Lookups of COMSOL's internal names. Public as `mk.physics_types`,
`mk.feature_types`, `mk.properties` and `mk.variables`.

COMSOL has no API that lists the feature types a physics interface can
create or what their properties mean; a wrong name only raises "Unknown
feature ID". The installation ships code-completion data in
`data/completion/*.xml` for COMSOL's method editor (undocumented, one set
per version; checked with 6.4, and checked for the expected structure
when read, see `_format_problem`): interfaces,
features, subfeatures and properties with descriptions, defaults and
choices. These helpers read that data and combine it with what the running
COMSOL reports: current values, allowed values, variables. The level of a
physics feature (domain, boundary, ...) is in neither, so it is found by
creating the feature and removing it again, with the model history
switched off: the model is left as it was.
"""
from __future__ import annotations

import functools
import re
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple
from xml.etree import ElementTree

import numpy
from mph.node import Node, get, join

from . import _comsol

GROUPS = ('geometries', 'physics', 'materials', 'meshes', 'studies')
# Catalogue elements that are not features
SKIP = ('ModelEntityImage', 'LicenseRequirements', 'PhysicsPropList',
        'description')
# Runtime types the catalogue lists under another name
ALIASES = {'MeshSizeDefault': 'Size'}
# First words of the "where" column of variable tables
WHERE = {'domain': 'domain', 'domains': 'domain', 'boundary': 'boundary',
         'boundaries': 'boundary', 'edge': 'edge', 'edges': 'edge',
         'point': 'point', 'points': 'point', 'global': 'global'}
DEPENDENT = 'Dependent variable'
# Dimension groups every geometry, mesh and physics catalogue has
GROUPS_BY_DIM = ('S1D', 'S2D', 'S3D', 'S1DAxi', 'S2DAxi')
# Share of properties and feature types that must have descriptions and
# property lists in a catalogue of the expected format
SHARE = 0.9
# What still works without the catalogue
FALLBACK = ("mk.properties of an existing node still shows values and "
            "choices, mk.variables still works, and model.save('model.java') "
            'shows the names COMSOL uses.')
# Searches skip properties that this share of the types have, in lists
# of at least this many types
COMMON_SHARE = 0.3
COMMON_MIN = 10


class Prop(NamedTuple):
    """A property in the catalogue."""
    name: str
    description: str | None
    default: str | None
    choices: dict[str, str | None] | None


class Entry(NamedTuple):
    """An interface or feature type in the catalogue."""
    kind: str
    type: str
    description: str | None
    tag: str | None
    props: dict[str, Prop]
    element: Any


class CatalogueError(RuntimeError):
    """The catalogue files are missing or not in the expected format."""


#############
# Catalogue #
#############

def _root() -> str:
    """Returns the installation folder of the running COMSOL client."""
    import jpype  # type: ignore[import-untyped]
    return str(jpype.JClass('java.lang.System').getProperty('cs.root'))


def _version() -> str:
    """Returns COMSOL's version, for error messages."""
    import jpype  # type: ignore[import-untyped]
    try:
        util = jpype.JClass('com.comsol.model.util.ModelUtil')
        return str(util.getComsolVersion())
    except Exception:
        return 'this COMSOL version'


def catalogue(kind: str) -> tuple[Any, dict[str, Any]]:
    """
    Returns the catalogue of `kind` ('physics', 'geom', ...): its top
    element and its properties by code. Raises `CatalogueError` when it is
    missing or not in the expected format.
    """
    return _load(kind, _root())


@functools.lru_cache(maxsize=None)
def _load(kind: str, root: str) -> tuple[Any, dict[str, Any]]:
    path = Path(root)/'data'/'completion'/f'{kind}.xml'
    if not path.is_file():
        raise CatalogueError(f"COMSOL's catalogue of names was not found at "
                             f'"{path}". {FALLBACK}')
    top: Any = None
    try:
        tree = ElementTree.parse(path).getroot()
        top = tree.find(f'model.{kind}')
        problem = _format_problem(kind, tree, top)
    except ElementTree.ParseError as error:
        problem = f'unreadable XML: {error}'
    if problem:
        raise CatalogueError(f"COMSOL's catalogue of names at \"{path}\" has "
                             f'an unexpected format ({problem}; '
                             f'{_version()}). {FALLBACK}')
    props: dict[str, Any] = {}
    for element in top.find('Properties'):
        props.setdefault(element.tag, element)
    return top, props


def _format_problem(kind: str, tree, top) -> str | None:
    """
    Checks the structure a catalogue file needs, without relying on any
    feature name, and returns what is wrong, or `None`. COMSOL does not
    document these files, so a later version may change them; this raises
    instead of giving empty or undescribed results.
    """
    if tree.tag != 'COMSOLCompletionData' or top is None:
        return f'no COMSOLCompletionData/model.{kind}'
    properties = top.find('Properties')
    if properties is None or not len(properties):
        return 'no properties'
    if any(not p.get('name') for p in properties):
        return 'properties without a name'
    described = sum(p.get('descr') is not None for p in properties)
    if described < SHARE * len(properties):
        return 'properties without descriptions'
    if kind == 'material':
        found = top.find('Material')
        features = [found] if found is not None else []
    elif kind == 'study':
        found = top.find('Study/StudyFeature')
        features = list(found) if found is not None else []
    else:
        features = []
        path = {'physics': 'Physics/{}',
                'geom': 'Geom/{}/GeomSequence/GeomFeature',
                'mesh': "Mesh/{}/MeshSequence[@hasgeom='true']/MeshFeature"
                }[kind]
        for name in GROUPS_BY_DIM:
            group = top.find(path.format(name))
            if group is None:
                return f'no {path.format(name)}'
            if kind == 'mesh':
                for element in group:
                    features.append(element)
                    features.extend(e for e in element if e.tag not in SKIP)
            elif kind == 'geom':
                features.extend(group)
        if kind == 'physics':
            problem = _physics_links(top)
            if problem:
                return problem
            features = list(top.find('Features'))
    features = [e for e in features if e.tag not in SKIP]
    if not features:
        return 'no feature types'
    linked = [e for e in features if e.get('props') is not None]
    if len(linked) < SHARE * len(features):
        return 'feature types without properties'
    codes = {p.tag for p in properties}
    if any(code not in codes for e in linked
           for code in e.get('props').split()):
        return 'unknown property codes'
    return None


def _physics_links(top) -> str | None:
    """Checks that physics interfaces name features the catalogue has."""
    features = top.find('Features')
    if features is None or any(not f.get('name') for f in features):
        return 'no Features with names'
    codes = {f.tag for f in features}
    for name in GROUPS_BY_DIM:
        for interface in top.find(f'Physics/{name}'):
            if interface.tag in SKIP:
                continue
            listed = interface.get('features')
            if listed is None:
                return f'{name}/{interface.tag} without features'
            if any(code not in codes for code in listed.split()):
                return 'unknown feature codes'
    return None


@functools.lru_cache(maxsize=None)
def _features(root: str) -> dict[str, Any]:
    """Returns the physics features by code."""
    top, _ = _load('physics', root)
    return {element.tag: element for element in top.find('Features')}


def text(descr: str | None) -> str | None:
    """
    Returns the readable text of a catalogue description, or `None`.

    Descriptions are COMSOL's resource keys, whose English text is the
    key itself: `Heat_transfer_coefficient#`. After a `#` come values that
    translations fill in (`Frequency_x #1`); they are appended in
    parentheses, or used alone when the key is empty (`#PNG`). Context
    tags such as `[set]` are dropped.
    """
    if not descr:
        return None
    key, _, rest = descr.partition('#')
    fills = [fill for fill in rest.split('#') if fill.strip()]
    key = re.sub(r'\[[A-Za-z_]+\]', '', key).replace('_', ' ')
    key = ' '.join(key.split()).rstrip(',')
    if not key:
        return ', '.join(fills) or None
    if fills:
        return f'{key} ({", ".join(fills)})'
    return key


def _in_model(value: str, descr: str) -> bool:
    """
    Tells whether a catalogue choice names something in the sample model
    the catalogue was made from: a variable (`root.mod1.T`), an object
    whose label ends in its tag (`sys1`, `#Boundary System 1 (sys1)`) or
    a node whose label ends in its number (see `_node_label`).
    """
    return value.startswith('root.') or bool(
        descr and re.fullmatch(r'#.*\(' + re.escape(value) + r'\)', descr)
    ) or _node_label(value, descr)


def _node_label(value: str, descr: str) -> bool:
    """
    Tells whether a choice is a node of the sample model: a tag ending in
    a number with the node's label, e.g. `sys1` with 'Boundary System 1'
    or `pp1` with '#Particle Properties 1' (not `CO2` with '#CO2').
    """
    found = re.fullmatch(r'[A-Za-z:]+(\d+)', value)
    return bool(found and descr and re.fullmatch(
        r'#?[A-Za-z][A-Za-z ]* ' + found.group(1), descr))


def _prop(element) -> Prop:
    """
    Reads a property of the catalogue, without the choices that name
    things in the sample model the catalogue was made from. Where some were
    dropped, the other choices' copied labels (starting with `#`, e.g.
    'Temperature (ht)' on `userdef`) are dropped too.
    """
    choices: dict[str, str | None] | None = None
    dropped: set[str] = set()
    values = element.get('values')
    if values is not None:
        names = values.split('|')
        descrs = (element.get('descrs') or '').split('|')
        if len(descrs) != len(names):
            descrs = [''] * len(names)
        pairs = list(zip(names, descrs))
        dropped = {name for name, descr in pairs if _in_model(name, descr)}
        found: dict[str, str | None] = {}
        for name, descr in pairs:
            if name in dropped:
                continue
            if dropped and descr.startswith('#'):
                descr = ''
            found.setdefault(name, text(descr))
        choices = found or None
    default = element.get('default')
    if default and (default.startswith('root.') or default in dropped):
        default = None
    return Prop(element.get('name'), text(element.get('descr')), default,
                choices)


def _entry(kind: str, element, type: str | None = None) -> Entry:
    """Reads an interface or feature of the catalogue."""
    _, props = catalogue(kind)
    found: dict[str, Prop] = {}
    for code in (element.get('props') or '').split():
        if code in props:
            prop = _prop(props[code])
            found.setdefault(prop.name, prop)
    return Entry(kind, type or element.tag, text(element.get('descr')),
                 element.get('id'), found, element)


def children(entry: Entry) -> list[Entry]:
    """Returns the feature types that can be created under `entry`."""
    if entry.kind == 'physics':
        features = _features(_root())
        found = [features[code] for code in
                 (entry.element.get('features') or '').split()
                 if code in features]
        return _unique([_entry('physics', element, element.get('name'))
                        for element in found])
    return _unique([_entry(entry.kind, element) for element in entry.element
                    if element.tag not in SKIP])


def _unique(entries: list[Entry]) -> list[Entry]:
    """Keeps the first entry of each type (study.xml repeats some)."""
    found: dict[str, Entry] = {}
    for entry in entries:
        found.setdefault(entry.type, entry)
    return list(found.values())


def _child(entry: Entry | None, type: str) -> Entry | None:
    """Returns the feature type `type` under `entry`, if listed."""
    if entry is None:
        return None
    type = ALIASES.get(type, type)
    for child in children(entry):
        if child.type == type:
            return child
    return None


def group(geometry) -> str:
    """Returns the catalogue group of a Java geometry, e.g. 'S2DAxi'."""
    dim = int(geometry.getSDim())
    try:
        axisymmetric = bool(geometry.isAxisymmetric())
    except Exception:
        axisymmetric = False
    return f'S{dim}D' + ('Axi' if axisymmetric else '')


def _sequence(kind: str, name: str) -> Entry:
    """Returns the geometry or mesh sequence of a catalogue group."""
    top, _ = catalogue(kind)
    if kind == 'geom':
        element = top.find(f'Geom/{name}/GeomSequence/GeomFeature')
    else:
        element = top.find(f"Mesh/{name}/MeshSequence[@hasgeom='true']"
                           '/MeshFeature')
    if element is None:
        raise TypeError(f'COMSOL lists no {kind} features for {name}.')
    return Entry(kind, name, None, None, {}, element)


def _interface(physics, geometry) -> Entry | None:
    """Returns the catalogue entry of a Java physics interface."""
    top, _ = catalogue('physics')
    element = top.find(f'Physics/{group(geometry)}/{physics.getType()}')
    if element is None:
        return None
    return Entry('physics', element.tag, text(element.get('descr')),
                 element.get('id'), {}, element)


#########
# Nodes #
#########

def _check_node(node, what: str):
    """Raises unless `node` is an MPh node inside one of the model groups."""
    if not isinstance(node, Node) or len(node.path) < 2 \
            or node.path[0] not in GROUPS:
        raise TypeError(f'{what}, not {node!r}.')


def _java(node: Node):
    """Returns the Java object of a node, also inside a work plane."""
    if node.path[0] == 'geometries':
        node = _comsol.WorkPlaneNode(node.model, join(node.path))
    return _comsol.java_of(node)


def _geometry_of_physics(model, physics):
    """Returns the Java geometry of a physics interface, or `None`."""
    try:
        component = model.java.component(str(physics.model()))
        tag = str(physics.geom())
        if tag in [str(t) for t in component.geom().tags()]:
            return component.geom(tag)
    except Exception:
        pass
    return None


def _no_geometry(node: Node) -> TypeError:
    return TypeError(f'Physics "{Node(node.model, join(node.path[:2]))}" '
                     'has no geometry; mphkit looks up names of physics '
                     'on a geometry only.')


def _physics_entry(node: Node) -> tuple[Any, Any, Entry | None]:
    """
    Returns the Java physics interface of a physics node or feature, its
    Java geometry (or `None`) and the catalogue entry of the node.
    """
    physics = _comsol.java_of(Node(node.model, join(node.path[:2])))
    geometry = _geometry_of_physics(node.model, physics)
    if geometry is None:
        return physics, None, None
    return physics, geometry, _walk(node, _interface(physics, geometry))


def _walk(node: Node, entry: Entry | None) -> Entry | None:
    """
    Follows the features below the interface or sequence (the first two
    parts of the node's path) down to the node, in the catalogue.
    """
    for depth in range(3, len(node.path) + 1):
        java = _comsol.java_of(Node(node.model, join(node.path[:depth])))
        entry = _child(entry, str(java.getType()))
    return entry


def _geometry_entry(node: Node, java) -> Entry | None:
    """Returns the catalogue entry of a geometry feature."""
    geometry = _comsol.java_of(Node(node.model, join(node.path[:2])))
    name = group(geometry)
    if len(node.path) > 3:
        parent = _java(Node(node.model, join(node.path[:-1])))
        if _comsol.is_workplane(parent):
            name = 'S2D'
    return _child(_sequence('geom', name), str(java.getType()))


def _mesh_geometry(node: Node):
    """Returns the Java geometry that a mesh sequence node belongs to."""
    tag = str(_comsol.java_of(Node(node.model, join(node.path[:2]))).tag())
    components = node.model.java.component()
    for ctag in components.tags():
        component = components.get(ctag)
        if tag in [str(t) for t in component.mesh().tags()]:
            geometries = list(component.geom().tags())
            if geometries:
                return component.geom(geometries[0])
    raise TypeError(f'Mesh "{Node(node.model, join(node.path[:2]))}" has '
                    'no geometry.')


def _mesh_entry(node: Node) -> Entry | None:
    """Returns the catalogue entry of a mesh feature or subfeature."""
    return _walk(node, _sequence('mesh', group(_mesh_geometry(node))))


def _study_steps() -> Entry:
    top, _ = catalogue('study')
    return Entry('study', 'Study', None, None, {},
                 top.find('Study/StudyFeature'))


def _material() -> Entry:
    top, _ = catalogue('material')
    return _entry('material', top.find('Material'))


##########
# Search #
##########

def name_key(name: str) -> tuple[str, str]:
    """Sorts names regardless of case, e.g. `init` among the `I`s."""
    return name.casefold(), name


def _words(search) -> list[str]:
    """Splits a search into lower-case words."""
    if search is None:
        return []
    if not isinstance(search, str):
        raise TypeError(f'search must be a string, not {search!r}.')
    return search.lower().split()


def _found(words: list[str], *texts) -> bool:
    """Tells whether every word occurs in the joined texts."""
    joined = ' '.join(t for t in texts if t).lower()
    return all(word in joined for word in words)


def _prop_text(prop: Prop) -> str:
    return f'{prop.name} ({prop.description})' if prop.description \
        else prop.name


def _searched(prop: Prop, common) -> bool:
    """
    Tells whether a property helps tell feature types apart in a search.
    Model inputs (`minput_*`), sources of a value (`*_src`, next to the
    property they feed) and properties most types share are left out.
    """
    return not (prop.name.startswith('minput_') or prop.name.endswith('_src')
                or prop.name in common)


def stage(entry: Entry, words: list[str], common=frozenset()
          ) -> tuple[int, bool, str | None] | None:
    """
    Returns how a feature type matches a search, or `None`: the stage (1
    by its own type, description or tag, 2 by a choice of a property, 3
    by a property), whether the words were only found together with the
    type's own text, and the choice or property that matched. Properties
    in `common` are not searched.
    """
    head = f'{entry.type} {entry.description or ""} {entry.tag or ""}'
    if _found(words, head):
        return 1, False, None
    props = [p for p in entry.props.values() if _searched(p, common)]
    candidates: list[tuple[int, str, str]] = []
    for prop in props:
        for value, descr in (prop.choices or {}).items():
            if value.startswith('{'):     # a reference to another node
                continue
            shown = f'{prop.name} = {value}' + (f' ({descr})' if descr
                                               else '')
            candidates.append((2, f'{value} {descr or ""}', shown))
    for prop in props:
        candidates.append((3, f'{prop.name} {prop.description or ""}',
                           _prop_text(prop)))
    for own in (True, False):
        for level, words_of, shown in candidates:
            if _found(words, words_of) if own \
                    else _found(words, head, words_of):
                return level, not own, shown
    return None


def search_types(entries: list[Entry], words: list[str]
                 ) -> list[tuple[tuple[int, bool, str | None], Entry]]:
    """
    Returns the feature types that match a search with how they match
    (see `stage`), in stage order, fallback matches last, then by type.

    In a list of at least `COMMON_MIN` types, properties that more than
    `COMMON_SHARE` of them have (common settings) do not make a type
    match, unless nothing else matches.
    """
    if not words:
        return sorted((((1, False, None), e) for e in entries),
                      key=lambda m: name_key(m[1].type))
    common: set[str] = set()
    if len(entries) >= COMMON_MIN:
        counts = Counter(name for e in entries for name in e.props)
        common = {name for name, count in counts.items()
                  if count > COMMON_SHARE * len(entries)}

    def matches(skip) -> list:
        found = [(stage(e, words, skip), e) for e in entries]
        return [(how, e) for how, e in found if how is not None]

    found = matches(common)
    if not found and common:
        found = matches(frozenset())
    return sorted(found,
                  key=lambda m: (m[0][0], m[0][1], name_key(m[1].type)))


###############
# Level trial #
###############

def _creates(container, type: str, level: int | None) -> list[int] | None:
    """
    Creates `type` in a Java feature list and removes it again. Returns
    the level COMSOL gave it (a list, empty for global or unknown), or
    `None` if it could not be created.
    """
    tag = str(container.uniquetag('mk'))
    try:
        try:
            if level is None:
                feature = container.create(tag, type)
            else:
                feature = container.create(tag, type, level)
        except Exception:
            return None
        try:
            return [int(d) for d in feature.selection().dimension()]
        except Exception:
            return []
    finally:
        if tag in [str(t) for t in container.tags()]:
            container.remove(tag)


def levels(container, type: str, dim: int) -> dict[str, int]:
    """
    Returns the levels at which `type` can be created in a Java feature
    list: level names to the numbers `create()` takes, highest first, or
    `{'global': -1}`. Call with the history switched off.

    Created without a level, a feature takes its default one; an empty
    selection level means global when -1 works too.
    """
    found: set[int] = set()
    default = None
    dims = _creates(container, type, None)
    if dims == [] and _creates(container, type, -1) is not None:
        return {'global': -1}
    if dims:
        default = dims[0]
        found.add(default)
    for level in range(dim, -1, -1):
        if level != default and _creates(container, type, level) \
                is not None:
            found.add(level)
    return {_comsol.entity_level_name(level, dim) or str(level): level
            for level in sorted(found, reverse=True)}


##########
# Values #
##########

def plain(value):
    """Converts numpy values and paths from MPh into plain Python values."""
    if isinstance(value, numpy.ndarray):
        if value.ndim == 0:
            return plain(value.item())
        return [plain(item) for item in value]
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _selection(java, name: str):
    """
    Reads a selection property of a geometry feature: the object names,
    or objects to entity numbers when some entities are picked.
    """
    try:
        selection = java.selection(name)
        objects = [str(o) for o in selection.objects()]
    except Exception:
        return None
    numbers: dict[str, list[int]] = {}
    for obj in objects:
        try:
            numbers[obj] = [int(n) for n in selection.entities(obj)]
        except Exception:     # an object consumed by a later operation
            numbers[obj] = []
    if any(numbers.values()):
        return {obj: found or None for obj, found in numbers.items()}
    return objects


def _value(java, name: str, geometry_feature: bool):
    """Returns the current value of a property as plain Python values."""
    if _comsol.value_type(java, name) == 'Selection':
        return _selection(java, name) if geometry_feature else None
    try:
        return plain(get(java, name))
    except Exception:
        return None


def _row(value, prop: Prop | None, choices: list[str] | None) -> dict:
    """Returns the description of one property."""
    known = prop.choices if prop and prop.choices else {}
    if choices is None:
        listed = prop.choices if prop else None
    else:
        listed = {c: known.get(c) for c in choices} if choices else None
    return {'value': value,
            'description': prop.description if prop else None,
            'default': prop.default if prop else None,
            'choices': listed}


def _current(java, props: dict[str, Prop] | None, geometry_feature: bool,
             prefix: str = '') -> dict[str, dict]:
    """
    Describes the properties of a Java object, filtered by the catalogue.
    An entry without properties filters nothing: the catalogue lists none
    for some features (e.g. initial values of pressure acoustics).
    """
    found = {}
    names = [str(n) for n in java.properties()]
    if props and not props.keys() & set(names):
        props = None      # the catalogue entry does not fit this node
    for name in names:
        if props and name not in props:
            continue
        found[prefix + name] = _row(
            _value(java, name, geometry_feature),
            props.get(name) if props else None,
            _comsol.allowed_values(java, name))
    return found


def _filter(rows: dict[str, dict], words: list[str]) -> dict[str, dict]:
    """Keeps the properties that match a search, sorted by name."""
    found = {}
    for name in sorted(rows, key=name_key):
        row = rows[name]
        choices = row['choices'] or {}
        texts = [name, row['description'], *choices,
                 *[d for d in choices.values() if d]]
        if _found(words, *texts):
            found[name] = row
    return found


##################
# Public helpers #
##################

def physics_types(geom: Node, /, *, search: str | None = None) -> list[dict]:
    """
    Returns the physics interfaces that can be added to a geometry, e.g.
    the one for heat conduction:

    ```python
    mk.physics_types(geom, search='heat transfer')
    # [..., {'type': 'HeatTransfer', 'description': 'Heat transfer in
    #        solids', 'tag': 'ht'}, ...]
    physics = (model/'physics').create('HeatTransfer', geom)
    ```

    Each item has `type`, the name `(model/'physics').create(type, geom)`
    takes, `description` and `tag`, the prefix of the tags COMSOL gives
    it. `search` keeps the interfaces whose type, description or tag hold
    every word of it (any case). The list depends on the dimension of the
    geometry and whether it is axisymmetric, and comes from the
    installation's catalogue (`data/completion`, COMSOL's undocumented
    code-completion data; a missing or changed one raises), which also
    lists interfaces of modules that may not be licensed; licenses are not
    checked. Interfaces that COMSOL creates only without a geometry (e.g.
    metal phase transformation) are left out. Sorted by type; a 3D
    geometry has more than 250, so search first.
    """
    words = _words(search)
    if not isinstance(geom, Node) or len(geom.path) != 2 \
            or geom.path[0] != 'geometries':
        raise TypeError(f'mk.physics_types takes a geometry, not {geom!r}.')
    top, _ = catalogue('physics')
    name = group(_comsol.java_of(geom))
    element = top.find(f'Physics/{name}')
    if element is None:
        raise TypeError(f'COMSOL lists no physics interfaces for {name}.')
    found = []
    for interface in element:
        if interface.tag in SKIP:
            continue
        item = {'type': interface.tag,
                'description': text(interface.get('descr')),
                'tag': interface.get('id')}
        if _found(words, *item.values()):
            found.append(item)
    return sorted(found, key=lambda item: name_key(item['type']))


def feature_types(parent: Node, /, *, search: str | None = None
                  ) -> list[dict]:
    """
    Returns the feature types that can be created under a node: the
    features of a physics interface (with the levels they go on) or the
    subfeatures of a physics feature, and the features of a geometry, work
    plane, mesh or mesh feature, or the steps of a study:

    ```python
    mk.feature_types(heat, search='convective')
    # [{'type': 'ConvectiveOutflow', ...},
    #  {'type': 'HeatFluxBoundary', 'description': 'Heat flux',
    #   'tag': 'hf', 'levels': {'boundary': 2},
    #   'match': 'HeatFluxType = ConvectiveHeatFlux (Convective heat
    #             flux)'}, ...]
    heat.create('HeatFluxBoundary', 2)
    mk.feature_types(study, search='time')   # ..., 'Transient', ...
                                             # (name matches first)
    ```

    Each item has `type`, the name to create it with, `description`,
    `tag` (the tag prefix), `levels` and `match`. Physics features are
    created with `physics.create(type, level)`, geometry features with
    `mk.feature(geom, type)` (also in a work plane, where MPh's `create`
    does not work), mesh features with `mesh.create(type)` and study
    steps with `study.create(type)`.

    `levels` is for physics features only (`None` for the others): the
    entity levels the feature can go on, mapped to the number `create()`
    takes for them, highest first: `{'domain': 3}`, `{'boundary': 2}`,
    several for some features, `{'global': -1}` for global ones (e.g.
    `GlobalEquations`, created with `-1` or no number), and `{}` when
    COMSOL creates the feature neither at a numbered level nor with -1,
    e.g. one that needs another setting or feature first. Subfeatures go
    on their parent's level. Mesh features take no level number; their
    selection sets it.

    `search` finds types whose type, description or tag hold every word
    of it (any case); then those with a property choice or a property
    that holds them, e.g. 'convective' finds `HeatFluxBoundary` through
    its choice `ConvectiveHeatFlux`, 'emissivity' finds
    `SurfaceToAmbientRadiation` through `epsilon_rad`. They come in that
    order, each sorted by type; types found only with words of their own
    name, as 'initial temperature' finds `init` through `Tinit`
    (Temperature), come after the others of their kind. `match` names the
    choice or property (`None` for the first kind). Settings most types of
    a long list share (e.g. `useparam` of study steps) do not count unless
    nothing else matches. Search first, with one distinctive word where
    possible: there are more than 70 physics features for heat transfer,
    97 geometry features in 3D and 81 study steps.

    Some types need objects first (Boolean operations in an empty
    geometry) or another feature (`ElseIf` after `If`).

    The types come from the installation's catalogue. Physics levels are
    found by creating each feature and removing it again, with the model
    history switched off, which leaves the model as it was; that takes
    about 0.03 s per feature (2 to 3 s for all of heat transfer or solid
    mechanics). Physics interfaces created without a geometry raise
    `TypeError`.
    """
    words = _words(search)
    _check_node(parent, 'mk.feature_types takes a physics interface or '
                        'feature, a geometry or work plane, a mesh or mesh '
                        'feature, or a study')
    physical = parent.path[0] == 'physics'
    geometry = None
    if physical:
        _, geometry, entry = _physics_entry(parent)
        if geometry is None:
            raise _no_geometry(parent)
        if entry is None:
            raise RuntimeError(f'"{parent}" is not in the catalogue of '
                               f'{_version()}.')
    else:
        entry = _parent_entry(parent, 'mk.feature_types')
    matches = search_types(children(entry), words)
    if geometry is None:
        return [_type_item(child, None, match)
                for (_, _, match), child in matches]
    container = _comsol.java_of(parent).feature()
    dim = int(geometry.getSDim())
    with _comsol.history_off(parent.model.java):
        return [_type_item(child, levels(container, child.type, dim), match)
                for (_, _, match), child in matches]


def _type_item(entry: Entry, found_levels, match) -> dict:
    return {'type': entry.type, 'description': entry.description,
            'tag': entry.tag, 'levels': found_levels, 'match': match}


def lists_properties(node) -> bool:
    """
    Tells whether `mk.properties(node)` lists properties that are set on
    the node itself: not those of a physics interface (`'group/name'`),
    nor of a geometry, mesh or study, which have none of their own.
    """
    return (isinstance(node, Node) and len(node.path) >= 2
            and node.path[0] in GROUPS
            and not (len(node.path) == 2 and node.path[0] in
                     ('geometries', 'meshes', 'studies', 'physics')))


def properties(node: Node, type: str | None = None, /, *,
               search: str | None = None) -> dict[str, dict]:
    """
    Returns the properties of a node, or of a feature type before it is
    created, with their meaning:

    ```python
    mk.properties(heat/'cooling', search='flux')
    # {'HeatFluxType': {'value': 'ConvectiveHeatFlux',
    #                   'description': 'Flux type',
    #                   'default': 'GeneralInwardHeatFlux',
    #                   'choices': {'GeneralInwardHeatFlux':
    #                               'General inward heat flux', ...}}, ...}
    mk.properties(heat, 'HeatFluxBoundary')     # before creating one
    ```

    Each property has `value` (the current value, `None` for a type),
    `description`, `default` and `choices` (allowed values with their
    descriptions, `None` when any value goes or, before creation, when the
    values are names in the model); unknown parts are `None`. Before
    creation, choices naming things in a model (coordinate systems, ...)
    are left out; the model's own are accepted too. `search`
    keeps properties whose name, description or choices hold every word
    of it (any case), e.g. 'size' (not 'mesh size') on a mesh size
    feature; search first, a feature has up to a hundred properties.
    Sorted by name.

    Nodes: physics features and subfeatures, the physics interface itself
    (keys `'group/name'`, set with
    `mk.set(physics.java.prop('ShapeProperty'), order_temperature=1)`),
    geometry features (also in work planes), mesh features, study steps,
    materials and their property groups. A node's properties are those
    COMSOL documents in the installation's catalogue, internal ones left
    out; nodes it does not list, such as a geometry's `Finalize` or a
    material's property group, show all properties without descriptions.
    Selections of geometry features show the object names, or objects and
    their entity numbers.

    With a `type`, `node` is where it would be created: a physics
    interface or feature, a geometry or work plane, a mesh or mesh
    feature, or a study, e.g. `mk.properties(geom, 'Cone')`,
    `mk.properties(mesh, 'FreeTet')`, `mk.properties(study, 'Transient')`.
    An unknown type raises `ValueError` with the closest names. A few
    types have no properties in the catalogue (e.g. `init` of pressure
    acoustics) and give `{}`; once created, their node shows all
    properties.
    """
    words = _words(search)
    _check_node(node, 'mk.properties takes a model node (physics, '
                      'geometry, mesh, study or material)')
    if type is None:
        return _filter(_existing(node), words)
    return _filter(_before(node, type), words)


def _existing(node: Node) -> dict[str, dict]:
    """Describes the properties of an existing node."""
    group_name, depth = node.path[0], len(node.path)
    if depth == 2 and group_name in ('geometries', 'meshes', 'studies'):
        raise TypeError(f'"{node}" has no properties of its own: pass a '
                        'feature, or a type to create, e.g. '
                        "mk.properties(geom, 'Block').")
    java = _java(node)
    if group_name == 'physics' and depth == 2:
        return _physics_settings(node, java)
    try:
        entry = _existing_entry(node, java)
    except CatalogueError:
        entry = None
    props = entry.props if entry is not None else None
    return _current(java, props, group_name == 'geometries')


def _existing_entry(node: Node, java) -> Entry | None:
    """Returns the catalogue entry of an existing feature, if listed."""
    group_name, depth = node.path[0], len(node.path)
    if group_name == 'physics':
        return _physics_entry(node)[2]
    if group_name == 'geometries':
        return _geometry_entry(node, java)
    if group_name == 'meshes':
        return _mesh_entry(node)
    if group_name == 'studies':
        return _child(_study_steps(), str(java.getType()))
    if group_name == 'materials' and depth == 2:
        return _material()
    return None


def _physics_settings(node: Node, java) -> dict[str, dict]:
    """Describes the property groups of a physics interface."""
    geometry = _geometry_of_physics(node.model, java)
    listed: dict[str, dict[str, Prop]] | None = None
    try:
        entry = _interface(java, geometry) if geometry is not None else None
    except CatalogueError:
        entry = None
    if entry is not None:
        _, props = catalogue('physics')
        listed = {}
        found = entry.element.find('PhysicsPropList')
        for group_element in (found if found is not None else []):
            listed[group_element.tag] = {}
            for code in (group_element.get('props') or '').split():
                if code in props:
                    prop = _prop(props[code])
                    listed[group_element.tag].setdefault(prop.name, prop)
    rows = {}
    for settings in java.prop():
        tag = str(settings.tag())
        if listed and tag not in listed:
            continue
        rows.update(_current(settings, listed.get(tag) if listed else None,
                             False, prefix=f'{tag}/'))
    return rows


def _before(parent: Node, type: str) -> dict[str, dict]:
    """Describes the properties of a feature type before it is created."""
    if not isinstance(type, str):
        raise TypeError(f'type must be a string, not {type!r}.')
    found = children(_parent_entry(parent, 'mk.properties(parent, type)'))
    for child in found:
        if child.type == type:
            return {name: _row(None, prop, None)
                    for name, prop in child.props.items()}
    names = [child.type for child in found]
    message = f'"{parent}" has no feature type "{type}".'
    close = get_close_matches(type, names, n=3, cutoff=0.6) or \
        [n for n in names if type.lower() in n.lower()][:3]
    if close:
        message += f' Did you mean {", ".join(repr(c) for c in close)}?'
    message += ' mk.feature_types(parent) lists the types.'
    raise ValueError(message)


def _parent_entry(parent: Node, caller: str) -> Entry:
    """
    Returns the catalogue entry that new features of `parent` go into;
    `caller` names the helper in error messages.
    """
    group_name, depth = parent.path[0], len(parent.path)
    java = _java(parent)
    entry: Entry | None
    if group_name == 'physics':
        _, geometry, entry = _physics_entry(parent)
        if geometry is None:
            raise _no_geometry(parent)
    elif group_name == 'geometries' and depth == 2:
        entry = _sequence('geom', group(java))
    elif group_name == 'geometries' and _comsol.is_workplane(java):
        entry = _sequence('geom', 'S2D')
    elif group_name == 'meshes' and depth == 2:
        entry = _sequence('mesh', group(_mesh_geometry(parent)))
    elif group_name == 'meshes' and depth == 3:
        entry = _mesh_entry(parent)
    elif group_name == 'studies' and depth == 2:
        entry = _study_steps()
    else:
        raise TypeError(f'{caller} takes a physics interface or feature, a '
                        'geometry or work plane, a mesh or mesh feature, or '
                        f'a study, not "{parent}".')
    if entry is None:
        raise RuntimeError(f'"{parent}" is not in the catalogue of '
                           f'{_version()}.')
    return entry


def variables(node: Node, /, *, search: str | None = None) -> list[dict]:
    """
    Returns the variables that physics interfaces define, for result
    expressions:

    ```python
    mk.variables(heat, search='heat flux')
    # [..., {'name': 'ht.ntflux', 'unit': 'W/m^2',
    #        'description': 'Normal total heat flux',
    #        'where': ['Boundaries 1–14'], 'levels': ['boundary']}, ...]
    mk.integral(geom, 'boundary', 'ht.ntflux', hot, unit='W')
    ```

    `node` is a physics interface, or a geometry for all physics on it.
    Each item has `name`, `unit`, `description`, `where` (the entities, as
    COMSOL writes them) and `levels`: 'domain', 'boundary', 'edge',
    'point' or 'global' (`None` for a text it does not recognize). Global
    ones are evaluated with MPh's `model.evaluate(name)`, the others with
    `mk.integral`, `mk.average`, `mk.maximum`, `mk.value` etc. on those
    entities. The dependent variables (`T`, `u`, `v`, `w`, ...) come
    first, described by their field, e.g. 'Dependent variable
    (temperature)', some of which COMSOL keeps for optional features;
    the rest is sorted by name. `search` keeps variables whose name or
    description hold every word of it (any case). Search first, with one
    distinctive word where possible: a physics interface defines hundreds.

    The geometry must have been built: before that COMSOL lists global
    variables only. Variables of multiphysics couplings (e.g. thermal
    expansion) are not included, since COMSOL lists none for them.
    Physics interfaces created without a geometry raise `TypeError`.
    """
    words = _words(search)
    _check_node(node, 'mk.variables takes a physics interface or a '
                      'geometry')
    if len(node.path) != 2 or node.path[0] not in ('physics', 'geometries'):
        raise TypeError('mk.variables takes a physics interface or a '
                        f'geometry, not "{node}".')
    java = _comsol.java_of(node)
    if node.path[0] == 'physics':
        geometry = _geometry_of_physics(node.model, java)
        if geometry is None:
            raise _no_geometry(node)
        interfaces = [java]
    else:
        geometry = java
        component = _comsol.component_of(node)
        tag = str(java.tag())
        interfaces = [component.physics(t) for t in
                      component.physics().tags()
                      if str(component.physics(t).geom()) == tag]
    if int(geometry.getNDomains()) + int(geometry.getNBoundaries()) \
            + int(geometry.getNVertices()) == 0:
        raise RuntimeError(f'The geometry of "{node}" is not built or '
                           'empty; run model.build(geom) first (COMSOL '
                           'lists only global variables before).')
    dependent: dict[str, dict] = {}
    others: dict[str, dict] = {}
    for physics in interfaces:
        for field in physics.field().tags():
            for name in physics.field(field).component():
                dependent.setdefault(str(name), {
                    'name': str(name), 'unit': None,
                    'description': f'{DEPENDENT} ({field})', 'where': None,
                    'levels': None})
        for row in _rows(physics):
            name, unit, descr, where = [
                '' if row[i] is None else str(row[i]) for i in (1, 3, 4, 5)]
            if name in dependent:
                continue
            item = others.setdefault(name, {
                'name': name, 'unit': unit or None,
                'description': descr or None, 'where': [], 'levels': []})
            if where and where not in item['where']:
                item['where'].append(where)
                level = where_level(where)
                if level not in item['levels']:
                    item['levels'].append(level)
    found = list(dependent.values()) + [others[n] for n in
                                        sorted(others, key=name_key)]
    return [item for item in found
            if _found(words, item['name'], item['description'])]


def where_level(where: str) -> str | None:
    """
    Returns the entity level of a variable table's "where" text, e.g.
    'boundary' for 'Boundaries 1–14', or `None` for an unknown text.
    """
    return WHERE.get(where.split(' ')[0].lower())


def _rows(physics) -> list:
    """Returns the rows of the variable tables of a physics interface."""
    rows: list = []

    def read(java):
        try:
            rows.extend(java.featureInfo('info').getInfoTable('Expression'))
        except Exception:
            pass
        try:
            features = java.feature()
        except Exception:
            return
        for tag in features.tags():
            read(features.get(tag))

    read(physics)
    return rows
