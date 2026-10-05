"""
A summary of a model's settings, independent of tags, feature order and
entity numbers. Public as `mk.describe`.

The defaults of each node come from a temporary model that has the same
components, geometries, physics interfaces, meshes and studies (same tags),
where every node is created again with its tag and type and read; it is
removed afterwards. Selections are described by the location and size of
their entities, measured on the user's geometry with the history off.
Comparing the solver sequences makes a temporary sequence in the user's
model, as `mk.problem_size` does, and removes it again.

The module has two layers: the functions that read COMSOL's Java objects
(`_Reader`, `_skeleton`) and plain functions that sort the values read
(`json_value`, `differs`, `step_map`, `owner`, `solver_changes`), which
the tests check without COMSOL.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from mph.model import Model
from mph.node import get

from . import _catalog, _check, _comsol, _solve

# Version of the result's layout; raise it when the layout changes
FORMAT = 1
# Tag prefix of the temporary model and solver sequences
SCRATCH = 'mkdesc'
AXES = 'xyz'

# Properties left out everywhere: the study step a node shows equations
# for ('std1/stat'), when and how a mesh was built, the plot groups of a
# study step, and its lists of physics, couplings and components by label
# ('heat (ht)'); they follow what was solved, shown or named, while
# `activate` and the like hold the settings. Also whether the COMSOL
# Desktop shows the model inputs ('on' in models it saved, 'off' in new
# ones).
SKIPPED = frozenset({'StudyStep', 'buildinfo', 'buildoutput', 'buildtime',
                     'plotgrouparr', 'plotgroupdummy', 'physselection',
                     'outputInterface', 'multiphysicsSelection',
                     'geomselection', 'minpVisibility'})
# Settings of a physics interface that follow its features
SKIPPED_SETTINGS = frozenset({'PhysicalModelProperty/hasDG'})
# Mesh sizes that COMSOL derives from the predefined size and the geometry
# while `custom` is off
MESH_SIZES = ('Size', 'MeshSizeDefault')
DERIVED_SIZES = ('hmax', 'hmin', 'hcurve', 'hgrad', 'hnarrow')
# Study step properties that list [tag, value, tag, value, ...] pairs
# (physics, 'frame:...', 'multi:...', components or geometries)
STEP_MAPS = frozenset({'activate', 'activateCoupling', 'discretization',
                       'equationform', 'equationform_freq', 'outputmap',
                       'outputselectionmap', 'shapeOrder', 'mesh'})
# How materials look in the COMSOL Desktop, not what they are
APPEARANCE = frozenset({
    'family', 'color', 'specular', 'diffuse', 'ambient', 'lighting',
    'fresnel', 'roughness', 'flipanisotropy', 'metallic', 'pearl',
    'clearcoat', 'reflectance', 'shininess', 'transparency',
    'uniformblending', 'diffusewrap', 'ambientscale', 'showambientscale',
    'info', 'updateIsValid', 'customize', 'showLabels', 'orientLine',
    'orientDist', 'widthRatio'})
APPEARANCE_PREFIXES = ('custom', 'noise', 'colornoise', 'normalnoise',
                       'anisotropy')
# Solver sequence properties that hold notes of the COMSOL Desktop or the
# log of the last run
SOLVER_SKIPPED = frozenset({'message', 'lastchangedproperty',
                            'changedproperties', 'hiddenchangedproperties',
                            'physicsselectionmain', 'physicsselectionmg'})
# Values that leave a material property unset
EMPTY: tuple = ('', [], [''], [[]], [['']], None)


def describe(model: Model, /, *, solver: bool = False) -> dict:
    """
    Returns a model's settings as plain values (JSON-ready), independent
    of tags, feature order and entity numbers, e.g. to check that a
    script rebuilt a model made in the COMSOL Desktop:

    ```python
    d = mk.describe(model)
    with open('old.json', 'w', encoding='utf-8') as file:
        json.dump(d, file, indent=1, ensure_ascii=False)
    [hot] = [f for f in d['components'][0]['physics'][0]['features']
             if f['type'] == 'TemperatureBoundary']
    hot
    # {'tag': 'temp1', 'path': 'comp1/ht/temp1',
    #  'type': 'TemperatureBoundary', 'label': 'Temperature 1',
    #  'active': True, 'properties': {'T0': 'Th'},
    #  'defaults': {'T0': '293.15[K]'},
    #  'selection': {'level': 'boundary', 'entities': [
    #      {'x': [0.0, 0.0], 'y': [0.0, 50.0], 'z': [0.0, 10.0],
    #       'size': 500.0}]},
    #  'selections': {}, 'features': []}
    ```

    The result has `parameters` (expression and SI value), the model's
    `functions`, `variables`, `couplings` (operators), `coordinates`,
    `materials` and `definitions` (each with `component` unless global),
    `components` with their `geometries`, `pairs`, `physics`,
    `multiphysics` and `meshes`, and `studies` with their steps and
    solver; `notes` list ({'path', 'kind', 'message'}) what could not be
    read, which defaults are unknown, why a solver was not compared and
    what COMSOL changed for good. Results (plots, datasets, evaluations,
    tables) are left out.

    A node (feature, step, operator, ...) has `tag`, `path` (tags from
    the component or study down, e.g. 'comp1/ht/temp1'), `type`, `label`,
    `active`, `properties`, `defaults`, `selection` (or None),
    `selections` (more selections of the node by name, e.g. a periodic
    condition's 'destinationDomains') and `features` (its subnodes).
    `properties` holds only the values that differ from the defaults of a
    new node of that type, and `defaults` the default values of the same
    keys. Where no default is known, `unknown_defaults` lists the keys;
    a node whose type could not be created for defaults has
    `all_properties: True` and all its values. Study step properties
    that pair tags with values (`activate`, ...) become a dict of the
    pairs that differ, e.g. {'ec': 'off'}; `solnum` and `notsolnum` count
    '1' and 'auto' as the default (solving sets one to the other).
    Other entries differ from nodes: a physics interface has
    `identifier` (its name in expressions, e.g. 'ht'), `settings` and
    `defaults` (by 'group/name') instead of `properties`; a variables
    node has `variables` (name to expression) and no `type`; a material
    has `groups` (property groups with their values and functions); a
    pair has `type`, `source` and `destination`.

    A selection is `{'level': 'boundary', 'entities': [...]}`: one entry
    per entity with its bounding box per axis and its `size` (volume,
    area or length; none for points, whose box is their coordinates), in
    the geometry's length unit (`length_unit` of the geometry), or 'all'
    for all entities of that level. `applied` is where the node applies,
    if that differs: a node later in the same physics overrides it on
    the entities it shares, and default features apply where nothing else
    does. `named` is the label of a named selection it uses. Mesh
    operations on the whole geometry or on what is left have the level
    'remaining' (COMSOL tells them apart only before meshing); global
    nodes 'global'. Boxes are single precision (1.1 reads
    1.100000023841858) and curved entities are measured on a rendering
    mesh: compare with a tolerance. Geometries list all their entities
    this way under `entities`.

    A mesh has `automatic` (controlled by the physics, the COMSOL
    Desktop's default; meshes made through MPh are not) and
    `size_level`. A study's `solver` has the tag of its solver
    `sequence` and a `status`: 'automatic' when it has none yet (a
    script model before solving), else 'not compared' with the `reason`.
    With `solver=True`, the sequence is compared with the one COMSOL
    would create now: 'compared' with the `changes` (by tag path), or
    'not compared' with the reason: a mesh the study uses is not built
    (run `model.mesh()` and describe again), a component with physics
    has no mesh, the study is disabled, or COMSOL could not make its own
    sequence. Two solution tags count as equal, so a changed initial
    solution does not show. This
    makes a temporary sequence in the model and compiles the equations;
    describe removes what that leaves, as far as COMSOL lets it: in a
    model saved by another COMSOL version or build (see `saved_with`),
    COMSOL may update the existing sequences and build the empty meshes
    of layered materials, which `notes` then list ('model_changed'). Such
    a model may also show changes that are only version differences.
    Without `solver=True`, nothing in the model is changed.

    To compare two results by hand: tags and order differ, so look nodes
    up by `type` and selection, not by their position in the lists. Not
    described: named selections themselves (nodes show the entities they
    select), node groups, batch jobs and results.
    Expressions are kept as written ('100[degC]' and '373.15[K]'
    differ). The two faces of a pair in an assembly have the same box
    and size. `comsol` is the version running, `saved_with` the one that
    last saved the model. The geometries must be built
    (`model.build(geom)`).
    """
    if not isinstance(model, Model):
        raise TypeError(f'mk.describe takes a model, not {model!r}.')
    from . import __version__
    reader = _Reader(model)
    reader.check_geometries()
    with _skeleton(reader) as scratch:
        reader.scratch = scratch
        result = {
            'format': FORMAT, 'mphkit': __version__,
            'comsol': _comsol_version(),
            'saved_with': str(model.java.getComsolVersion()),
            'parameters': reader.parameters(),
            'functions': reader.listed('func'),
            'variables': reader.variables(),
            'couplings': reader.listed('cpl'),
            'coordinates': reader.listed('coordSystem'),
            'materials': reader.materials(),
            'definitions': reader.listed('common', subnodes=_create),
            'components': [reader.component(tag)
                           for tag in reader.components],
            'studies': [reader.study(tag, solver)
                        for tag in _tags(model.java.study())],
        }
    result['notes'] = reader.notes
    return result


###################
# Plain functions #
###################

def json_value(value):
    """
    Returns a value read from COMSOL as JSON-ready values: lists for
    tuples and arrays, and 'NaN', 'Infinity' or '-Infinity' for those
    floats, so that a result read back from JSON compares equal.
    """
    value = _catalog.plain(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def differs(values: dict, base: dict) -> tuple[dict, dict, list[str]]:
    """
    Returns the values that differ from those of a new node (`base`), the
    defaults of the same keys and the keys the new node does not have.
    """
    properties: dict = {}
    defaults: dict = {}
    unknown: list[str] = []
    for name, value in values.items():
        if name not in base:
            properties[name] = value
            unknown.append(name)
        elif base[name] != value:
            properties[name] = value
            defaults[name] = base[name]
    return properties, defaults, unknown


def pairs_of(value) -> dict | None:
    """
    Returns a list of [key, value, key, value, ...] as a dict, or None if
    it is not such a list.
    """
    if not isinstance(value, list) or len(value) % 2:
        return None
    keys = value[::2]
    if not all(isinstance(key, str) for key in keys) or \
            len(set(keys)) != len(keys):
        return None
    return dict(zip(keys, value[1::2]))


def step_map(value, base) -> tuple[dict, dict]:
    """
    Returns the pairs of a study step's pair list that differ from the
    default one, and their defaults; pairs the default lacks count as
    differing with a default of None. Older models lack a component's
    pair that new ones have: that is no difference.
    """
    own, default = pairs_of(value) or {}, pairs_of(base) or {}
    changed = {key: item for key, item in own.items()
               if default.get(key) != item}
    return changed, {key: default.get(key) for key in changed}


def owner(scope: str, components: set[str], tag: str | None = None
          ) -> str | None | Literal[False]:
    """
    Returns the component a node belongs to by its scope, None for a
    global node, or False for a node COMSOL made itself: one whose scope
    lies below a physics interface ('root.comp1.ht') or in a component
    that is not listed. Coordinate systems and materials have their own
    `tag` at the end of the scope ('root.comp1.sys1').
    """
    parts = scope.split('.')
    if tag is not None and len(parts) > 1 and parts[-1] == tag:
        parts = parts[:-1]
    if parts == ['root']:
        return None
    if len(parts) == 2 and parts[0] == 'root' and parts[1] in components:
        return parts[1]
    return False


def hidden_component(tag: str, materials: set[str]) -> bool:
    """
    Tells whether a component is one a layered material made
    ('mat4_xdim', "Extra dimension from ..."), not the user.
    """
    return tag.endswith('_xdim') and tag[:-len('_xdim')] in materials


def appearance(name: str) -> bool:
    """Tells whether a material property is about how it looks."""
    return name in APPEARANCE or name.startswith(APPEARANCE_PREFIXES)


def solver_changes(model_nodes: dict, automatic: dict, tags: dict,
                   solutions: set[str]) -> list[dict]:
    """
    Returns the differences between the active nodes of a solver sequence
    and those of the sequence COMSOL would create, both given as
    {tag path: (type, label path, values)}. `tags` maps tags of the
    temporary sequence to the model's own, also inside lists; two
    solution tags count as equal (a store-solution node names a new one
    each time), so a changed initial solution does not show.
    """
    def mapped(value):
        if isinstance(value, list):
            return [mapped(item) for item in value]
        return tags.get(value, value) if isinstance(value, str) else value

    def same(own, auto) -> bool:
        auto = mapped(auto)
        if own == auto:
            return True
        return (isinstance(own, str) and isinstance(auto, str)
                and own in solutions and auto in solutions)

    found = []
    for path, (kind, labels, values) in model_nodes.items():
        if path not in automatic or automatic[path][0] != kind:
            found.append({'path': path, 'labels': labels, 'type': kind,
                          'change': 'only_in_model'})
            continue
        auto = automatic[path][2]
        changed = {name: [values.get(name), auto.get(name)]
                   for name in sorted(set(values) | set(auto))
                   if not same(values.get(name), auto.get(name))}
        if changed:
            found.append({'path': path, 'labels': labels, 'type': kind,
                          'change': 'property', 'properties': changed})
    for path, (kind, labels, _) in automatic.items():
        if path not in model_nodes or model_nodes[path][0] != kind:
            found.append({'path': path, 'labels': labels, 'type': kind,
                          'change': 'only_in_automatic'})
    return found


################
# Java objects #
################

def _tags(container) -> list[str]:
    """Returns the tags of a Java list, [] if it cannot be read."""
    try:
        return [str(tag) for tag in container.tags()]
    except Exception:
        return []


def _label(java) -> str:
    try:
        return str(java.label())
    except Exception:
        return ''


def _active(java) -> bool:
    try:
        return bool(java.isActive())
    except Exception:
        return True


def _type(java) -> str:
    try:
        return str(java.getType())
    except Exception:
        return ''


def _level(java) -> int | None:
    """Returns the one entity level of a node's selection, or None."""
    try:
        dims = _comsol.selection_dims(java.selection())
    except Exception:
        return None
    return dims[0] if len(dims) == 1 else None


class _Reader:
    """Reads a model's nodes, with the temporary model for defaults."""

    def __init__(self, model: Model):
        self.model = model
        self.java = model.java
        self.scratch: Any = None
        self.notes: list[dict] = []
        materials = set(_tags(self.java.material()))
        self.components = [tag for tag in _tags(self.java.component())
                           if not hidden_component(tag, materials)]
        self.geometries: dict[str, Any] = {}
        for ctag in self.components:
            component = self.java.component(ctag)
            for gtag in _tags(component.geom()):
                self.geometries[gtag] = component.geom(gtag)
        self.places: dict[tuple[str, int, int], dict] = {}
        # what is being read: 'mesh' and 'study' nodes get their own rules
        self.context = ''
        self.measures: dict[str, Any] = {}
        self.vertices: dict[str, list[list[float]]] = {}
        self.blanks: dict[tuple[str | None, str], Any] = {}

    def note(self, path: str, kind: str, message: str):
        self.notes.append({'path': path, 'kind': kind, 'message': message})

    def check_geometries(self):
        for geometry in self.geometries.values():
            _comsol.check_geometry_built(geometry,
                                         _comsol.name_of(geometry))

    # Values

    def read(self, java, path: str, skipped=SKIPPED,
             noted: bool = True) -> dict:
        """
        Returns the properties of a Java node as JSON-ready values; those
        that cannot be read are left out, and noted unless `noted` is
        off (for the temporary model's nodes).
        """
        values: dict = {}
        try:
            names = [str(name) for name in java.properties()]
        except Exception:
            return values
        for name in names:
            if name in skipped:
                continue
            try:
                if _comsol.value_type(java, name) == 'Selection':
                    continue
                values[name] = json_value(get(java, name))
            except Exception as error:
                if noted:
                    self.note(path, 'unreadable', f'property {name!r}: '
                              f'{_comsol.reason(error)}')
        return values

    def node(self, java, base, path: str, subnodes=None,
             kept=None) -> dict:
        """
        Describes a node against its counterpart `base` in the temporary
        model (None if there is none). `subnodes` creates the
        counterparts of its own subnodes; `kept` tells which property
        names count.
        """
        values = self.read(java, path)
        if kept is not None:
            values = {name: v for name, v in values.items() if kept(name)}
        kind = _type(java)
        entry: dict = {'tag': str(java.tag()), 'path': path, 'type': kind,
                       'label': _label(java), 'active': _active(java)}
        properties: dict
        defaults: dict
        unknown: list[str]
        maps = STEP_MAPS & set(values) if self.context == 'study' \
            else set()
        if base is None:
            for name in maps:
                values[name] = pairs_of(values[name]) or values[name]
            properties, defaults, unknown = values, {}, []
            entry['all_properties'] = True
            self.note(path, 'defaults_unknown',
                      f'no new {kind} node could be made for defaults; '
                      'all properties are listed')
        else:
            base_values = self.read(base, path, noted=False)
            for name in maps:
                if pairs_of(values[name]) is None:
                    continue
                changed, default = step_map(values[name],
                                            base_values.get(name))
                values[name] = changed
                base_values[name] = default if changed else changed
            # the first compile sets these from '1' to 'auto'
            for name in _comsol.FIRST_COMPILE:
                if str(values.get(name)) in ('1', 'auto') and \
                        str(base_values.get(name)) in ('1', 'auto'):
                    base_values[name] = values[name]
            properties, defaults, unknown = differs(values, base_values)
        if kind in MESH_SIZES and values.get('custom') == 'off':
            for name in DERIVED_SIZES:
                properties.pop(name, None)
                defaults.pop(name, None)
                if name in unknown:
                    unknown.remove(name)
        if unknown and base is not None:
            entry['unknown_defaults'] = unknown
            self.note(path, 'defaults_unknown', 'no default for '
                      + ', '.join(repr(name) for name in unknown))
        entry['properties'] = properties
        entry['defaults'] = defaults
        entry['selection'] = self.selection(java, path)
        entry['selections'] = self.extra_selections(java, path)
        entry['features'] = [] if subnodes is None else \
            self.nodes(java, base, path, subnodes)
        return entry

    def nodes(self, parent, base_parent, path: str, make) -> list[dict]:
        """
        Describes the subnodes of a node (its `feature()` list); `make`
        creates a counterpart in the temporary model's list.
        """
        try:
            container = parent.feature()
        except Exception:
            return []
        try:
            base_container = base_parent.feature() \
                if base_parent is not None else None
        except Exception:
            base_container = None
        found = []
        for tag in _tags(container):
            java = container.get(tag)
            base = self.counterpart(base_container, java, make)
            found.append(self.node(java, base, f'{path}/{tag}',
                                   subnodes=make))
        return found

    def counterpart(self, container, java, make) -> Any:
        """
        Returns the node of the same tag and type in a list of the
        temporary model, created there if missing, or None. A node that
        fails half-made is removed: it would make later ones fail too.
        """
        if container is None:
            return None
        tag, kind = str(java.tag()), _type(java)
        if tag in _tags(container):
            base = container.get(tag)
            return base if _type(base) == kind else None
        try:
            make(container, java)
            return container.get(tag)
        except Exception:
            if tag in _tags(container):
                try:
                    container.remove(tag)
                except Exception:
                    pass
            return None

    # Selections

    def selection(self, java, path: str, name: str | None = None,
                  geometry: str | None = None) -> dict | None:
        """Describes a node's selection (or one by name), or None."""
        try:
            selection = java.selection(name) if name else java.selection()
            dims = _comsol.selection_dims(selection)
        except Exception:
            return None
        return self.described(selection, dims, path, geometry)

    def described(self, selection, dims: list[int], path: str,
                  geometry: str | None = None) -> dict:
        gtag = geometry
        try:
            value = selection.geom()
            if value is not None and str(value) in self.geometries:
                gtag = str(value)
        except Exception:
            pass
        if gtag is None:
            return {'level': 'global'}
        sdim = int(self.geometries[gtag].getSDim())
        # A mesh operation on what is left reads like one on the whole
        # geometry once the mesh is built, and has no levels before
        whole = sorted(dims) == list(range(sdim + 1))
        if not dims or whole and self.context == 'mesh':
            return {'level': 'remaining'}
        if whole:
            return {'level': 'geometry'}
        if len(dims) > 1:
            self.note(path, 'levels_unknown', 'a selection on several '
                      f'levels {dims}: its entities are not described')
            return {'levels': [_comsol.entity_level_name(d, sdim)
                               for d in dims], 'entities': 'unknown'}
        dim = dims[0]
        try:
            given = selection.inputEntities()
        except Exception:
            given = None
        applied = [int(e) for e in selection.entities()]
        chosen = applied if given is None else [int(e) for e in given]
        found: dict = {'level': _comsol.entity_level_name(dim, sdim),
                       'entities': self.located(gtag, dim, chosen, path)}
        if sorted(set(chosen)) != sorted(set(applied)):
            found['applied'] = self.located(gtag, dim, applied, path)
        try:
            named = str(selection.named())
            if named:
                found['named'] = _label(self.java.selection(named)) or named
        except Exception:
            pass
        return found

    def located(self, gtag: str, dim: int, numbers: list[int], path: str):
        """Returns the places of entities, or 'all' for all of a level."""
        count = _comsol.count_entities(self.geometries[gtag], dim)
        if count and sorted(set(numbers)) == list(range(1, count + 1)):
            return 'all'
        return [self.place(gtag, dim, n, path) for n in numbers]

    def place(self, gtag: str, dim: int, number: int, path: str) -> dict:
        """Returns the bounding box and size of one entity."""
        key = (gtag, dim, number)
        if key in self.places:
            return self.places[key]
        geometry = self.geometries[gtag]
        sdim = int(geometry.getSDim())
        found: dict
        try:
            if not 1 <= number <= _comsol.count_entities(geometry, dim):
                raise ValueError('no such entity')
            if dim == 0:
                if gtag not in self.vertices:
                    self.vertices[gtag] = [
                        [float(v) for v in row]
                        for row in geometry.getVertexCoord()]
                found = {AXES[i]: [row[number - 1]] * 2 for i, row in
                         enumerate(self.vertices[gtag][:sdim])}
            else:
                if gtag not in self.measures:
                    self.measures[gtag] = geometry.measureFinal()
                measure = self.measures[gtag]
                with _comsol.history_off(self.java):
                    measure.selection().geom(gtag, dim)
                    measure.selection().set([number])
                    box = [float(v) for v in measure.getBoundingBox()]
                    size = float(measure.getVolume())
                found = {AXES[i]: json_value(box[2*i:2*i + 2])
                         for i in range(len(box) // 2)}
                found['size'] = json_value(size)
        except Exception as error:
            self.note(path, 'place_unknown', f'entity {number} at level '
                      f'{dim} of "{gtag}": {_comsol.reason(error)}')
            found = {'unknown': number}
        self.places[key] = found
        return found

    def extra_selections(self, java, path: str) -> dict:
        try:
            names = [str(n) for n in java.getExtraSelectionNames()]
        except Exception:
            return {}
        found = {}
        for name in names:
            described = self.selection(java, path, name)
            if described is not None:
                found[name] = described
        return found

    # Model level

    def parameters(self) -> dict:
        params = self.java.param()
        found = {}
        for name in [str(n) for n in params.varnames()]:
            try:
                value = json_value(float(params.evaluate(name)))
            except Exception:       # e.g. a complex value
                value = None
            found[name] = {'expression': str(params.get(name)),
                           'value': value}
        return found

    def scratch_list(self, name: str, component: str | None):
        """Returns a list of the temporary model, of a component or not."""
        try:
            holder = self.scratch if component is None \
                else self.scratch.component(component)
            return getattr(holder, name)()
        except Exception:
            return None

    def listed(self, name: str, subnodes=None) -> list[dict]:
        """
        Describes the nodes of one of the model's lists (functions,
        operators, coordinate systems, definitions), which hold the
        components' nodes too, with the component they belong to; nodes
        COMSOL made itself are left out.
        """
        container = getattr(self.java, name)()
        found = []
        for tag in _tags(container):
            java = container.get(tag)
            component = owner(str(java.scope()), set(self.components),
                              tag if name == 'coordSystem' else None)
            if component is False:
                continue
            path = tag if component is None else f'{component}/{tag}'
            base = self.counterpart(self.scratch_list(name, component),
                                    java, _create)
            entry = self.node(java, base, path, subnodes=subnodes)
            if component is not None:
                entry['component'] = component
            found.append(entry)
        return found

    def variables(self) -> list[dict]:
        found = []
        container = self.java.variable()
        for tag in _tags(container):
            java = container.get(tag)
            component = owner(str(java.scope()), set(self.components))
            if component is False or \
                    _comsol.DERIVED_VARIABLES.fullmatch(tag):
                continue
            path = tag if component is None else f'{component}/{tag}'
            entry: dict = {'tag': tag, 'path': path, 'label': _label(java),
                           'active': _active(java)}
            if component is not None:
                entry['component'] = component
            entry['variables'] = {str(n): str(java.get(n))
                                  for n in java.varnames()}
            entry['selection'] = self.selection(java, path)
            found.append(entry)
        return found

    def materials(self) -> list[dict]:
        found = []
        container = self.java.material()
        for tag in _tags(container):
            java = container.get(tag)
            component = owner(str(java.scope()), set(self.components), tag)
            if component is False:
                continue
            path = tag if component is None else f'{component}/{tag}'
            entry = self.material(java, component, path)
            if component is not None:
                entry['component'] = component
            found.append(entry)
        return found

    def material(self, java, component: str | None, path: str) -> dict:
        """Describes a material, a link or a switch with its materials."""
        entry = self.node(java, self.blank(component, _type(java)), path,
                          kept=lambda name: not appearance(name))
        groups = {}
        try:
            group_tags = _tags(java.propertyGroup())
        except Exception:
            group_tags = []
        for gtag in group_tags:
            group = java.propertyGroup(gtag)
            values = {name: value for name, value in
                      self.read(group, f'{path}/{gtag}').items()
                      if value not in EMPTY}
            described: dict = {'properties': values}
            functions = {}
            try:
                function_list = group.func()
            except Exception:
                function_list = None
            for ftag in _tags(function_list):
                function = group.func(ftag)
                functions[ftag] = {
                    'type': _type(function),
                    **{name: value for name, value in
                       self.read(function, f'{path}/{gtag}/{ftag}').items()
                       if value not in EMPTY}}
            if functions:
                described['functions'] = functions
            if values or functions:
                groups[gtag] = described
        entry['groups'] = groups
        members = []
        for mtag in _tags(java.feature()) if hasattr(java, 'feature') \
                else []:
            member = java.feature(mtag)
            members.append(self.material(member, component,
                                         f'{path}/{mtag}'))
        entry['features'] = members
        return entry

    def blank(self, component: str | None, kind: str) -> Any:
        """Returns a new material of a type in the temporary model."""
        key = (component, kind)
        if key not in self.blanks:
            container = self.scratch_list('material', component)
            base = None
            if container is not None:
                tag = str(container.uniquetag(SCRATCH))
                try:
                    if component is None:
                        container.create(tag, kind, '')
                    else:
                        container.create(tag, kind)
                    base = container.get(tag)
                except Exception:
                    base = None
            self.blanks[key] = base
        return self.blanks[key]

    # Components

    def component(self, ctag: str) -> dict:
        java = self.java.component(ctag)
        scratch = self.scratch_list('component', None)
        scomp = scratch.get(ctag) if scratch is not None and \
            ctag in _tags(scratch) else None
        return {
            'tag': ctag, 'label': _label(java),
            'geometries': [self.geometry(gtag)
                           for gtag in _tags(java.geom())],
            'pairs': self.pairs(java, ctag),
            'physics': [self.physics(java, scomp, ctag, ptag)
                        for ptag in _tags(java.physics())],
            'multiphysics': self.nodes_of(
                java.multiphysics(),
                scomp.multiphysics() if scomp is not None else None,
                ctag, _create_physics),
            'meshes': [self.mesh(java, scomp, ctag, mtag)
                       for mtag in _tags(java.mesh())],
        }

    def nodes_of(self, container, base_container, path: str,
                 make) -> list[dict]:
        """Describes the nodes of a list, e.g. a component's multiphysics."""
        found = []
        for tag in _tags(container):
            java = container.get(tag)
            base = self.counterpart(base_container, java, make)
            found.append(self.node(java, base, f'{path}/{tag}',
                                   subnodes=make))
        return found

    def geometry(self, gtag: str) -> dict:
        java = self.geometries[gtag]
        sdim = int(java.getSDim())
        try:
            box = json_value([float(v) for v in java.getBoundingBox()])
        except Exception:       # e.g. an empty geometry
            box = None
        entities = {}
        for dim in range(sdim, -1, -1):
            name = _comsol.entity_level_name(dim, sdim)
            count = _comsol.count_entities(java, dim)
            entities[name] = [self.place(gtag, dim, n, gtag)
                              for n in range(1, count + 1)]
        try:
            axisymmetric = bool(java.isAxisymmetric())
        except Exception:
            axisymmetric = False
        return {'tag': gtag, 'label': _label(java), 'dimension': sdim,
                'axisymmetric': axisymmetric,
                'length_unit': str(java.lengthUnit()),
                'voids': _count(java.getNFiniteVoids),
                'bounding_box': None if box is None else
                {AXES[i]: box[2*i:2*i + 2] for i in range(sdim)},
                'entities': entities}

    def pairs(self, component, ctag: str) -> list[dict]:
        found = []
        geometries = _tags(component.geom())
        default = geometries[0] if geometries else None
        for tag in _tags(component.pair()):
            java = component.pair(tag)
            path = f'{ctag}/{tag}'
            entry: dict = {'tag': tag, 'path': path, 'label': _label(java),
                           'type': str(java.type()), 'active': _active(java)}
            for side in ('source', 'destination'):
                try:
                    selection = getattr(java, side)()
                    entry[side] = self.described(
                        selection, _comsol.selection_dims(selection), path,
                        default)
                except Exception as error:
                    entry[side] = None
                    self.note(path, 'unreadable',
                              f'{side}: {_comsol.reason(error)}')
            found.append(entry)
        return found

    def physics(self, component, scomp, ctag: str, ptag: str) -> dict:
        java = component.physics(ptag)
        path = f'{ctag}/{ptag}'
        base = None
        if scomp is not None and ptag in _tags(scomp.physics()):
            base = scomp.physics(ptag)
            if _type(base) != _type(java):
                base = None
        settings: dict = {}
        defaults: dict = {}
        for group in java.prop():
            gtag = str(group.tag())
            values = {f'{gtag}/{name}': value for name, value in
                      self.read(group, path).items()}
            for name in SKIPPED_SETTINGS & set(values):
                del values[name]
            if base is None:
                settings.update(values)
                continue
            try:
                base_values = {f'{gtag}/{name}': value for name, value in
                               self.read(base.prop(gtag), path,
                                         noted=False).items()}
            except Exception:
                base_values = {}
            changed, default, _ = differs(values, base_values)
            settings.update(changed)
            defaults.update(default)
        entry: dict = {'tag': ptag, 'path': path,
                       'identifier': str(java.identifier()),
                       'type': _type(java), 'label': _label(java),
                       'active': _active(java)}
        if base is None:
            entry['all_properties'] = True
            self.note(path, 'defaults_unknown', 'the physics interface '
                      'could not be made for defaults; all properties '
                      'are listed')
        entry['settings'] = settings
        entry['defaults'] = defaults
        entry['selection'] = self.selection(java, path)
        entry['features'] = self.nodes(java, base, path, _create_physics)
        return entry

    def mesh(self, component, scomp, ctag: str, mtag: str) -> dict:
        java = component.mesh(mtag)
        path = f'{ctag}/{mtag}'
        base = None
        if scomp is not None and mtag in _tags(scomp.mesh()):
            base = scomp.mesh(mtag)
        geometries = _tags(component.geom())
        try:
            size_level: Any = float(java.autoMeshSize())
            size_level = int(size_level) if size_level.is_integer() \
                else size_level
        except Exception:
            size_level = None
        self.context = 'mesh'
        try:
            features = self.nodes(java, base, path, _create)
        finally:
            self.context = ''
        return {'tag': mtag, 'path': path, 'label': _label(java),
                'geometry': geometries[0] if geometries else None,
                'automatic': bool(java.isAutomatic()),
                'size_level': size_level, 'features': features}

    # Studies

    def study(self, stag: str, solver: bool) -> dict:
        java = self.java.study(stag)
        base = None
        studies = self.scratch_list('study', None)
        if studies is not None and stag in _tags(studies):
            base = studies.get(stag)
        self.context = 'study'
        try:
            steps = self.nodes(java, base, stag, _create)
        finally:
            self.context = ''
        return {'tag': stag, 'path': stag, 'label': _label(java),
                'active': _active(java), 'steps': steps,
                'solver': self.solver(java, stag, solver)}

    def solver(self, study, stag: str, compare: bool) -> dict:
        java = self.java
        sequence = _solve.attached_sequence(java, stag)
        tag = str(sequence.tag()) if sequence is not None else None
        if sequence is None:
            return {'status': 'automatic', 'sequence': None}
        if not compare:
            return self.not_compared(
                stag, tag, 'not asked for: mk.describe(model, solver=True) '
                'compares it', note=False)
        if not _active(study):
            return self.not_compared(stag, tag, 'study disabled')
        reason = self.unready(study)
        if reason is not None:
            return self.not_compared(stag, tag, reason)
        # the model's own nodes first: compiling may update them
        own = self.solver_nodes(sequence)
        sequences = java.sol()
        automatic: dict | None = None
        with _comsol.history_off(java), \
                _comsol.compiled_traces_removed(java) as changed:
            temporary = str(sequences.uniquetag(SCRATCH))
            try:
                made = sequences.create(temporary)
                made.study(stag)
                made.createAutoSequence(stag)
                automatic = self.solver_nodes(made)
            except Exception as error:
                reason = ('COMSOL could not make its own solver sequence: '
                          f'{_comsol.reason(error)}')
            finally:
                if temporary in _tags(sequences):
                    sequences.remove(temporary)
        for message in changed:
            self.note(stag, 'model_changed', message)
        if automatic is None:
            return self.not_compared(stag, tag, str(reason))
        changes = solver_changes(own, automatic, {temporary: str(tag)},
                                 set(_tags(sequences)) | {temporary})
        return {'status': 'compared', 'sequence': tag, 'changes': changes}

    def not_compared(self, stag: str, tag: str | None, reason: str,
                     note: bool = True) -> dict:
        if note:
            self.note(stag, 'solver_not_compared', reason)
        return {'status': 'not compared', 'sequence': tag,
                'reason': reason}

    def unready(self, study) -> str | None:
        """
        Returns why a study's meshes are not ready for a temporary solver
        sequence, which would build them silently, or None.
        """
        try:
            interfaces = _check.active_physics(self.model)
        except Exception as error:
            return _comsol.reason(error)
        with_physics = {str(p.geometry.tag()) for p in interfaces}
        for gtag, mtag in _solve.used_meshes(study, interfaces):
            if mtag == 'nomesh':
                if gtag in with_physics:
                    return (f'geometry "{gtag}" has physics but no mesh; '
                            'create one, run model.mesh() and describe '
                            'again')
                continue
            try:
                _comsol.check_mesh_built(self.java.mesh(mtag))
            except Exception as error:
                return (f'{_comsol.reason(error)} Then describe again.')
        return None

    def solver_nodes(self, sequence) -> dict:
        """Returns the active nodes of a solver sequence by tag path."""
        found: dict = {}

        def walk(node, path: str, labels: str):
            for tag in _tags(node.feature()):
                child = node.feature(tag)
                if not _active(child):
                    continue
                own = f'{path}/{tag}' if path else tag
                label = f'{labels}/{_label(child)}' if labels \
                    else _label(child)
                found[own] = (_type(child), label,
                              self.read(child, own,
                                        SKIPPED | SOLVER_SKIPPED))
                walk(child, own, label)

        walk(sequence, '', '')
        return found


def _comsol_version() -> str:
    """Returns the version of COMSOL that runs."""
    import jpype  # type: ignore[import-untyped]
    util = jpype.JClass('com.comsol.model.util.ModelUtil')
    return str(util.getComsolVersion())


def _count(read) -> int | None:
    """Returns a count COMSOL gives, or None if it cannot."""
    try:
        return int(read())
    except Exception:
        return None


def _create(container, java):
    container.create(str(java.tag()), _type(java))


def _create_physics(container, java):
    """Creates a physics feature at its selection's level, if it has one."""
    level = _level(java)
    if level is None:
        container.create(str(java.tag()), _type(java))
    else:
        container.create(str(java.tag()), _type(java), level)


@contextmanager
def _skeleton(reader: _Reader) -> Iterator[Any]:
    """
    Yields a temporary model with the components, geometries, physics
    interfaces, multiphysics couplings, meshes and studies of the user's
    model (same tags, no features), removed afterwards.
    """
    import jpype  # type: ignore[import-untyped]
    util = jpype.JClass('com.comsol.model.util.ModelUtil')
    tag = str(util.uniquetag(SCRATCH))
    java = reader.java
    try:
        scratch = util.create(tag)
        for ctag in reader.components:
            component = java.component(ctag)
            scomp = scratch.component().create(ctag, True)
            for gtag in _tags(component.geom()):
                _scratch_geometry(scomp, component.geom(gtag))
            for ptag in _tags(component.physics()):
                physics = component.physics(ptag)
                try:
                    geometry = physics.geom()
                    if geometry is None:
                        made = scomp.physics().create(ptag, _type(physics))
                    else:
                        made = scomp.physics().create(
                            ptag, _type(physics), str(geometry))
                    made.identifier(str(physics.identifier()))
                except Exception:
                    pass        # its nodes list all their properties
            for mtag in _tags(component.multiphysics()):
                coupling = component.multiphysics(mtag)
                try:
                    level = _level(coupling)
                    gtag = str(coupling.selection().geom())
                    if level is None:
                        scomp.multiphysics().create(mtag, _type(coupling),
                                                    gtag)
                    else:
                        scomp.multiphysics().create(mtag, _type(coupling),
                                                    gtag, level)
                except Exception:
                    pass
            geometries = _tags(component.geom())
            for mtag in _tags(component.mesh()):
                try:
                    if geometries:
                        scomp.mesh().create(mtag, geometries[0])
                    else:
                        scomp.mesh().create(mtag)
                except Exception:
                    pass
        for stag in _tags(java.study()):
            scratch.study().create(stag)
        yield scratch
    finally:
        if tag in [str(t) for t in util.tags()]:
            util.remove(tag)


def _scratch_geometry(component, geometry):
    """
    Makes a geometry like the user's in a temporary component: same tag,
    dimension, axisymmetry and length unit, with one block as large as the
    user's geometry (mesh size defaults follow the geometry's size). Sides
    of size 0 (a shell plate, a wire) become 0.001 of the largest side: a
    block cannot be flat.
    """
    gtag, sdim = str(geometry.tag()), int(geometry.getSDim())
    made = component.geom().create(gtag, sdim)
    try:
        if geometry.isAxisymmetric():
            made.axisymmetric(True)
    except Exception:
        pass
    made.lengthUnit(str(geometry.lengthUnit()))
    if sdim == 0:
        return
    try:
        box = [float(v) for v in geometry.getBoundingBox()]
        lows = box[0::2][:sdim]
        sides = [high - low for low, high in zip(lows, box[1::2][:sdim])]
        if not all(math.isfinite(v) for v in lows + sides):
            raise ValueError('no bounding box')
    except Exception:
        lows, sides = [0.0] * sdim, [1.0] * sdim
    largest = max(sides) if max(sides) > 0 else 1.0
    sides = [side if side > 0 else 1e-3 * largest for side in sides]
    if sdim == 1:
        made.create('mkbox', 'Interval').set(
            'coord', [str(lows[0]), str(lows[0] + sides[0])])
    else:
        block = made.create('mkbox', 'Block' if sdim == 3 else 'Rectangle')
        block.set('size', [str(side) for side in sides])
        block.set('pos', [str(low) for low in lows])
    made.run()
