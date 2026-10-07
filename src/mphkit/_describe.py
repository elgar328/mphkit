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

# Version of the result's layout; raise it when the layout changes or
# what describe leaves out as unused
FORMAT = 6
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
# Study steps that write their solutions to `filename` only with `save` on
SWEEPS = frozenset({'Parametric', 'MaterialSweep', 'FunctionSweep',
                    'BatchSweep'})
# A port's excitation values, unused while `PortExcitation` is off
EXCITATION = ('pamp', 'P0', 'IncidentWave')
# A probe's surface and volume integrals, which change nothing in 3D
INTEGRALS = ('intsurface', 'intvolume')
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
# Metres per length unit, for the units COMSOL accepts for a geometry
LENGTH_SCALES = {'m': 1.0, 'mm': 1e-3, 'cm': 1e-2, 'dm': 0.1, 'km': 1e3,
                 'µm': 1e-6, 'nm': 1e-9, 'Å': 1e-10, 'in': 0.0254,
                 'ft': 0.3048, 'yd': 0.9144, 'mi': 1609.344, 'nmi': 1852.0,
                 'mil': 2.54e-5, 'µin': 2.54e-8}
# Properties that hold the name expressions call a node by (default: the
# tag); a file interpolation names its functions in `funcnametable`
NAMES = ('funcname', 'opname', 'probename')
# Properties of a global equations feature with one entry per equation
ROWS = ('name', 'equation', 'initialValueU', 'initialValueUt', 'description')


def describe(model: Model, /, *, solver: bool = False) -> dict:
    """
    Returns a model's settings as plain values (JSON-ready), independent
    of tags, feature order and entity numbers, e.g. to check that a
    script rebuilt a model made in the COMSOL Desktop:

    ```python
    import json
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
    #  'si': {'T0': {'value': 373.15, 'unit': 'K'}},
    #  'si_defaults': {'T0': {'value': 293.15, 'unit': 'K'}},
    #  'selection': {'level': 'boundary', 'geometry': 'geom1',
    #      'entities': [{'x': [0.0, 0.0], 'y': [0.0, 50.0],
    #                    'z': [0.0, 10.0], 'size': 500.0}]},
    #  'selections': {}, 'features': []}
    ```

    The result has `format` (6; `mk.compare` takes this one only) and
    `mphkit` (the version that described), `parameters` (expression, SI
    value and SI unit), the model's `functions`, `variables`, `couplings`
    (operators),
    `coordinate_systems`, `materials`, `definitions` and `probes` (each with
    `component` unless global), `components` with their `geometries`,
    `pairs`, `physics`, `multiphysics`, `meshes` and `mass_properties`,
    `studies` with their steps and solver, and the tags of the model's
    `solutions`; `notes` list ({'path', 'kind', 'message'}) what could
    not be read ('unreadable', 'place_unknown', 'levels_unknown',
    'length_unknown'), which defaults are unknown ('defaults_unknown'),
    why a solver was not compared ('solver_not_compared') and what
    COMSOL changed for good ('model_changed'). Results (plots, datasets,
    evaluations, tables) are left out.

    A node (feature, step, operator, ...) has `tag`, `path` (tags from
    the component or study down, e.g. 'comp1/ht/temp1'), `type`, `label`,
    `active`, `properties`, `defaults`, `selection` (or None),
    `selections` (more selections of the node by name, e.g. a periodic
    condition's 'destinationDomains') and `features` (its subnodes).
    `properties` holds only the values that differ from the defaults of a
    new node of that type, and `defaults` the default values of the same
    keys. Where no default is known, `unknown_defaults` lists the keys;
    a node whose type could not be created for defaults has
    `all_properties: True` and all its values. `si` and `si_defaults`
    give the values COMSOL can evaluate with the model's parameters as
    {'value', 'unit'} in SI units (a list of them for a list), so that
    '100[degC]' and '373.15[K]' compare equal; choices among named
    options are not evaluated. `unused` lists properties the node's
    other settings leave unused (alternatives a choice does not pick,
    values whose switch is off, a value `p` while `p_src` takes it from
    elsewhere, mesh sizes and their switches while `custom` is off, the
    predefined size `hauto` while it is on, a sweep's `filename` while
    `save` is off, a port's excitation values while `PortExcitation` is
    off and a probe's `intsurface` and `intvolume` in 3D); they are left
    out. `names` lists the names expressions call a function,
    operator, probe or mass properties node by. Global equations keep
    their per-equation lists under `rows` (and the defaults of one row
    under `row_defaults`). Study step properties that pair tags with
    values (`activate`, ...) become a dict of the pairs that differ, e.g.
    {'ec': 'off'}; `solnum` and `notsolnum` count '1' and 'auto' as the
    default (solving sets one to the other). `unused`, `names`,
    `unknown_defaults`, `all_properties`, `rows`, `row_defaults` and,
    in selections, `applied`, `exterior` and `named` are there only
    where they apply.
    Other entries differ from nodes: a physics interface has
    `identifier` (its name in expressions, e.g. 'ht'), `settings` and
    `defaults` (by 'group/name') instead of `properties`, with `si`,
    `si_defaults` and `unknown_defaults` the same way; a variables
    node has `variables` (name to expression) and no `type`; a material
    has `groups` (property groups with their values, `si` and
    functions); a pair has `type`, `source` and `destination`.

    A selection is `{'level': 'boundary', 'geometry': 'geom1', 'entities':
    [...]}`: one entry per entity with its bounding box per axis and its
    `size` (volume, area or length; none for points, whose box is their
    coordinates), in the geometry's length unit (`length_unit` of the
    geometry, `length_scale` metres), or 'all' for all entities of that
    level. `applied` is where the node applies, if that differs: a node
    later in the same physics overrides it on the entities it shares, and
    default features apply where nothing else does. `named` is the label of
    a named selection it uses. Mesh operations on the whole geometry or on
    what is left have the level 'remaining' (COMSOL tells them apart only
    before meshing); other nodes on the whole geometry 'geometry', global
    nodes 'global', one that selects nothing 'none', and a selection on
    several levels 'several' (its entities 'unknown', its `levels` listed).
    An entity that could not be measured is {'unknown': number}. Domain 0,
    the exterior of boundary elements, is not an entity of the geometry: a
    selection that has it has `exterior: True` (and `exterior_applied` where
    that differs). Boxes are single precision (1.1 reads 1.100000023841858)
    and curved entities are measured on a rendering mesh: compare with a
    tolerance. Geometries list all their entities this way under `entities`,
    with their `dimension`, `axisymmetric`, number of `voids`,
    `bounding_box`, `finalize` (whether they form a union or an assembly)
    and `representation` (the geometry kernel: 'comsol' or 'cadps', the CAD
    kernel; they measure curved entities slightly differently).

    A mesh has `automatic` (controlled by the physics, the COMSOL
    Desktop's default; meshes made through MPh are not) and
    `size_level`. A study's `solver` has the tag of its solver
    `sequence` and a `status`: 'automatic' when it has none yet (a
    script model before solving), else 'not_asked'. With
    `solver=True`, the sequence is compared with the one COMSOL
    would create now: 'compared' with the `changes` (each with `path`,
    `labels`, `type`, `change`: 'property', 'only_in_model' or
    'only_in_automatic', and `properties`: name to [model's value,
    COMSOL's value]), or
    'not_compared' with the reason: a mesh the study uses is not built
    (run `model.mesh()` and describe again), a component with physics
    has no mesh, the study is disabled, or COMSOL could not make its own
    sequence. Two solution tags count as equal, so a changed initial
    solution does not show. This makes a temporary sequence in the model
    and compiles the equations; describe sets back what that changes
    (settings of study steps and existing sequences) and removes what it
    adds, as far as COMSOL lets it: in a
    model saved by another COMSOL version or build (see `saved_with`),
    COMSOL may update the existing sequences and build the empty meshes
    of layered materials, which `notes` then list ('model_changed'). Such
    a model may also show changes that are only version differences.
    As solving does, compiling may also remove, remake or change nodes of
    COMSOL's own that an earlier solve made (e.g. a derived variable
    iexpr_root_freq); they are not in the Java export. Without
    `solver=True`, nothing in the model is changed.

    `mk.compare` lists the differences between two results (or
    models). Not described: named selections themselves (nodes show the
    entities they select), node groups, batch jobs and results.
    Expressions are kept as written; `si` makes the evaluable ones
    comparable. The two faces of a pair in an assembly have the same box
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
            'coordinate_systems': reader.listed('coordSystem'),
            'materials': reader.materials(),
            'definitions': reader.listed('common', subnodes=_create),
            'probes': reader.listed('probe', subnodes=_create),
            'components': [reader.component(tag)
                           for tag in reader.components],
            'studies': [reader.study(tag, solver)
                        for tag in _tags(model.java.study())],
            'solutions': _tags(model.java.sol()),
        }
    result['notes'] = reader.notes
    return result


###################
# Plain functions #
###################

def json_value(value) -> Any:
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


def names_of(values: dict, kind: str = '') -> list[str]:
    """
    Returns the names expressions call a node by: its function, operator,
    probe or mass properties name, and for an interpolation read from a
    file the names of its functions.
    """
    keys = NAMES + ('name',) if kind == 'MassProperties' else NAMES
    found = [values[key] for key in keys
             if isinstance(values.get(key), str) and values[key]]
    table = values.get('funcnametable')
    if values.get('source') == 'file' and isinstance(table, list):
        found += [row[0] for row in table if isinstance(row, list) and row
                  and isinstance(row[0], str) and row[0]]
    return found


def has_rows(values: dict) -> bool:
    """Tells whether a node lists equations by row (global equations)."""
    return isinstance(values.get('name'), list) and \
        isinstance(values.get('equation'), list)


def unused_of(values: dict, allowed, kind: str = '',
              probe_sdim: int | None = None) -> set[str]:
    """
    Returns the properties of a node that its other settings leave
    unused: the alternatives a choice property does not pick (its allowed
    values, from `allowed(name)`, are all names of the node's
    properties), a property `p` whose switch `pactive` is off, a property
    `p` whose source `p_src` takes the value from elsewhere instead of
    its option 'userdef', the mesh sizes COMSOL derives and their
    switches while `custom` is off, the predefined size `hauto` while it
    is on, the physics of mass properties whose density does not come
    from a chosen physics, the file name of a sweep that does not save
    to a file, the excitation values of a port that is not excited, and
    the surface and volume integrals of a probe whose component is 3D
    (`probe_sdim`, given for probes only).
    """
    unused: set[str] = set()
    for name, value in values.items():
        switched = name[:-len('active')]
        if name.endswith('active') and switched in values and \
                value in ('off', False):
            unused.add(switched)
        source = name[:-len('_src')]
        if name.endswith('_src') and source in values and \
                isinstance(value, str) and value not in ('', 'userdef') \
                and 'userdef' in allowed(name):
            unused.add(source)
        if isinstance(value, str) and value != name and value in values:
            options = allowed(name)
            if len(options) > 1 and value in options and \
                    all(option in values for option in options):
                unused.update(o for o in options if o != value)
    if kind in MESH_SIZES and values.get('custom') == 'off':
        unused.update(name for size in DERIVED_SIZES
                      for name in (size, size + 'active') if name in values)
    if kind in MESH_SIZES and values.get('custom') == 'on' and \
            'hauto' in values:
        unused.add('hauto')
    if kind == 'MassProperties' and 'physics' in values and \
            values.get('densitySource') != 'fromSpecifiedPhysics':
        unused.add('physics')
    if kind in SWEEPS and values.get('save') in ('off', False) and \
            'filename' in values:
        unused.add('filename')
    if values.get('PortExcitation') in ('off', False):
        unused.update(name for name in EXCITATION if name in values)
    if probe_sdim == 3:
        unused.update(name for name in INTEGRALS if name in values)
    return unused


def unit_text(parts) -> str:
    """Joins the parts of a unit from `evaluateUnit`; '1' for none."""
    if parts is None:
        return '1'
    return ''.join(str(part) for part in parts) or '1'


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

    def alike(own, auto) -> bool:
        if isinstance(own, list) and isinstance(auto, list):
            return len(own) == len(auto) and \
                all(alike(a, b) for a, b in zip(own, auto))
        if own == auto:
            return True
        return (isinstance(own, str) and isinstance(auto, str)
                and own in solutions and auto in solutions)

    def same(own, auto) -> bool:
        return alike(own, mapped(auto))

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
        # the space dimension of each component with a geometry
        self.dimensions: dict[str, int] = {}
        for ctag in self.components:
            component = self.java.component(ctag)
            for gtag in _tags(component.geom()):
                self.geometries[gtag] = component.geom(gtag)
                self.dimensions[ctag] = int(component.geom(gtag).getSDim())
        self.places: dict[tuple[str, int, int], dict] = {}
        self.noted_places: set[tuple] = set()
        # what is being read: 'mesh' and 'study' nodes get their own rules
        self.context = ''
        self.measures: dict[str, Any] = {}
        self.vertices: dict[str, list[list[float]]] = {}
        self.blanks: dict[tuple[str | None, str], Any] = {}
        # SI values by expression, and the temporary parameter that
        # evaluates them (a name the user's parameters do not use)
        self.evaluations: dict[str, dict | None] = {}
        names = {str(n) for n in self.java.param().varnames()}
        self.expression_name = next(
            f'mkexpr{i}' for i in range(len(names) + 1)
            if f'mkexpr{i}' not in names)

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
             kept=None, probe_sdim: int | None = None) -> dict:
        """
        Describes a node against its counterpart `base` in the temporary
        model (None if there is none). `subnodes` creates the
        counterparts of its own subnodes; `kept` tells which property
        names count; `probe_sdim` is the space dimension of a probe's
        component.
        """
        values = self.read(java, path)
        if kept is not None:
            values = {name: v for name, v in values.items() if kept(name)}
        kind = _type(java)
        entry: dict = {'tag': str(java.tag()), 'path': path, 'type': kind,
                       'label': _label(java), 'active': _active(java)}
        base_values = None if base is None else \
            self.read(base, path, noted=False)
        # what the raw values tell before they are reduced to differences
        names = names_of(values, kind)
        if names:
            entry['names'] = names
        if has_rows(values):
            entry['rows'] = {name: values.pop(name) for name in ROWS
                             if name in values}
            if base_values is not None:
                entry['row_defaults'] = {
                    name: row[0] for name in ROWS
                    if isinstance(row := base_values.pop(name, None), list)
                    and row}
        unused = unused_of(values,
                           lambda name: _comsol.allowed_values(java, name),
                           kind, probe_sdim)
        for name in unused:
            values.pop(name, None)
            if base_values is not None:
                base_values.pop(name, None)
        properties: dict
        defaults: dict
        unknown: list[str]
        maps = STEP_MAPS & set(values) if self.context == 'study' \
            else set()
        if base_values is None:
            for name in maps:
                values[name] = pairs_of(values[name]) or values[name]
            properties, defaults, unknown = values, {}, []
            entry['all_properties'] = True
            self.note(path, 'defaults_unknown',
                      f'no new {kind} node could be made for defaults; '
                      'all properties are listed')
        else:
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
        if unknown:
            entry['unknown_defaults'] = unknown
            self.note(path, 'defaults_unknown', 'no default for '
                      + ', '.join(repr(name) for name in unknown))
        if unused:
            entry['unused'] = sorted(unused)
        entry['properties'] = properties
        entry['defaults'] = defaults
        self.add_si(entry, java, properties, defaults)
        entry['selection'] = self.selection(java, path)
        entry['selections'] = self.extra_selections(java, path)
        entry['features'] = [] if subnodes is None else \
            self.nodes(java, base, path, subnodes)
        return entry

    def add_si(self, entry: dict, java, properties: dict, defaults: dict,
               names=None, java_of=None):
        """
        Adds the SI values of a node's values (`si`) and defaults
        (`si_defaults`) to its entry, for those COMSOL can evaluate and
        that are no choice among named options; `names` maps a key to the
        property name on the Java node (`java`, or `java_of(key)`).
        """
        for key, values in (('si', properties), ('si_defaults', defaults)):
            found = {}
            for name, value in values.items():
                si = self.si(value)
                owner_java = java_of(name) if java_of else java
                if si is not None and not _comsol.allowed_values(
                        owner_java, names(name) if names else name):
                    found[name] = si
            if found:
                entry[key] = found

    def si(self, value):
        """
        Returns the SI value of a property value: {'value', 'unit'} for a
        string, a list of those (or None) for a list of strings, or None.
        """
        if isinstance(value, str):
            return self.evaluated(value)
        if isinstance(value, list) and value and \
                all(isinstance(item, str) for item in value):
            found = [self.evaluated(item) for item in value]
            return found if any(item is not None for item in found) \
                else None
        return None

    def evaluated(self, expression: str) -> dict | None:
        """
        Returns {'value', 'unit'} of an expression in SI units, as the
        temporary model with the user's parameters evaluates it, or None
        if it cannot (variables, functions, lists, choices).
        """
        if expression in self.evaluations:
            return self.evaluations[expression]
        found: dict | None = None
        text = expression.strip()
        try:
            found = {'value': json_value(float(text)), 'unit': '1'}
        except ValueError:
            if text and self.scratch is not None:
                found = self.evaluate_scratch(expression)
        self.evaluations[expression] = found
        return found

    def evaluate_scratch(self, expression: str) -> dict | None:
        params = self.scratch.param()
        name = self.expression_name
        try:
            params.set(name, expression)
            try:
                value: Any = json_value(float(params.evaluate(name)))
            except Exception:       # e.g. a complex value
                parts = params.evaluateComplex(name)
                value = [json_value(float(parts[0])),
                         json_value(float(parts[1]))]
            return {'value': value,
                    'unit': unit_text(params.evaluateUnit(name))}
        except Exception:
            return None

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
        if self.context == 'mesh' and (not dims or whole):
            return {'level': 'remaining'}
        if whole:
            return {'level': 'geometry'}
        if not dims:
            return {'level': 'none'}
        if len(dims) > 1:
            self.note(path, 'levels_unknown', 'a selection on several '
                      f'levels {dims}: its entities are not described')
            return {'level': 'several',
                    'levels': [_comsol.entity_level_name(d, sdim)
                               for d in dims], 'entities': 'unknown'}
        dim = dims[0]
        applied = [int(e) for e in selection.entities()]
        try:
            named = str(selection.named())
        except Exception:
            named = ''
        chosen: list[int] | None = None
        if named:
            # inputEntities() gives the input of a derived named selection
            # (the domains of an Adjacent), not what it selects
            try:
                chosen = [int(e) for e in
                          self.java.selection(named).entities(dim)]
            except Exception:
                chosen = applied
        else:
            try:
                given = selection.inputEntities()
                chosen = None if given is None else [int(e) for e in given]
            except Exception:
                chosen = None
        if chosen is None:
            chosen = applied
        # domain 0 is the exterior of boundary elements: no place of its own
        exterior = dim == sdim and 0 in chosen
        exterior_applied = dim == sdim and 0 in applied
        if dim == sdim:
            chosen = [n for n in chosen if n != 0]
            applied = [n for n in applied if n != 0]
        found: dict = {'level': _comsol.entity_level_name(dim, sdim),
                       'geometry': gtag,
                       'entities': self.located(gtag, dim, chosen, path)}
        if sorted(set(chosen)) != sorted(set(applied)):
            found['applied'] = self.located(gtag, dim, applied, path)
        if exterior:
            found['exterior'] = True
        if exterior_applied != exterior:
            found['exterior_applied'] = exterior_applied
        if named:
            try:
                found['named'] = _label(self.java.selection(named)) or named
            except Exception:
                found['named'] = named
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
            cached = self.places[key]
            if 'unknown' in cached and (key, path) not in self.noted_places:
                self.noted_places.add((key, path))
                self.note(path, 'place_unknown', f'entity {number} at level '
                          f'{dim} of "{gtag}" could not be measured')
            return cached
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
            self.noted_places.add((key, path))
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
            value: Any = None
            try:
                value = json_value(float(params.evaluate(name)))
            except Exception:
                try:                # a complex value
                    parts = params.evaluateComplex(name)
                    value = [json_value(float(parts[0])),
                             json_value(float(parts[1]))]
                except Exception:
                    pass
            try:
                unit: str | None = unit_text(params.evaluateUnit(name))
            except Exception:
                unit = None
            found[name] = {'expression': str(params.get(name)),
                           'value': value, 'unit': unit}
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
            entry = self.node(
                java, base, path, subnodes=subnodes,
                probe_sdim=self.dimensions.get(component or '')
                if name == 'probe' else None)
            if name == 'probe':
                # a point probe names its expressions in subnodes
                names = entry.get('names', []) + [
                    n for sub in entry['features']
                    for n in sub.get('names', [])]
                if names:
                    entry['names'] = names
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
            self.add_si(described, group, values, {})
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
            'mass_properties': self.mass_properties(java, scomp, ctag),
        }

    def mass_properties(self, component, scomp, ctag: str) -> list[dict]:
        try:
            container = component.massProp()
        except Exception:
            return []
        try:
            base = scomp.massProp() if scomp is not None else None
        except Exception:
            base = None
        return self.nodes_of(container, base, ctag, _create)

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
        unit = str(java.lengthUnit())
        if unit not in LENGTH_SCALES:
            self.note(gtag, 'length_unknown',
                      f'length unit {unit!r} has no known size in metres')
        try:
            finalize: str | None = str(get(java.feature('fin'), 'action'))
        except Exception:
            finalize = None
        try:
            representation: str | None = str(java.geomRep())
        except Exception:
            representation = None
        return {'tag': gtag, 'label': _label(java), 'dimension': sdim,
                'axisymmetric': axisymmetric,
                'length_unit': unit,
                'length_scale': LENGTH_SCALES.get(unit),
                'finalize': finalize,
                'representation': representation,
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
        unknown: list[str] = []
        groups: dict = {}
        for group in java.prop():
            gtag = str(group.tag())
            groups[gtag] = group
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
            changed, default, missing = differs(values, base_values)
            settings.update(changed)
            defaults.update(default)
            unknown += missing
        entry: dict = {'tag': ptag, 'path': path,
                       'identifier': str(java.identifier()),
                       'type': _type(java), 'label': _label(java),
                       'active': _active(java)}
        if base is None:
            entry['all_properties'] = True
            self.note(path, 'defaults_unknown', 'the physics interface '
                      'could not be made for defaults; all properties '
                      'are listed')
        elif unknown:
            entry['unknown_defaults'] = unknown
            self.note(path, 'defaults_unknown', 'no default for '
                      + ', '.join(repr(name) for name in unknown))
        entry['settings'] = settings
        entry['defaults'] = defaults

        def on_group(key):
            return groups[key.split('/', 1)[0]]

        self.add_si(entry, None, settings, defaults,
                    names=lambda key: key.split('/', 1)[1], java_of=on_group)
        entry['selection'] = self.selection(java, path)
        entry['features'] = self.nodes(java, base, path, _create_physics)
        return entry

    def mesh(self, component, scomp, ctag: str, mtag: str) -> dict:
        java = component.mesh(mtag)
        path = f'{ctag}/{mtag}'
        base = None
        if scomp is not None and mtag in _tags(scomp.mesh()):
            base = scomp.mesh(mtag)
        try:
            geometry: str | None = str(java.geom())
        except Exception:
            geometries = _tags(component.geom())
            geometry = geometries[0] if geometries else None
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
                'geometry': geometry,
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
            return {'status': 'not_asked', 'sequence': tag}
        if not _active(study):
            return self.not_compared(stag, tag, 'study disabled')
        reason = self.unready(study)
        if reason is not None:
            return self.not_compared(stag, tag, reason)
        # the model's own nodes first: compiling may update them
        own = self.solver_nodes(sequence, f'{stag}/{tag}')
        sequences = java.sol()
        automatic: dict | None = None
        solutions: set[str] = set()
        with _comsol.history_off(java), \
                _comsol.compiled_traces_removed(java) as changed:
            temporary = str(sequences.uniquetag(SCRATCH))
            try:
                made = sequences.create(temporary)
                made.study(stag)
                made.createAutoSequence(stag)
                automatic = self.solver_nodes(made, '', noted=False)
                # a store-solution node names a solution that goes with
                # the temporary sequence
                solutions = set(_tags(sequences))
            except Exception as error:
                reason = ('COMSOL could not make its own solver sequence: '
                          f'{_comsol.reason(error)}')
            finally:
                _comsol.undo(changed, 'the temporary solver sequence',
                             lambda: temporary in _tags(sequences)
                             and sequences.remove(temporary))
        for message in changed:
            self.note(stag, 'model_changed', message)
        if automatic is None:
            return self.not_compared(stag, tag, str(reason))
        changes = solver_changes(own, automatic, {temporary: str(tag)},
                                 solutions | set(_tags(sequences)))
        return {'status': 'compared', 'sequence': tag, 'changes': changes}

    def not_compared(self, stag: str, tag: str | None, reason: str) -> dict:
        self.note(stag, 'solver_not_compared', reason)
        return {'status': 'not_compared', 'sequence': tag,
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

    def solver_nodes(self, sequence, prefix: str,
                     noted: bool = True) -> dict:
        """
        Returns the active nodes of a solver sequence by tag path;
        properties that cannot be read are noted under `prefix` (the
        study and sequence) unless `noted` is off.
        """
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
                              self.read(child, f'{prefix}/{own}',
                                        SKIPPED | SOLVER_SKIPPED, noted))
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
        # the user's parameters, for SI values of expressions that use them
        params = java.param()
        for name in [str(n) for n in params.varnames()]:
            try:
                scratch.param().set(name, str(params.get(name)))
            except Exception:
                pass
        for ctag in reader.components:
            component = java.component(ctag)
            try:
                scomp = scratch.component().create(ctag, True)
            except Exception as error:
                # its nodes list all their properties
                reader.note(ctag, 'defaults_unknown', 'no new component '
                            'could be made for defaults: '
                            f'{_comsol.reason(error)}')
                continue
            unmade = set()
            for gtag in _tags(component.geom()):
                try:
                    _scratch_geometry(scomp, component.geom(gtag))
                except Exception as error:
                    # meshes of a wrong block would show sizes as changed
                    unmade.add(gtag)
                    reader.note(f'{ctag}/{gtag}', 'defaults_unknown',
                                'no new geometry could be made for '
                                f'defaults: {_comsol.reason(error)}')
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
            for mtag in _tags(component.mesh()):
                try:
                    gtag = str(component.mesh(mtag).geom())
                    if gtag not in unmade:
                        scomp.mesh().create(mtag, gtag)
                except Exception:
                    pass        # its features list all their properties
        for stag in _tags(java.study()):
            try:
                scratch.study().create(stag)
            except Exception as error:
                reader.note(stag, 'defaults_unknown', 'no new study could '
                            f'be made for defaults: {_comsol.reason(error)}')
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
