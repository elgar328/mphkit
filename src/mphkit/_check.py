"""
The check before solving. Public as `mk.check`.

COMSOL solves some mistakes without an error: an expression in the wrong
unit is taken as a number in the expected one, a condition that selects
nothing or applies nowhere does nothing, a model without a mesh gives an
empty solution. `check()` finds those from what COMSOL reports about the
model: the selection a feature was given and the entities it applies on,
the variable tables of features and materials (with their units), the
parameters, and the physics each study step solves. It leaves nothing in
the model, not even in its history; the default features of a physics
interface are found in a separate, temporary model.
"""
from __future__ import annotations

import re
from typing import Any, NamedTuple

from mph.model import Model
from mph.node import Node, escape

from . import _catalog, _comsol, _materials

KINDS = ('unit_mismatch', 'no_material', 'empty_selection', 'not_applied',
         'no_mesh', 'not_solved', 'default_condition')
# Entity numbers written out in a message; `entities` has them all
SHOWN = 10
PLURALS = {'domain': 'domains', 'boundary': 'boundaries', 'edge': 'edges',
           'point': 'points'}
# Names in an expression, after the units in brackets are removed
NAME = re.compile(r'(?<![\w.])[A-Za-z_][\w.]*')
UNITS = re.compile(r'\[[^\]]*\]')
# A number with one unit, e.g. 5[m] or 1e3[W/m^2]
SIMPLE = re.compile(r'[-+]?[\d.]+(?:[eE][-+]?\d+)?\[[^\]]*\]')
MATERIAL = re.compile(r'\bmaterial\.([A-Za-z_]\w*)')
# Default features found in a temporary model: (type, sdim, axisymmetric)
# to the (tag, type) pairs of the features a new interface has
_DEFAULTS: dict[tuple[str, int, bool], frozenset[tuple[str, str]]] = {}


class Feature(NamedTuple):
    """A top-level feature of a physics interface, with one level."""
    tag: str
    java: Any
    level: int
    selected: list[int]
    applied: list[int]
    path: str | None


class Physics(NamedTuple):
    """An active physics interface on a geometry."""
    java: Any
    component: Any
    geometry: Any
    sdim: int
    dim: int


def check(model: Model, /) -> list[dict]:
    """
    Lists what COMSOL would get wrong silently or only vaguely when
    solving the model:

    ```python
    problems = [p for p in mk.check(model) if p['severity'] == 'warning']
    # [{'kind': 'unit_mismatch', 'severity': 'warning',
    #   'node': 'physics/heat/hot', 'level': None, 'entities': None,
    #   'message': '"hot": T0 = 5[m] is in m, not K. ...'}]
    ```

    Call it once the physics, materials, mesh and study exist, right
    before `model.solve()`, and fix the items with severity `'warning'`;
    `'info'` items are things to know (a model without mistakes still has
    some, such as the boundaries left at the default condition). Each item
    has `kind`, `severity`, a `message` that says what to do next, `node`
    (the path of the node concerned, e.g. `'physics/heat/hot'`, opened
    with `model/node`, or `None`), `level` ('domain', 'boundary', 'edge',
    'point' or `None`) and `entities` (entity numbers or `None`). Warnings
    come first. The kinds:

    - `'unit_mismatch'`: an expression a feature or material property was
      given has another unit than COMSOL expects, e.g. `T0 = 5[m]` for a
      temperature; COMSOL solves with the number in the expected unit.
      Only expressions of numbers, constants and parameters are checked;
      array elements are counted from 0, as in Python. Material values
      are matched by their text, so one that COMSOL rewrote (`1e3` as
      `1000`) may be skipped.
    - `'no_material'`: a feature takes properties from a material (e.g.
      k, rho, Cp of a solid) on domains that have none. The solve then
      fails with "Undefined material property", unless the study does not
      use them (e.g. Cp in a stationary study). Domains with a
      multiphysics coupling are skipped, since couplings may supply
      properties; so a missing material under, e.g., thermal expansion is
      not reported (the solve still fails clearly).
    - `'empty_selection'`: a feature selects nothing.
    - `'not_applied'`: a feature does not apply on some of what it
      selects: a heat flux on an interior boundary, a condition outside
      the domains of its physics or on the symmetry axis, or a later
      feature that applies there instead. A warning when it applies
      nowhere and no later feature takes its place, else info.
    - `'no_mesh'`: a component with physics has no mesh; the solve would
      give an empty solution.
    - `'not_solved'`: no study step solves a physics interface, or there
      is no study.
    - `'default_condition'` (info): boundaries left at the default
      condition of a physics interface, e.g. thermal insulation.

    It checks the top-level features of active physics interfaces on a
    geometry, which must be built (`model.build(geom)`, also after loading
    a file), and the materials in use (component materials and global
    materials a link points to, not those inside a material switch). Not
    checked: subfeatures, pair features, global features (without a
    selection), physics settings, and what the solve itself reports
    clearly (undefined parameters or variables, syntax errors). If the
    temporary model cannot create a physics interface, its selections and
    default conditions are not checked. After solving, MPh's `model.problems()` lists COMSOL's
    own messages. Default features are recognised in a temporary model of
    the same kind; in models from older COMSOL versions they may show up
    as user features. Returns plain values (`json.dumps` works) and
    leaves nothing in the model.
    """
    if not isinstance(model, Model):
        raise TypeError(f'mk.check takes a model, not {model!r}.')
    interfaces = active_physics(model)
    found: list[dict] = []
    components: set[str] = set()
    materials: set[str] = set()
    for physics in interfaces:
        defaults = _defaults(str(physics.java.getType()), physics.sdim,
                             _axisymmetric(physics.geometry))
        features = _features(model, physics.java)
        for feature in features:
            _units(model, feature.java, feature.path, f'.{feature.tag}.',
                   found)
        if physics.dim == physics.sdim:
            _no_material(physics, features, found)
        if defaults is not None:
            _selections(model, physics, features, defaults, found)
        ctag = str(physics.component.tag())
        if ctag not in components:
            components.add(ctag)
            _material_units(model, physics.component, materials, found)
    _meshes(model, interfaces, found)
    _studies(model, interfaces, found)
    order = {kind: n for n, kind in enumerate(KINDS)}
    return sorted(found, key=lambda item: (item['severity'] != 'warning',
                                           order[item['kind']]))


###########
# Parsing #
###########

def active_physics(model: Model) -> list[Physics]:
    """
    Returns the active physics interfaces on a geometry, after checking
    that their geometries are built.
    """
    found = []
    checked: set[tuple[str, str]] = set()
    components = model.java.component()
    for ctag in components.tags():
        component = components.get(ctag)
        for ptag in component.physics().tags():
            java = component.physics(ptag)
            try:
                if not java.isActive():
                    continue
                dims = _comsol.selection_dims(java.selection())
            except Exception:
                continue
            geometry = _catalog.geometry_of_physics(model, java)
            if geometry is None or not dims:
                continue
            key = (str(ctag), str(geometry.tag()))
            if key not in checked:
                checked.add(key)
                _comsol.check_geometry_built(
                    geometry, 'geometries/' + _comsol.name_of(geometry))
            found.append(Physics(java, component, geometry,
                                 int(geometry.getSDim()), dims[0]))
    return found


def _axisymmetric(geometry) -> bool:
    try:
        return bool(geometry.isAxisymmetric())
    except Exception:
        return False


def _features(model: Model, physics) -> list[Feature]:
    """
    Returns the active top-level features of a physics interface that
    have a selection at one level, but no pair features.
    """
    found = []
    container = physics.feature()
    for tag in container.tags():
        feature = container.get(tag)
        try:
            if not feature.isActive() or feature.hasProperty('pairs'):
                continue
            selection = feature.selection()
            dims = _comsol.selection_dims(selection)
            selected = [int(e) for e in selection.inputEntities()]
            applied = [int(e) for e in selection.entities()]
        except Exception:
            continue
        if len(dims) == 1:
            path = _path(model, [('physics', None),
                                 (str(physics.label()), physics),
                                 (str(feature.label()), feature)])
            found.append(Feature(str(tag), feature, dims[0], selected,
                                 applied, path))
    return found


def _table(java) -> list[list[str]]:
    """Returns the rows of a node's own variable table, as strings."""
    try:
        rows = java.featureInfo('info').getInfoTable('Expression')
    except Exception:
        return []
    return [['' if cell is None else str(cell) for cell in row]
            for row in rows]


def _values(java) -> dict[str, list[tuple[str, str]]]:
    """
    Returns the text values of a node's properties: property name to
    (element, value) pairs, where element is '' for a single value, '[i]'
    for an array element and '[i][j]' for a matrix element.
    """
    found: dict[str, list[tuple[str, str]]] = {}
    try:
        names = [str(p) for p in java.properties()]
    except Exception:
        return found
    for name in names:
        try:
            kind = str(java.getValueType(name))
            if kind == 'String':
                found[name] = [('', str(java.getString(name)))]
            elif kind == 'StringArray':
                found[name] = [(f'[{i}]', str(v)) for i, v in
                               enumerate(java.getStringArray(name))]
            elif kind == 'StringMatrix':
                found[name] = [(f'[{i}][{j}]', str(v))
                               for i, row in
                               enumerate(java.getStringMatrix(name))
                               for j, v in enumerate(row)]
        except Exception:
            continue
    return found


#########
# Nodes #
#########

def _path(model: Model, parts: list[tuple[str, Any]]) -> str | None:
    """
    Returns the MPh path of a Java node from the labels along it, given
    as (label, Java node) pairs (`None` for a group), or `None` if the
    path leads elsewhere: MPh opens the first node of a repeated label.
    """
    names: list[str] = []
    try:
        for label, java in parts:
            names.append(escape(label))
            if java is not None and \
                    Node(model, '/'.join(names)).tag() != str(java.tag()):
                return None
    except Exception:
        return None
    return '/'.join(names)


def _entities(level: str | None, numbers: list[int]) -> str:
    """Writes entity numbers, e.g. 'boundary 6' or 'boundaries 2, 3'."""
    shown = ', '.join(str(n) for n in numbers[:SHOWN])
    if len(numbers) > SHOWN:
        shown += ', …'
    word = level or 'entity'
    if len(numbers) != 1:
        word = PLURALS.get(word, 'entities')
    return f'{word} {shown}'


def _item(found: list[dict], kind: str, severity: str, message: str,
          node: str | None = None, level: str | None = None,
          entities: list[int] | None = None):
    found.append({'kind': kind, 'severity': severity, 'message': message,
                  'node': node, 'level': level, 'entities': entities})


#########
# Units #
#########

def _units(model: Model, java, node: str | None, row_suffix: str | None,
           found: list[dict], what: str | None = None):
    """
    Reports the property values of a node whose unit differs from the
    one its variable table expects. `row_suffix` (e.g. '.temp1.') finds
    the table row named after a property; otherwise rows are matched by
    their expression.
    """
    rows = _table(java)
    if not rows:
        return
    by_name = {row[1]: row[3] for row in rows}
    by_expression: dict[str, set[str]] = {}
    for row in rows:
        by_expression.setdefault(row[2], set()).add(row[3])
    parameters = model.java.param()
    what = what or f'"{java.label()}"'
    for prop, values in _values(java).items():
        named = None
        if row_suffix:
            named = next((unit for name, unit in by_name.items()
                          if name.endswith(row_suffix + prop)), None)
        for element, value in values:
            unit = named
            if unit is None:
                units = by_expression.get(value, set())
                unit = next(iter(units)) if len(units) == 1 else None
            if not unit or unit == '1' or not value.strip():
                continue
            wrong = _wrong_unit(parameters, value, unit)
            if wrong:
                example = ''
                if SIMPLE.fullmatch(value.strip()):
                    example = f' (e.g. {UNITS.sub(f"[{unit}]", value)})'
                _item(found, 'unit_mismatch', 'warning',
                      f'{what}: {prop}{element} = {value} is in {wrong}, '
                      f'not {unit}. COMSOL solves with the number as '
                      f'{unit}; write it in {unit}{example}, or as a plain '
                      f'number in {unit}.', node)


def _wrong_unit(parameters, expression: str, unit: str) -> str | None:
    """
    Returns the unit of an expression of numbers, constants and
    parameters if it does not fit `unit`, else `None`, also for
    expressions of other variables, which are not checked.
    """
    text = UNITS.sub('', expression)
    for match in NAME.finditer(text):
        if text[match.end():].lstrip().startswith('('):
            continue                    # a function
        try:
            parameters.evaluate(match.group())
        except Exception:
            return None                 # not global, e.g. T or ht.x
    try:
        parameters.evaluate(expression)
    except Exception:
        return None
    try:
        parameters.evaluate(expression, unit)
        return None
    except Exception:
        pass
    try:
        parts = parameters.evaluateUnit(expression)
    except Exception:
        return None
    if parts is None:
        return None                     # a plain number: taken in `unit`
    found = ''.join(str(part) for part in parts)
    return None if found in ('', '1') else found


def _material_units(model: Model, component, done: set[str],
                    found: list[dict]):
    """
    Checks the units of the materials a component uses, except those in
    `done` (tags), which it extends.
    """
    members = component.material()
    used = []
    for tag in members.tags():
        member = members.get(tag)
        try:
            if not member.isActive():
                continue
            if str(member.getType()) == 'Link':
                used.append(model.java.material(member.getString('link')))
            else:
                used.append(member)
        except Exception:
            continue
    for material in used:
        if str(material.tag()) in done:
            continue
        done.add(str(material.tag()))
        try:
            groups = material.propertyGroup()
            tags = list(groups.tags())
        except Exception:
            continue                    # e.g. a material switch
        for tag in tags:
            group = groups.get(tag)
            node = _path(model, [('materials', None),
                                 (str(material.label()), material),
                                 (str(group.label()), group)])
            _units(model, group, node, None, found,
                   f'"{material.label()}" ({group.label()})')


#############
# Materials #
#############

def _no_material(physics: Physics, features: list[Feature], found):
    """Reports domain features that take properties from no material."""
    sdim = physics.sdim
    held = {n for _, numbers in _materials.held_domains(physics.component,
                                                        sdim)
            for n in numbers}
    coupled: set[int] = set()
    try:
        container = physics.component.multiphysics()
        couplings = [container.get(tag) for tag in container.tags()]
    except Exception:
        couplings = []
    for coupling in couplings:
        try:
            if not coupling.isActive():
                continue
            selection = coupling.selection()
            if _comsol.selection_dims(selection) == [sdim]:
                coupled.update(int(e) for e in selection.entities())
        except Exception:
            continue
    for entry in features:
        feature, applied = entry.java, entry.applied
        if entry.level != sdim:
            continue
        values = _values(feature)
        taken = sorted({base for row in _table(feature)
                        for name in MATERIAL.findall(row[2])
                        for base in [re.sub(r'\d+$', '', name)]
                        if values.get(f'{base}_mat') == [('', 'from_mat')]},
                       key=str.lower)
        missing = [n for n in applied if n not in held and n not in coupled]
        if taken and missing:
            _item(found, 'no_material', 'warning',
                  f'"{feature.label()}" takes {", ".join(taken)} from a '
                  f'material, but {_entities("domain", missing)} '
                  f'{"has" if len(missing) == 1 else "have"} none. Find one '
                  'with mk.materials(search=...) and add it with '
                  "mk.material(geom, 'Name', selection), where selection "
                  'is a domain selection such as mk.sel.box(...) or the '
                  f'numbers {missing}. A property the study does not use '
                  '(e.g. Cp in a stationary study) is no problem.',
                  entry.path, 'domain', missing)


##############
# Selections #
##############

def _defaults(type: str, sdim: int, axisymmetric: bool
              ) -> frozenset[tuple[str, str]] | None:
    """
    Returns the (tag, type) pairs of the features a new physics interface
    of `type` has, found in a temporary model, or `None` if it cannot be
    created there. Kept for the session once found.
    """
    key = (type, sdim, axisymmetric)
    if key in _DEFAULTS:
        return _DEFAULTS[key]
    import jpype  # type: ignore[import-untyped]
    util = jpype.JClass('com.comsol.model.util.ModelUtil')
    tag = str(util.uniquetag('mkcheck'))
    try:
        scratch = util.create(tag)
        component = scratch.component().create('comp1', True)
        geometry = component.geom().create('geom1', sdim)
        if axisymmetric:
            geometry.axisymmetric(True)
        physics = component.physics().create('phys1', type, 'geom1')
        features = physics.feature()
        found = frozenset((str(t), str(features.get(t).getType()))
                          for t in features.tags())
    except Exception:
        return None
    finally:
        if tag in [str(t) for t in util.tags()]:
            util.remove(tag)
    _DEFAULTS[key] = found
    return found


def _selection_hint(model: Model, feature, level: str | None,
                    numbers: list[int]) -> str:
    """Says how to look at the selection of a feature."""
    try:
        tag = str(feature.selection().named())
    except Exception:
        tag = ''
    selections = model.java.selection()
    if tag and tag in [str(t) for t in selections.tags()]:
        label = str(selections.get(tag).label())
        if sum(str(selections.get(t).label()) == label
               for t in selections.tags()) == 1:
            path = 'selections/' + escape(label)
            return ('Check its selection with mk.sel.entities(geom, '
                    f'model/{path!r}).')
        return f'Check its selection "{label}".'
    if tag:
        return f'Check its selection (tag {tag!r}).'
    if not numbers:
        return 'Give it a selection with .select(...).'
    return f'It selects {_entities(level, numbers)} by number.'


def _selections(model: Model, physics: Physics, features, defaults, found):
    """Reports empty selections, features that do not apply, defaults."""
    sdim = physics.sdim
    for i, entry in enumerate(features):
        tag, feature, level = entry.tag, entry.java, entry.level
        selected, applied, node = entry.selected, entry.applied, entry.path
        name = _comsol.entity_level_name(level, sdim)
        label = str(feature.label())
        if (tag, str(feature.getType())) in defaults:
            if level == physics.dim - 1 and applied and \
                    not str(feature.getType()).startswith('AxialSymmetry'):
                text = _entities(name, applied)
                _item(found, 'default_condition', 'info',
                      f'{text[0].upper()}{text[1:]} keep'
                      f'{"s" if len(applied) == 1 else ""} the default '
                      f'"{label}".', node, name, applied)
            continue
        if not selected:
            _item(found, 'empty_selection', 'warning',
                  f'"{label}" selects nothing, so it has no effect. '
                  f'{_selection_hint(model, feature, name, [])}',
                  node, name, [])
            continue
        kept = set(applied)
        lost = [n for n in selected if n not in kept]
        if not lost or applied and \
                len(selected) == _comsol.count_entities(physics.geometry, level):
            continue
        later = [str(other.java.label()) for other in features[i + 1:]
                 if other.level == level and set(lost) & set(other.applied)]
        text = _entities(name, lost)
        if later:
            names = ', '.join(f'"{other}"' for other in later)
            _item(found, 'not_applied', 'info',
                  f'"{label}" does not apply on {text}; the later {names} '
                  f'appl{"y" if len(later) > 1 else "ies"} there. If '
                  'intended, nothing to do.', node, name, lost)
        elif applied:
            _item(found, 'not_applied', 'info',
                  f'"{label}" does not apply on {text} of its selection '
                  '(outside the physics or not applicable there).',
                  node, name, lost)
        else:
            _item(found, 'not_applied', 'warning',
                  f'"{label}" applies nowhere: COMSOL drops it on {text} '
                  '(e.g. a heat flux on an interior boundary, a condition '
                  "outside the physics' domains or on the symmetry axis). "
                  f'{_selection_hint(model, feature, name, selected)}',
                  node, name, lost)


###################
# Meshes, studies #
###################

def _meshes(model: Model, interfaces: list[Physics], found: list[dict]):
    """Reports components with physics but no mesh."""
    done: set[str] = set()
    for physics in interfaces:
        component = physics.component
        ctag = str(component.tag())
        if ctag in done:
            continue
        done.add(ctag)
        if not list(component.mesh().tags()):
            path = _path(model, [('geometries', None),
                                 (str(physics.geometry.label()),
                                  physics.geometry)])
            target = f'model/{path!r}' if path else 'geom'
            _item(found, 'no_mesh', 'warning',
                  f'Component "{component.label()}" has physics but no '
                  'mesh; model.solve() would give an empty solution. '
                  f"Create one: (model/'meshes').create({target}).")


def _studies(model: Model, interfaces: list[Physics], found: list[dict]):
    """Reports physics interfaces that no study step solves."""
    if not interfaces:
        return
    steps = []
    active = False
    studies = model.java.study()
    for tag in studies.tags():
        study = studies.get(tag)
        try:
            if not study.isActive():
                continue
        except Exception:
            continue
        active = True
        for step_tag in study.feature().tags():
            step = study.feature(step_tag)
            try:
                if not step.isActive():
                    continue
                pairs = [str(v) for v in step.getStringArray('activate')]
            except Exception:
                continue            # e.g. a parametric sweep
            steps.append(dict(zip(pairs[::2], pairs[1::2])))
    if not active:
        _item(found, 'not_solved', 'warning',
              "The model has no study; add one, e.g. (model/'studies')"
              ".create().create('Stationary').")
        return
    for physics in interfaces:
        ptag = str(physics.java.tag())
        ctag = str(physics.component.tag())
        if not any(step.get(ptag) == 'on' and step.get(ctag) != 'off'
                   for step in steps):
            label = str(physics.java.label())
            _item(found, 'not_solved', 'warning',
                  f'"{label}" is not solved by any study step; enable it '
                  'in a step (Physics and Variables Selection).',
                  _path(model, [('physics', None), (label, physics.java)]))
