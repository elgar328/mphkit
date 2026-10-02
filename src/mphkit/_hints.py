"""
Suggestions for names that do not exist in `mphkit` or `mphkit.sel`.

People and AI assistants guess helper names: `mk.box` for `mk.sel.box`,
`mk.select_box`, `mk.sel_box`, `mk.sel.block`. Both modules answer an
unknown name with the helper that was probably meant and a pointer to
the overview, `mphkit.__doc__`, which lists every helper.

The hook is a module subclass rather than a module-level `__getattr__`:
type checkers treat a module with `__getattr__` as having every attribute,
so editors would stop flagging misspelled names, and `help()` would list
the hook among the helpers. `from mphkit import box` shows no hint: Python
turns the error into its own `ImportError`.
"""
from __future__ import annotations

import sys
from difflib import SequenceMatcher
from types import ModuleType

from ._comsol import entity_suggestion

MAIN = 'mphkit'
SEL = 'mphkit.sel'

# Prefixes of guessed names and what they ask for, as in `select_box`.
SELECT_PREFIXES = ('select_', 'find_', 'get_')
CREATE_PREFIXES = ('create_', 'make_', 'add_', 'new_')

# Shape words: the selection helper and the helper creating that shape.
SHAPES = {
    'box': ('box', 'block'), 'block': ('box', 'block'),
    'cube': ('box', 'block'), 'rectangle': ('box', 'rectangle'),
    'square': ('box', 'square'), 'ball': ('ball', 'sphere'),
    'sphere': ('ball', 'sphere'), 'disk': ('disk', 'circle'),
    'circle': ('disk', 'circle'),
}

# Words for what a helper does rather than its name.
MEANINGS: dict[str, str | tuple[str, ...]] = {
    'volume': 'mphkit.measure', 'area': 'mphkit.measure',
    'length': 'mphkit.measure', 'bbox': 'mphkit.bounding_box',
    'bounds': 'mphkit.bounding_box', 'select': 'mphkit.sel',
    'selection': 'mphkit.sel', 'selections': 'mphkit.sel',
    'subtract': 'mphkit.difference', 'cut': 'mphkit.difference',
    'minus': 'mphkit.difference', 'merge': 'mphkit.union',
    'fuse': 'mphkit.union', 'combine': 'mphkit.union',
    'unite': 'mphkit.union',
    'translate': 'mphkit.move', 'shift': 'mphkit.move',
    'rotation': 'mphkit.rotate', 'extrusion': 'mphkit.extrude',
    'line': 'mphkit.line_segment', 'polyline': 'mphkit.polygon',
    'info': 'mphkit.summary', 'geominfo': 'mphkit.summary',
    'geometry_info': 'mphkit.summary', 'geom_info': 'mphkit.summary',
    'describe': 'mphkit.summary', 'stats': 'mphkit.summary',
    'picture': 'mphkit.image',
    'screenshot': 'mphkit.image', 'snapshot': 'mphkit.image',
    'render': 'mphkit.image', 'show': 'mphkit.image', 'draw': 'mphkit.image',
    'save_image': 'mphkit.image', 'export_image': 'mphkit.image',
    'view': 'mphkit.image', 'display': 'mphkit.image',
    'statistics': 'mphkit.summary',
    'coords': 'mphkit.coordinates',
    'vertex_coordinates': 'mphkit.coordinates',
    'neighbours': 'mphkit.sel.neighbors', 'adjacency': 'mphkit.sel.neighbors',
    'adj': 'mphkit.sel.neighbors',
    # LiveLink for MATLAB
    'mphgeominfo': 'mphkit.summary', 'mphgetadj': 'mphkit.sel.neighbors',
    'mphgetcoords': 'mphkit.coordinates', 'mphgeom': 'mphkit.image',
    'mphviewselection': 'mphkit.image', 'mphmeasure': 'mphkit.measure',
    'mphselectbox': 'mphkit.sel.box', 'mphselectcoords': 'mphkit.sel.ball',
    'mphint2': 'mphkit.integral', 'mphmean': 'mphkit.average',
    'mphmax': 'mphkit.maximum', 'mphmin': 'mphkit.minimum',
    'mphinterp': 'mphkit.value', 'mphplot': 'mphkit.plot',
    'mphmesh': 'mphkit.image',
    # Results
    'integrate': 'mphkit.integral', 'mean': 'mphkit.average',
    'avg': 'mphkit.average', 'max': 'mphkit.maximum',
    'min': 'mphkit.minimum', 'interp': 'mphkit.value',
    'interpolate': 'mphkit.value', 'probe': 'mphkit.value',
    'evaluate': 'mphkit.value', 'eval': 'mphkit.value',
    'slice': 'mphkit.plot', 'slices': 'mphkit.plot',
    # COMSOL's names; "features" may also mean a geometry feature
    'features': ('mphkit.feature_types', 'mphkit.feature'),
    'feature_type': 'mphkit.feature_types',
    'list_features': 'mphkit.feature_types',
    'feature_list': 'mphkit.feature_types',
    'physics_features': 'mphkit.feature_types',
    'physics_feature': 'mphkit.feature_types',
    'mesh_types': 'mphkit.feature_types',
    'study_types': 'mphkit.feature_types',
    'mesh_features': 'mphkit.feature_types',
    'study_steps': 'mphkit.feature_types',
    'types': 'mphkit.feature_types', 'list_types': 'mphkit.feature_types',
    'geometry_features': ('mphkit.feature_types', 'mphkit.feature'),
    'geometry_types': ('mphkit.feature_types', 'mphkit.feature'),
    'geom_types': ('mphkit.feature_types', 'mphkit.feature'),
    'feature_properties': 'mphkit.properties',
    'property_values': 'mphkit.properties',
    'list_properties': 'mphkit.properties', 'props': 'mphkit.properties',
    'settings': ('mphkit.properties', 'mphkit.set'),
    'options': ('mphkit.properties', 'mphkit.set'),
    'list_variables': 'mphkit.variables', 'vars': 'mphkit.variables',
    'variable': 'mphkit.variables', 'expressions': 'mphkit.variables',
    'physics': 'mphkit.physics_types', 'physics_type': 'mphkit.physics_types',
    'interfaces': 'mphkit.physics_types',
    'list_physics': 'mphkit.physics_types',
    'physics_list': 'mphkit.physics_types',
    'physics_interfaces': 'mphkit.physics_types',
    # The check before solving
    'validate': 'mphkit.check', 'diagnose': 'mphkit.check',
    'precheck': 'mphkit.check', 'check_model': 'mphkit.check',
    'lint': 'mphkit.check', 'verify': 'mphkit.check',
    'sanity_check': 'mphkit.check',
    # Mesh quality
    'mesh_stats': 'mphkit.mesh_quality', 'meshstats': 'mphkit.mesh_quality',
    'mesh_statistics': 'mphkit.mesh_quality',
    'element_quality': 'mphkit.mesh_quality',
    'mesh_info': 'mphkit.mesh_quality', 'mphmeshstats': 'mphkit.mesh_quality',
    'meshstat': 'mphkit.mesh_quality', 'min_quality': 'mphkit.mesh_quality',
    'quality_histogram': 'mphkit.mesh_quality',
    'element_count': 'mphkit.mesh_quality',
    'num_elements': 'mphkit.mesh_quality',
    'mesh_elements': 'mphkit.mesh_quality',
    'worst_elements': 'mphkit.mesh_quality',
    'skewness': 'mphkit.mesh_quality',
    # Long solves
    'estimate': 'mphkit.problem_size', 'dofs': 'mphkit.problem_size',
    'solve_size': 'mphkit.problem_size',
    'show_progress': 'mphkit.log_progress',
    'showprogress': 'mphkit.log_progress',
    'progress_log': 'mphkit.log_progress',
    'solver_log': 'mphkit.log_progress',
    'progress_bar': 'mphkit.progress', 'solve_status': 'mphkit.progress',
    'memory_usage': 'mphkit.progress', 'stop_solve': 'mphkit.progress',
    'cancel_solve': 'mphkit.progress', 'kill_solve': 'mphkit.progress',
    'run_background': 'mphkit.progress', 'monitor': 'mphkit.progress',
    # Values of sweeps and steps
    'outer_solutions': 'mphkit.outer_values',
    'outer_parameters': 'mphkit.outer_values',
    'sweep_values': 'mphkit.outer_values', 'mphsolinfo': 'mphkit.outer_values',
    'parameter_values': ('mphkit.outer_values', 'mphkit.step_values'),
    'param_values': ('mphkit.outer_values', 'mphkit.step_values'),
    'time_values': 'mphkit.step_values', 'time_steps': 'mphkit.step_values',
    'timesteps': 'mphkit.step_values', 'frequencies': 'mphkit.step_values',
    'eigenvalues': 'mphkit.step_values', 'eigen_values': 'mphkit.step_values',
    'eigenfrequencies': 'mphkit.step_values',
    'frequency_values': 'mphkit.step_values',
    'freq_values': 'mphkit.step_values', 'sweep_steps': 'mphkit.step_values',
    'outer_value': 'mphkit.outer_values', 'step_value': 'mphkit.step_values',
    'steps': 'mphkit.step_values', 'step': 'mphkit.step_values',
    'times': 'mphkit.step_values', 'get_steps': 'mphkit.step_values',
    'list_steps': 'mphkit.step_values', 'get_times': 'mphkit.step_values',
    'maxval': 'mphkit.maximum', 'minval': 'mphkit.minimum',
    # Materials from COMSOL's libraries
    'matlib': ('mphkit.materials', 'mphkit.material'),
    'material_library': ('mphkit.materials', 'mphkit.material'),
    'library_materials': 'mphkit.materials',
    'list_materials': 'mphkit.materials',
    'insert_material': 'mphkit.material',
    'add_material': 'mphkit.material',
}

# Words that, next to `mesh`, ask for a picture of it, as in `plot_mesh`.
PICTURES = ('plot', 'image', 'picture', 'png', 'show', 'view')
# Words that, next to `mesh`, ask for its quality in numbers
QUALITY = ('quality', 'skewness', 'skew', 'stat', 'stats', 'statistics')

# Last words of guessed names that ask for a result whatever comes first,
# as in `volume_integral`, `point_value` or `surface_plot`, unless the
# first word asks for one too (`max_value`). `max` and `min` count only
# first: `bbox_max` asks for a bounding box.
RESULTS = ('integral', 'integrate', 'average', 'mean', 'avg', 'value',
           'values', 'plot')
RESULT_FIRST = ('integral', 'integrate', 'average', 'mean', 'avg',
                'maximum', 'max', 'minimum', 'min')

# MPh evaluates global expressions and values at all mesh nodes.
EVALUATE_NOTE = ("MPh's model.evaluate(expression, unit) gives global values "
                 'and values at all mesh nodes; mphkit.integral, '
                 'mphkit.average, mphkit.maximum, mphkit.minimum and '
                 'mphkit.value evaluate over entities or at points.')

# Notes added when a helper is the only suggestion.
NOTES = {
    'mphkit.union': 'Pass intbnd=False to merge touching or overlapping '
                    'objects into one domain.',
    'mphkit.image': 'Pictures of the geometry, selections and the mesh '
                    '(mesh=True); mphkit.plot draws results.',
    'mphkit.sel.neighbors': 'mphkit.sel.adjacent makes a selection instead.',
}

# Notes for guessed names of mphkit itself that ask for plain MPh or an
# argument, and the helpers to suggest with them
SWEEP_NOTE = ("A parametric sweep is plain MPh: sweep = study.create("
              "'Parametric'); mk.set(sweep, pname=['Th'], plistarr=['100 "
              "200'], punit=['degC']). Read it with outer= or step= of the "
              'results helpers; mphkit.outer_values tells which.')
PARAMETER_NOTE = ("Parameters are plain MPh: model.parameter('L', "
                  "'0.1[m]').")
DATASET_NOTE = ('The results helpers take dataset= (a dataset or its '
                "study); MPh's model.datasets() lists them.")
SOLVE_NOTE = ("Studies, meshes and solving are plain MPh: (model/'studies')"
              ".create(name='s'), study.create('Stationary'), "
              "(model/'meshes').create(geom), model.solve('s'); "
              'mphkit.feature_types(study) lists the step types.')
LOAD_NOTE = ("An existing model: old = client.load('file.mph') in MPh; "
             'mphkit.import_ imports CAD geometry. help(mphkit.sel.find) '
             'moves its numbered selections and tells how to inspect it.')
JAVA_NOTE = ("node.java is the COMSOL Java object of an MPh node (mk.set "
             "takes it too); model.save('model.java') writes a model as "
             'Java.')
GUESSED: dict[str, tuple[tuple[str, ...], str]] = {
    **{name: ((), SWEEP_NOTE) for name in (
        'parametric_sweep', 'param_sweep', 'parameter_sweep', 'parametric',
        'outer_sweep', 'sweep_parameter', 'sweep_parameters',
        'parametric_sweeps')},
    **{name: (("mphkit.feature(geom, 'Sweep', ...) (a geometry sweep)",),
              SWEEP_NOTE) for name in (
        'sweep', 'sweeps', 'get_sweep', 'sweep_results', 'sweep_table')},
    **{name: ((), PARAMETER_NOTE) for name in (
        'parameter', 'param', 'parameters', 'params', 'set_parameter',
        'set_param', 'set_parameters', 'add_parameter',
        'define_parameter')},
    **{name: ((), DATASET_NOTE) for name in (
        'datasets', 'dataset', 'list_datasets', 'get_dataset',
        'solutions', 'solution')},
    **{name: ((), SOLVE_NOTE) for name in (
        'solve', 'study', 'studies', 'create_study', 'mesh', 'create_mesh',
        'meshes')},
    'outer': ((), 'outer= is an argument of the results helpers and '
                  'mphkit.plot; mphkit.outer_values lists its values.'),
    **{name: ((), 'The results helpers are mphkit.integral, '
                  'mphkit.average, mphkit.maximum, mphkit.minimum, '
                  'mphkit.value and mphkit.plot; mphkit.sel.result selects '
                  'what a geometry feature left.')
       for name in ('result', 'results')},
    'flux': (('mphkit.integral',), "e.g. mk.integral(geom, 'boundary', "
             "'ht.ntflux', faces, unit='W'); mphkit.variables(physics, "
             "search='flux') finds the names."),
    **{name: ((), 'Animations stay plain MPh; mphkit.plot draws one step '
                  "per call, e.g. step={'t': t} for each t of "
                  "mphkit.step_values(geom)['t'].")
       for name in ('animate', 'animation')},
    **{name: ((), LOAD_NOTE) for name in (
        'load_model', 'load_mph', 'open_model', 'import_model',
        'import_mph')},
    **{name: ((), JAVA_NOTE) for name in ('to_java', 'export_java')},
}

# Notes for whole names only: with a prefix they mean other things, e.g.
# `add_load` a structural load
EXACT = {'load': LOAD_NOTE, 'java': JAVA_NOTE}

# How selection helpers are called; `{kind}` stands for the entity kind.
SEL_CALL = '(geom, {kind}, ...)'
SEL_CALLS = {
    'all': '(geom, {kind})', 'adjacent': '(geom, input, {kind})',
    'result': '(geom, feature, {kind})',
    'cumulative': '(geom, group, {kind})',
    'entities': '(geom, selection)', 'layer': '(geom, feature, layer)',
}

# Geometry features without a named helper, created with `feature()`.
FEATURES = {
    'cone': 'Cone', 'torus': 'Torus', 'sweep': 'Sweep', 'loft': 'Loft',
    'scale': 'Scale', 'copy': 'Copy', 'ellipse': 'Ellipse',
    'ellipsoid': 'Ellipsoid', 'helix': 'Helix',
}


class HintModule(ModuleType):
    """Module whose unknown attributes raise errors naming the right one."""

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(
                f'module {self.__name__!r} has no attribute {name!r}')
        raise missing_attribute(self.__name__, name)


def missing_attribute(module: str, name: str) -> AttributeError:
    """
    Returns the error for the unknown attribute `name` of `module`.

    The error carries `name` but no `obj`, so that Python does not append
    its own, narrower "Did you mean" to the message.
    """
    try:
        return AttributeError(_message(module, name), name=name, obj=None)
    except Exception:
        return AttributeError(f'module {module!r} has no attribute {name!r}')


# The end of every message: the overview lists all helpers (its Index)
POINTER = {MAIN: 'print(mphkit.__doc__) lists all helpers.',
           SEL: "print(mphkit.__doc__) lists all helpers, mphkit.sel's too."}


def _message(module: str, name: str) -> str:
    """
    Builds the error message with suggestions and a pointer to the
    overview.
    """
    suggestions, note = _suggest(module, name)
    if note is None and len(suggestions) == 1:
        note = NOTES.get(suggestions[0])
    message = f'module {module!r} has no attribute {name!r}.'
    if suggestions:
        message += f' Did you mean {" or ".join(suggestions)}?'
    if note:
        message += f' {note}'
    return f'{message} {POINTER[module]}'


def _names() -> dict[str, dict[str, str]]:
    """Returns the public names of both modules, keyed by lower case."""
    main = sys.modules[MAIN]
    sel = sys.modules[SEL]
    return {MAIN: {n.lower(): n for n in [*main.__all__, 'set']},
            SEL: {n.lower(): n for n in sel.__all__}}


def _parse(low: str) -> tuple[str | None, str | None, str]:
    """Splits a guessed name into its intent, its prefix and the rest."""
    for intent, prefixes in (('select', SELECT_PREFIXES),
                             ('create', CREATE_PREFIXES)):
        for prefix in prefixes:
            if low.startswith(prefix) and len(low) > len(prefix):
                return intent, prefix, low[len(prefix):]
    return None, None, low


def _call(helper: str, kind: str | None = None) -> str:
    """
    Returns a call of a selection helper, e.g.
    `mphkit.sel.all(geom, 'domain')`.
    """
    shape = SEL_CALLS.get(helper, SEL_CALL)
    if kind is None and '{kind}' in shape:
        raise ValueError(f'sel.{helper} needs an entity kind.')
    return f'{SEL}.{helper}' + shape.format(kind=repr(kind))


def _select_note(kind: str) -> str:
    """Explains how entities of a kind are selected."""
    note = (f'Entities are selected by location, e.g. '
            f'{_call("box", kind)} or {_call("all", kind)}.')
    if kind == 'point':
        note += ' mphkit.point creates a point.'
    return note


Result = tuple[list[str], 'str | None']


def _suggest(module: str, name: str) -> Result:
    """Returns the qualified names that `name` probably meant, and a note."""
    names = _names()
    intent, prefix, rest = _parse(name.lower())
    if rest in ('evaluate', 'eval'):
        return [], EVALUATE_NOTE
    if module == MAIN and name.lower() in EXACT:
        return [], EXACT[name.lower()]
    if module == MAIN and rest in GUESSED:
        meant, guessed = GUESSED[rest]
        return list(meant), guessed
    # values of a sweep or of steps, as in `outer_sweep_values`
    if module == MAIN and rest.endswith(('_values', '_value')):
        parts = set(rest.split('_'))
        if parts & {'outer', 'sweep', 'sweeps', 'parametric'}:
            return [f'{MAIN}.outer_values'], None
        if parts & {'step', 'steps', 'time', 'times', 'inner'}:
            return [f'{MAIN}.step_values'], None
    words = rest.split('_')
    if 'mesh' in words and any(word in PICTURES for word in words):
        return [f'{MAIN}.image'], ('Pass mesh=True for a picture of the '
                                   'mesh. mphkit.mesh_quality gives the '
                                   'numbers.')
    if 'mesh' in words and any(word in QUALITY for word in words):
        return [f'{MAIN}.mesh_quality'], None

    # Selecting or finding entities of a kind, as in `select_faces`
    if intent == 'select':
        if prefix == 'get_' and rest in MEANINGS:
            return _meant(rest), None
        kind = entity_suggestion(rest)
        if kind:
            if prefix == 'select_':
                return [], _select_note(kind)
            found = [_call('find', kind)]
            if prefix == 'get_':
                found.append(_call('entities'))
            return found, None
    # In `mphkit.sel`, an entity word asks for a selection (`sel.lines`)
    if module == SEL:
        kind = entity_suggestion(rest)
        if kind:
            return [], _select_note(kind)
    direct = _direct(module, names, intent, rest)
    if direct:
        return direct
    if module == MAIN:
        kind = entity_suggestion(rest)
        if kind:
            return [], _select_note(kind)
    # A selection helper and an entity kind, as in `all_boundaries`, or a
    # helper and what it acts on, as in `measure_volume`
    head, _, tail = rest.partition('_')
    # A module prefix, as in `sel_box` for `sel.box`
    if head == 'sel' and tail:
        found, note = _suggest(SEL, tail)
        if note or any(s.startswith(SEL) for s in found):
            return found, note
    last = rest.rsplit('_', 1)[-1]
    if tail and last in RESULTS and head not in RESULT_FIRST:
        direct = _direct(module, names, intent, last.removesuffix('s'))
        if direct:
            return direct
    kind = entity_suggestion(tail) if tail else None
    real = names[SEL].get(head)
    if kind and real and '{kind}' in SEL_CALLS.get(real, SEL_CALL):
        return [_call(real, kind)], None
    if tail:
        direct = _direct(module, names, intent, head)
        if direct:
            return direct
    # Plurals, as in `boxes`
    for stem in (rest[:-2] if rest.endswith('es') else None,
                 rest[:-1] if rest.endswith('s') else None):
        if stem:
            direct = _direct(module, names, intent, stem)
            if direct:
                return direct
    return _close(module, names, rest), None


def _direct(module: str, names: dict[str, dict[str, str]],
            intent: str | None, key: str) -> Result | None:
    """Suggests helpers for a shape word, a helper name or a meaning."""
    if key in SHAPES:
        select, create = SHAPES[key]
        both = [f'mphkit.sel.{select} (select)', f'mphkit.{create} (create)']
        if intent == 'create' or (intent is None and module == MAIN
                                  and key not in names[SEL]):
            both.reverse()
        return both, None
    order = {'select': (SEL, MAIN), 'create': (MAIN, SEL)}.get(
        intent or '', (module, SEL if module == MAIN else MAIN))
    for where in order:
        if key in names[where]:
            return [f'{where}.{names[where][key]}'], None
    if key in MEANINGS and MEANINGS[key] != module:
        found = _meant(key)
        # In mphkit.sel, a word for a Boolean operation means the selection
        # operation of that name
        meant = found[0].rsplit('.', 1)[1]
        if module == SEL and len(found) == 1 and meant in names[SEL]:
            return [f'{SEL}.{names[SEL][meant]}'], None
        return found, None
    if key in FEATURES:
        return [f'mphkit.feature(geom, {FEATURES[key]!r}, ...)'], None
    return None


def _meant(key: str) -> list[str]:
    """Returns the helpers a meaning word stands for."""
    meant = MEANINGS[key]
    return list(meant) if isinstance(meant, tuple) else [meant]


def _close(module: str, names: dict[str, dict[str, str]],
           key: str) -> list[str]:
    """Returns up to three names close to `key`, from both modules."""
    scored = []
    for where in (module, SEL if module == MAIN else MAIN):
        for low, real in names[where].items():
            ratio = SequenceMatcher(None, key, low).ratio()
            if ratio >= 0.7:
                scored.append((-ratio, where != module, f'{where}.{real}'))
    scored.sort()
    if not scored:
        return []
    best = -scored[0][0]
    found: list[str] = []
    for ratio, _, qualified in scored:
        if -ratio >= best - 0.1 and qualified not in found:
            found.append(qualified)
    return found[:3]
