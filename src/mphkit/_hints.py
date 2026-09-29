"""
Suggestions for names that do not exist in `mphkit` or `mphkit.sel`.

People and AI assistants guess helper names: `mk.box` for `mk.sel.box`,
`mk.select_box`, `mk.sel_box`, `mk.sel.block`. Both modules answer an
unknown name with the helper that was probably meant and a pointer to
`help()`.

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
MEANINGS = {
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
    'plot': 'mphkit.image', 'picture': 'mphkit.image',
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
}

# Notes added when a helper is the only suggestion.
NOTES = {
    'mphkit.union': 'Pass intbnd=False to merge touching or overlapping '
                    'objects into one domain.',
    'mphkit.image': 'Pictures of the geometry and selections; plot results '
                    'with MPh.',
    'mphkit.sel.neighbors': 'mphkit.sel.adjacent makes a selection instead.',
}

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


def _message(module: str, name: str) -> str:
    """Builds the error message with suggestions and a pointer to help()."""
    suggestions, note = _suggest(module, name)
    if note is None and len(suggestions) == 1:
        note = NOTES.get(suggestions[0])
    message = f'module {module!r} has no attribute {name!r}.'
    if suggestions:
        message += f' Did you mean {" or ".join(suggestions)}?'
    if note:
        message += f' {note}'
    return f'{message} See help({module}) for all helpers.'


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

    # Selecting or finding entities of a kind, as in `select_faces`
    if intent == 'select':
        if prefix == 'get_' and rest in MEANINGS:
            return [MEANINGS[rest]], None
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
        # In mphkit.sel, a word for a Boolean operation means the selection
        # operation of that name
        meant = MEANINGS[key].rsplit('.', 1)[1]
        if module == SEL and meant in names[SEL]:
            return [f'{SEL}.{names[SEL][meant]}'], None
        return [MEANINGS[key]], None
    if key in FEATURES:
        return [f'mphkit.feature(geom, {FEATURES[key]!r}, ...)'], None
    return None


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
