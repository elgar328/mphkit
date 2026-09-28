"""
Suggestions for names that do not exist in `mphkit` or `mphkit.sel`.

People and AI assistants guess helper names: `mk.box` for `mk.sel.box`,
`mk.select_box`, `mk.sel.block`. Both modules answer an unknown name with
the helper that was probably meant and a pointer to `help()`.

The hook is a module subclass rather than a module-level `__getattr__`:
type checkers treat a module with `__getattr__` as having every attribute,
so editors would stop flagging misspelled names, and `help()` would list
the hook among the helpers.
"""
from __future__ import annotations

import sys
from difflib import get_close_matches
from types import ModuleType

from ._comsol import entity_suggestion

MAIN = 'mphkit'
SEL = 'mphkit.sel'

# Prefixes of guessed names, as in `create_block` or `select_box`.
PREFIXES = ('create_', 'make_', 'add_', 'new_', 'get_', 'select_', 'find_')

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
    message = f'module {module!r} has no attribute {name!r}.'
    if suggestions:
        message += f' Did you mean {" or ".join(suggestions)}?'
    if note:
        message += f' {note}'
    return f'{message} See help({module}) for all helpers.'


def _names() -> dict[str, list[str]]:
    """Returns the public names of both modules."""
    main = sys.modules[MAIN]
    sel = sys.modules[SEL]
    return {MAIN: sorted(set(main.__all__) | {'set'}), SEL: sorted(sel.__all__)}


def _suggest(module: str, name: str) -> tuple[list[str], str | None]:
    """Returns qualified names that `name` probably meant, and a note."""
    names = _names()
    other = SEL if module == MAIN else MAIN
    low = name.lower()
    keys = [low] + [low[len(p):] for p in PREFIXES
                    if low.startswith(p) and len(low) > len(p)]
    if module == SEL:
        for key in keys:
            kind = entity_suggestion(key)
            if kind:
                return [], (f'Entities are selected by location, e.g. '
                            f'mphkit.sel.box(geom, {kind!r}, ...) or '
                            f'mphkit.sel.all(geom, {kind!r}).')
    for key in keys:
        if key in SHAPES:
            select, create = SHAPES[key]
            both = [f'mphkit.sel.{select} (select)',
                    f'mphkit.{create} (create)']
            if module == MAIN and key not in names[SEL]:
                both.reverse()
            return both, None
    for key in keys:
        for where in (module, other):
            if key in names[where]:
                return [f'{where}.{key}'], None
    for key in keys:
        if key in MEANINGS:
            return [MEANINGS[key]], None
    close = []
    for where in (module, other):
        close += [f'{where}.{n}'
                  for n in get_close_matches(low, names[where], n=3)]
    return close[:3], None
