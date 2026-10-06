"""
The differences between two models' settings, from `mk.describe`. Public
as `mk.compare`.

Nodes are paired without their tags: by name where expressions call them
by name (parameters, variables, functions, operators, probes), else by
type and the place of their selections, measured on a table that pairs
the entities of two geometries by bounding box and size. Tags of model b
are then translated to those of model a, so that values can be compared.

The module is plain Python on the dicts `mk.describe` returns (also read
back from JSON); only `compare` itself calls COMSOL, to describe a model.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from collections.abc import Iterable
from typing import Any

from mph.model import Model

from . import _describe

# What `ignore` may leave out besides kinds, and what `show` may add
IGNORABLE = frozenset({'empty', 'applied', 'order', 'solver', 'mesh',
                       'expression', 'unchecked', 'note', 'same_applied'})
SHOWABLE = frozenset({'label', 'material_info'})
# Sections of the result, in order
SECTIONS = {'parameter': 0, 'geometry': 1, 'unchecked': 3, 'note': 4}
# Properties that name a node, compared through the pairing instead
NAME_KEYS = frozenset({'funcname', 'opname', 'probename', 'funcnametable'})
# Material properties that describe a library entry, not the material
MATERIAL_INFO = ('sys', 'argders')
# Mesh size properties in the geometry's length unit
LENGTH_KEYS = frozenset({'hmax', 'hmin', 'blhmin', 'blhtot', 'cellsize'})
# Mesh numbers that do not depend on the length unit (measured: equal
# defaults in a geometry in m and in mm)
PLAIN_MESH_KEYS = frozenset({
    'adapsolnum', 'blhminfact', 'blnlayers', 'blstretch', 'buildtime',
    'dimension', 'elemcount', 'elementspar', 'elemratio', 'globalminpar',
    'hauto', 'hcurve', 'hgrad', 'hmeshgrad', 'hnarrow', 'horder',
    'layerdec', 'maxcoarsening', 'maxrefinement', 'minangle', 'numcell',
    'numelem', 'numrefine', 'refinement', 'scale', 'smoothmaxdepth',
    'smoothmaxiter', 'splitangle', 'splitdivangle', 'splitminangle',
    'trimmaxangle', 'trimminangle', 'weights', 'worstpar', 'xscale',
    'yscale', 'zscale'})
# Relative tolerances: sizes (rendering meshes), values, boxes of curved
# pieces measured on different seams
SIZE_RTOL = 1e-3
VALUE_RTOL = 1e-9
CURVED = 1e-3
AXES = 'xyz'
LEVELS_ONLY = ('global', 'geometry', 'remaining', 'none')
WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
PATH = re.compile(r'[A-Za-z_]\w*(/[A-Za-z_]\w*)+')
NUMBER = re.compile(r'[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?')
RANGE = re.compile(r'range\(([^(),]+),([^(),]+),([^(),]+)\)')
# A file name: with a backslash, or from the root, home or a drive
FILE = re.compile(r'.*\\|/|\./|~|[A-Za-z]:[\\/]')


def compare(a: Model | dict, b: Model | dict, /, *,
            tolerance: float = 1e-6, ignore: str | Iterable[str] = (),
            show: str | Iterable[str] = ()) -> list[dict]:
    """
    Returns the differences between the settings of two models, as a
    list of dicts, e.g. to check a script that rebuilds a model made in
    the COMSOL Desktop:

    ```python
    for d in mk.compare(old, model):
        print(d['kind'], d['message'])
    # property Temperature 1 (a ht/temp1, b ht/temp3): T0 is 100[degC]
    #     in a, 50[degC] in b
    ```

    `a` and `b` are models or results of `mk.describe` (also read back from
    JSON); the list is empty where nothing differs. Solvers are compared
    when both models are described with `solver=True` (a model without a
    solver sequence counts as COMSOL's own); a note says so when only one
    was. Nodes are paired without their tags or order: by name where
    expressions call them by name (parameters, variables, functions,
    operators, probes, mass properties, global equations), else by type and
    where their selections lie, on a table that pairs the entities of the
    two geometries by bounding box and size (also where a face is split into
    pieces differently); a node that selects nothing pairs with one of the
    same tag and label; meshes pair by geometry, the same tag first, then
    in order. Values are compared after model b's tags in them are
    translated to model a's (`ht2.T`, `comp2.`, an operator called by
    another name, `mass1.mass`); values COMSOL evaluates to the same SI
    value and unit are equal ('100[degC]' and '373.15[K]'), lists like
    'range(0,0.1,1)' are compared by their numbers.

    Each item has `kind`, `path` and `label` (each {'a', 'b'}; None on
    the side that lacks it), `message` (one line, naming each side's
    path without its component) and the values `a` and `b` as each
    model has them. Items come in this order: parameters, geometries
    (and components only one model has), the differences of nodes, then
    'unchecked' and 'note'; within each, what only model b has comes
    last. Kinds: 'parameter', 'geometry' (dimension, bounding box,
    entities without a counterpart, union against assembly),
    'only_in_a', 'only_in_b' (`a` or `b` holds the whole node; `empty:
    True` if it selects nothing; a component only one model has is one
    item for its geometries, physics, multiphysics couplings, meshes,
    pairs, mass properties and materials, with what its operators,
    functions, probes, definitions, variables and coordinate systems
    give as its `consequences`),
    'property' (`name` of the setting; `from_default: True` if one side
    has it from its defaults), 'expression' (same value, but one side
    uses parameters or leaves out the unit), 'variable', 'active',
    'selection' (with the entity `numbers` that differ; `same_applied:
    True` if both apply to the same entities), 'applied' (where a node
    applies differs in the entities both models select: another node
    overrides it in one of them), 'order' (of mesh operations or study
    steps; the nodes that moved), 'solver' and 'label'; then 'unchecked'
    (what could not be compared, e.g. unknown defaults) and 'note'. A
    'property' item with `used_only: True` lists values one side sets
    and uses while the other's settings leave them unused (e.g. sizes of
    a mesh node with `custom` on in one model, next to the `custom` item
    itself). Places are in each model's length unit; comparisons are in
    SI.

    An 'applied' item goes into the `consequences` of the item that
    explains it, its first `causes` (the paths of all candidates): a
    node of the same interface (of any interface where the component has
    couplings) or the same component's materials that selects other
    entities, is only in one model or enabled in one only, or a pair for
    a node that applies on pairs only, covering every entity where it
    differs. A selection explains its own level only; an interface on in
    one model only explains all levels. Nodes under a node disabled in
    one model are in its 'active' item. Messages end in '(+N
    consequences)'; `causes` may name an item `ignore` hides.

    Where a geometry differs, what follows from it is in the geometry item's
    `consequences` (fix the geometry first): selections that differ only in
    entities without a counterpart, nodes only one model has where the other
    has some of that type, and mesh lengths (`hmax`, `hmin`, boundary layer
    thicknesses) from the defaults. A selection that also differs in
    entities both geometries have stays at the top with those entities only;
    `elsewhere` counts the others. Settings stay at the top; a node paired
    only by order there has `matched_by_order: True` and its messages end in
    '(paired by order)'.

    `tolerance` is relative to the size of the geometries. `ignore` takes
    kinds and 'empty', 'mesh' (all of the meshes) and 'same_applied'
    (one name or several).
    Labels and library entries of materials (`material_info: True`, e.g.
    'sys' against a library's 'none' or a function's derivatives) are
    hidden unless `show` names 'label' or 'material_info'; `show` wins
    over `ignore`. The consequences of a hidden item take its place.

    Not compared: results, the order of physics features and materials
    (where they apply is: a cause may come before the node it
    overrides), expressions COMSOL cannot evaluate other than as
    written ('2*a' and 'a*2' differ), the two faces of a pair in an
    assembly and other entities with the same box and size (they are one
    row of the table), a probe's name used as a variable, the same mass
    properties name in two components, where boundary elements apply on
    the exterior (domain 0; where they select it is compared), tags in
    file names with a relative path ('data/comp1.mph'), tags in
    properties other than the ones of nodes, physics, materials,
    coordinate systems, pairs, meshes, studies and solvers (e.g. load
    groups), a mesh deleted and made again in another order (its tag
    pairs it with another; the differences of the two show), and a
    feature of global equations without equations that one model has.
    Compared although they may be unused: a probe's `intsurface` and
    `intvolume` outside 3D, a sweep's `filename` while both save to a
    file (its default differs from one computer to the next), values
    picked by a choice whose name does not end in `_src` (e.g. `k` while
    `k_mat` is 'from_mat') and the settings of physics interfaces.
    """
    ignored, shown_ = _names(ignore), _names(show)
    unknown = (ignored - IGNORABLE - SHOWABLE - set(KINDS)) | \
        (shown_ - SHOWABLE)
    if unknown:
        raise ValueError(
            f'mk.compare does not know {sorted(unknown)}; ignore takes '
            f'{sorted(IGNORABLE | SHOWABLE | set(KINDS))}, show takes '
            f'{sorted(SHOWABLE)}.')
    if isinstance(tolerance, bool) or \
            not isinstance(tolerance, (int, float)) or \
            not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f'mk.compare takes a tolerance of 0 or more, not '
                         f'{tolerance!r}.')
    plain = [name for name, side in (('a', a), ('b', b))
             if not isinstance(side, Model)]
    described_a, described_b = _described(a, 'a'), _described(b, 'b')
    try:
        found = _Comparison(described_a, described_b, tolerance).run()
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        if not plain:
            raise
        raise ValueError(
            f'mk.compare could not read {" or ".join(plain)} '
            f'({type(error).__name__}: {error}). Pass complete results of '
            'mk.describe; if they are, this is a bug in mphkit.'
        ) from error
    hidden = (ignored | {'label', 'material_info'}) - shown_
    return _filtered(found, hidden)


def _names(value) -> set[str]:
    """The names given to `ignore` or `show`: one name or several."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict) or not isinstance(value, Iterable) or \
            not all(isinstance(name, str) for name in value):
        raise ValueError(f'mk.compare takes a name or a list of names for '
                         f'ignore and show, not {value!r}.')
    return set(value)


# Lists at the top of a result of mk.describe
LISTS = ('functions', 'variables', 'couplings', 'coordinate_systems',
         'materials', 'definitions', 'probes', 'studies', 'solutions',
         'notes')

KINDS = ('parameter', 'geometry', 'only_in_a', 'only_in_b', 'property',
         'expression', 'variable', 'active', 'selection', 'applied',
         'order', 'solver', 'label', 'unchecked', 'note')


def _described(side, name: str) -> dict:
    if isinstance(side, Model):
        return _describe.describe(side)
    if not isinstance(side, dict):
        raise TypeError(f'mk.compare takes models or results of '
                        f'mk.describe, not {side!r} as {name}.')
    if side.get('format') != _describe.FORMAT:
        raise ValueError(
            f'{name} has format {side.get("format")!r}, this mphkit '
            f'compares format {_describe.FORMAT}: describe the model again '
            'with this mphkit.')
    problem = _shape_problem(side)
    if problem:
        raise ValueError(f'{name} is not a complete result of mk.describe: '
                         f'{problem}.')
    return side


def _shape_problem(side: dict) -> str | None:
    """Tells what is wrong with the top level of a result of
    mk.describe, or None."""
    if not isinstance(side.get('parameters'), dict):
        return "'parameters' is no dict"
    components = side.get('components')
    if not isinstance(components, list):
        return "'components' is no list"
    for component in components:
        if not isinstance(component, dict) or 'tag' not in component:
            return 'a component has no tag'
        if not isinstance(component.get('geometries'), list):
            return f"component {component['tag']!r} has no list " \
                "'geometries'"
    for key, value in side.items():
        if key in LISTS and not isinstance(value, list):
            return f'{key!r} is no list'
    return None


def _filtered(items: list[dict], hidden: set[str]) -> list[dict]:
    """
    Leaves out what `ignore` names, also among consequences (what a
    hidden item explains moves up in its place), counts the consequences
    left in the message and drops the internal keys (starting with '_').
    """
    return _stripped(_kept(items, hidden))


def _kept(items: list[dict], hidden: set[str]) -> list[dict]:
    found = []
    for item in items:
        consequences = _kept(item['consequences'], hidden) \
            if 'consequences' in item else None
        if item['kind'] in hidden or \
                ('empty' in hidden and item.get('empty')) or \
                ('same_applied' in hidden and item.get('same_applied')) or \
                ('material_info' in hidden and item.get('material_info')) \
                or ('mesh' in hidden and item.get('_mesh')):
            found.extend(consequences or [])
            continue
        item = dict(item)
        if consequences is not None:
            item['consequences'] = consequences
            count = len(consequences)
            if count:
                item['message'] += f' (+{count} consequence' + \
                    ('s' if count > 1 else '') + (
                    ': fix the geometry first)' if item['kind'] == 'geometry'
                    else ')')
            elif item['kind'] != 'geometry':
                del item['consequences']
        found.append(item)
    found.sort(key=_section)
    return found


def _stripped(items: list[dict]) -> list[dict]:
    found = []
    for item in items:
        item = {key: value for key, value in item.items()
                if not key.startswith('_')}
        if 'consequences' in item:
            item['consequences'] = _stripped(item['consequences'])
        found.append(item)
    return found


def _section(item: dict) -> tuple:
    part = item.get('_part', SECTIONS.get(item['kind'], 2))
    return (part, item['kind'] == 'only_in_b', item.get('_order', 0))


##########
# Values #
##########

def spaced(text: str) -> str:
    """
    Collapses the white space of an expression: runs become one space,
    kept only between two words or numbers ('0 10 20' stays, '1 + 2'
    becomes '1+2').
    """
    text = re.sub(r'\s+', ' ', text.strip())
    # a space stays between two numbers or words, also before a sign
    # that starts a number ('0 -1' is a list, '0-1' a difference)
    return re.sub(r'(?<=[^\w.)]) | (?![\w.]|[+-][\d.])', '', text)


def numbers(text: str) -> list[float] | None:
    """
    Returns the numbers of a list such as 'range(0,0.1,1)', '0 0.1 0.2',
    '0,0.1' or a mix of ranges and numbers, as COMSOL lists them, or None
    if it is no such list of plain numbers.
    """
    found: list[float] = []
    text = re.sub(r'\s*([(),])\s*', r'\1', re.sub(r'\s+', ' ', text.strip()))
    if not text:
        return None
    for token in re.split(r'[ ,](?![^(]*\))', text):
        match = RANGE.fullmatch(token)
        if match:
            try:
                start, step, stop = (float(v) for v in match.groups())
            except ValueError:
                return None
            if not all(math.isfinite(v) for v in (start, step, stop)) or \
                    step == 0 or (stop - start) * step < 0:
                return None
            # too long to compare value by value
            if (stop - start) / step + 1 > 10**6:
                return None
            slack = 1e-9 * abs(step)
            count = 0
            while True:
                value = start + count * step
                if (value - stop) * math.copysign(1, step) > slack:
                    break
                if count >= 10**6:
                    return None     # too long to compare value by value
                found.append(value)
                count += 1
        elif NUMBER.fullmatch(token):
            found.append(float(token))
        else:
            return None
    return found


def close(x: float, y: float, rtol: float = VALUE_RTOL,
          atol: float = 0.0) -> bool:
    return abs(x - y) <= max(rtol * max(abs(x), abs(y)), atol)


def uses_parameters(text: str, parameters: frozenset[str] | set[str]
                    ) -> bool:
    """
    Tells whether an expression uses parameters: names outside units
    ('[...]') that are not members ('ht.h') or functions.
    """
    text = re.sub(r'\[[^\]]*\]', '', text)
    for match in WORD.finditer(text):
        start, end = match.span()
        if start and text[start - 1] == '.':
            continue
        if text[end:].lstrip().startswith(('(', '.')):
            continue
        if match.group() in parameters:
            return True
    return False


def same_value(a, b, si_a=None, si_b=None,
               parameters: frozenset[str] = frozenset()) -> str | None:
    """
    Compares two values of a property (model b's tags already translated
    to model a's) with their SI values from `mk.describe` if known.
    Returns None if they are equal, 'expression' if they evaluate the
    same but are written differently in a way that matters (one uses
    parameters, or only one has a unit), else 'property'.
    """
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == 1 and len(b) == 3 or len(a) == 3 and len(b) == 1:
            # an isotropic tensor as one value or as its diagonal
            one, three = (a, b) if len(a) == 1 else (b, a)
            si_one, si_three = (si_a, si_b) if len(a) == 1 else (si_b, si_a)
            if all(same_value(one[0], item, _item(si_one, 0),
                              _item(si_three, i), parameters) is None
                   for i, item in enumerate(three)):
                return None
            return 'property'
        if len(a) != len(b):
            return 'property'
        worst = None
        for i, (x, y) in enumerate(zip(a, b)):
            found = same_value(x, y, _item(si_a, i), _item(si_b, i),
                               parameters)
            if found == 'property':
                return found
            worst = worst or found
        return worst
    if isinstance(a, str) and isinstance(b, str):
        if spaced(a) == spaced(b):
            return None
        listed_a, listed_b = numbers(a), numbers(b)
        if listed_a is not None and listed_b is not None and \
                (len(listed_a) > 1 or len(listed_b) > 1):
            scale = max([abs(v) for v in listed_a + listed_b] + [0.0])
            if len(listed_a) == len(listed_b) and all(
                    close(x, y, atol=1e-9 * scale)
                    for x, y in zip(listed_a, listed_b)):
                return None
            return 'property'
        if isinstance(si_a, dict) and isinstance(si_b, dict) and \
                _close_values(si_a.get('value'), si_b.get('value')):
            if si_a.get('unit') == si_b.get('unit'):
                if uses_parameters(a, parameters) or \
                        uses_parameters(b, parameters):
                    return 'expression'
                return None
            if '1' in (si_a.get('unit'), si_b.get('unit')):
                return 'expression'
        return 'property'
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            not isinstance(a, bool) and not isinstance(b, bool):
        return None if close(float(a), float(b)) else 'property'
    return None if a == b and type(a) is type(b) else 'property'


def _item(si, index: int):
    if isinstance(si, list) and index < len(si):
        return si[index]
    return si if isinstance(si, dict) else None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _close_values(x, y) -> bool:
    if _number(x) and _number(y):
        return close(float(x), float(y))
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return all(_close_values(p, q) for p, q in zip(x, y))
    if _number(x) and isinstance(y, list):     # real against complex
        return _close_values([x, 0.0], y)
    if isinstance(x, list) and _number(y):
        return _close_values(x, [y, 0.0])
    return False


def shown(value, limit: int = 80) -> str:
    """Writes a value for a message, cut to `limit` characters."""
    if value is None:
        return '(none)'
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def same_text(a, b) -> bool:
    """Tells whether two values read the same, white space aside."""
    if isinstance(a, str) and isinstance(b, str):
        return spaced(a) == spaced(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(same_text, a, b))
    return a == b


def in_a_tags(raw_a, raw_b, translated_b, limit: int = 80) -> str:
    """
    The end of a message where b's value reads as a's but names other
    nodes: b's value in a's tags ('ht' in b may be another interface than
    in a); '' otherwise.
    """
    if translated_b != raw_b and same_text(raw_a, raw_b):
        return f" (in a's tags: {shown(translated_b, limit)})"
    return ''


def place_text(place: dict) -> str:
    """Writes the box of an entity, e.g. 'x=100, y 0..50, z 0..10'."""
    if 'unknown' in place:
        return f'entity {place["unknown"]} (not measured)'
    # single precision noise next to large coordinates reads as 0
    scale = max([abs(v) for axis in AXES if axis in place
                 for v in place[axis]] + [0.0])

    def plain(value: float) -> float:
        return 0.0 if abs(value) <= 1e-9 * scale else value

    parts = []
    for axis in AXES:
        if axis in place:
            low, high = (plain(v) for v in place[axis])
            parts.append(f'{axis}={low:.6g}' if low == high
                         else f'{axis} {low:.6g}..{high:.6g}')
    return ', '.join(parts)


def places_text(places: list[dict], limit: int = 3) -> str:
    text = '; '.join(place_text(p) for p in places[:limit])
    if len(places) > limit:
        text += f'; +{len(places) - limit} more'
    return text


##################
# Entity tables  #
##################

def level_dims(sdim: int) -> dict[str, int]:
    """Returns the entity levels of a geometry by name, with their dim."""
    found = {}
    for dim in range(sdim, -1, -1):
        name = _describe._comsol.entity_level_name(dim, sdim)
        if name:
            found[name] = dim
    return found


def place_key(place: dict) -> str:
    return json.dumps(place, sort_keys=True)


class Table:
    """
    The entities of one level of a geometry in SI units. Entities with
    the same box and size (the two faces of a pair in an assembly) are
    one row.
    """

    def __init__(self, places: list, scale: float, dim: int):
        self.rows: list[dict] = []
        self.row_of: dict[int, int] = {}
        self.by_place: dict[str, int] = {}
        for number, place in enumerate(places, 1):
            if 'unknown' in place:
                continue
            key = place_key(place)
            if key in self.by_place:
                row = self.by_place[key]
                self.rows[row]['numbers'].append(number)
            else:
                row = len(self.rows)
                self.by_place[key] = row
                self.rows.append({
                    'box': [(place[a][0] * scale, place[a][1] * scale)
                            for a in AXES if a in place],
                    'size': place['size'] * scale**dim
                    if isinstance(place.get('size'), (int, float)) else None,
                    'numbers': [number], 'place': place})
            self.row_of[number] = row

    def rows_of(self, places) -> set[int] | None:
        """The rows of a selection's places ('all' for all), None if one
        is not in the table."""
        if places == 'all':
            return set(range(len(self.rows)))
        found = set()
        for place in places:
            row = self.by_place.get(place_key(place))
            if row is None:
                return None
            found.add(row)
        return found

    def size(self, rows: Iterable[int]) -> float:
        """Total size of rows; points count one each."""
        total = 0.0
        for row in rows:
            size = self.rows[row]['size']
            total += 1.0 if size is None else size
        return total

    def box(self, rows: Iterable[int]) -> list[tuple[float, float]]:
        boxes = [self.rows[r]['box'] for r in rows]
        return [(min(b[i][0] for b in boxes), max(b[i][1] for b in boxes))
                for i in range(len(boxes[0]))]


class Tolerance:
    """Position tolerance: relative to the model size, plus the single
    precision of COMSOL's boxes."""

    def __init__(self, tolerance: float, size: float):
        self.absolute = tolerance * size
        self.size = size

    def of(self, x: float) -> float:
        return self.absolute + 2e-7 * abs(x)

    def boxes(self, one, other, slack: float = 0.0) -> bool:
        return len(one) == len(other) and all(
            abs(a - c) <= self.of(a) + slack and
            abs(b - d) <= self.of(b) + slack
            for (a, b), (c, d) in zip(one, other))

    def inside(self, inner, outer) -> bool:
        return all(a >= c - self.of(c) and b <= d + self.of(d)
                   for (a, b), (c, d) in zip(inner, outer))


def sizes_close(x, y) -> bool:
    if x is None or y is None:
        return x is y
    return close(x, y, SIZE_RTOL, 1e-300)


def diagonal(box) -> float:
    return math.sqrt(sum((b - a)**2 for a, b in box))


def match_tables(ta: Table, tb: Table, tol: Tolerance):
    """
    Pairs the rows of two tables: one to one by box and size, then one to
    many where pieces left over on one side fill exactly the box and size
    of one left on the other. Returns the cells (sets of rows of a and of
    b that are the same region) and the rows left on each side.
    """
    cells: list[tuple[frozenset, frozenset]] = []
    width = max(tol.absolute, 2e-7 * max(
        [abs(v) for row in ta.rows + tb.rows for pair in row['box']
         for v in pair] + [0.0]), 1e-30) * 4
    buckets: dict[tuple, list[int]] = {}

    def key(row) -> tuple:
        return tuple(round((lo + hi) / 2 / width) for lo, hi in row['box'])

    for j, row in enumerate(tb.rows):
        buckets.setdefault(key(row), []).append(j)
    used_b: set[int] = set()
    left_a = []
    for i, row in enumerate(ta.rows):
        center = key(row)
        found = None
        for offset in _offsets(len(center)):
            near = tuple(c + o for c, o in zip(center, offset))
            for j in buckets.get(near, []):
                if j in used_b:
                    continue
                other = tb.rows[j]
                if tol.boxes(row['box'], other['box']) and \
                        sizes_close(row['size'], other['size']):
                    found = j
                    break
            if found is not None:
                break
        if found is None:
            left_a.append(i)
        else:
            used_b.add(found)
            cells.append((frozenset({i}), frozenset({found})))
    left_b = [j for j in range(len(tb.rows)) if j not in used_b]
    for one, many, flip in ((ta, tb, False), (tb, ta, True)):
        singles = left_b if flip else left_a
        pieces_left = left_a if flip else left_b
        # the pieces by the low end of their x range, to find the ones in
        # a box without trying them all
        position = {j: k for k, j in enumerate(pieces_left)}
        lows = sorted((many.rows[j]['box'][0][0], j) for j in pieces_left)
        starts = [low for low, _ in lows]
        for i in list(singles):
            row = one.rows[i]
            if row['size'] is None:
                continue
            low, high = row['box'][0]
            first = bisect.bisect_left(starts, low - tol.of(low))
            last = bisect.bisect_right(starts, high + tol.of(high))
            spanned = sorted((j for _, j in lows[first:last]
                              if j in position), key=position.__getitem__)
            pieces = [j for j in spanned
                      if tol.inside(many.rows[j]['box'], row['box'])]
            if not pieces:
                continue
            if sizes_close(row['size'], many.size(pieces)) and \
                    tol.boxes(row['box'], many.box(pieces)):
                singles.remove(i)
                for j in pieces:
                    pieces_left.remove(j)
                    del position[j]
                cell = (frozenset(pieces), frozenset({i})) if flip else \
                    (frozenset({i}), frozenset(pieces))
                cells.append(cell)
    return cells, left_a, left_b


def _offsets(dim: int) -> list[tuple]:
    found: list[tuple] = [()]
    for _ in range(dim):
        found = [o + (d,) for o in found for d in (0, -1, 1)]
    return found


def same_region(ta: Table, rows_a, tb: Table, rows_b,
                tol: Tolerance) -> bool:
    """
    Tells whether rows without cells cover the same region: the same
    total size, the same box (with a margin for curved pieces) and flat
    pieces in the same planes (two faces at x = 0 and 1 are not the two
    at z = 0 and 1).
    """
    if not rows_a or not rows_b:
        return not rows_a and not rows_b
    box = ta.box(rows_a)
    return sizes_close(ta.size(rows_a), tb.size(rows_b)) and \
        tol.boxes(box, tb.box(rows_b), CURVED * diagonal(box)) and \
        _same_planes(_planes(ta, rows_a, tol), _planes(tb, rows_b, tol), tol)


def _planes(table: Table, rows, tol: Tolerance) -> set[tuple[int, float]]:
    """The axis-aligned planes flat rows lie in, as (axis, coordinate)."""
    found = set()
    for row in rows:
        for axis, (low, high) in enumerate(table.rows[row]['box']):
            if high - low <= tol.of(low):
                found.add((axis, low))
    return found


def _same_planes(one: set, other: set, tol: Tolerance) -> bool:
    def covered(planes, by):
        return all(any(axis == a and abs(value - v) <= tol.of(value)
                       for a, v in by) for axis, value in planes)
    return covered(one, other) and covered(other, one)


#############
# Geometries #
#############

class GeometryPair:
    """Two geometries, with their entity tables matched level by level."""

    def __init__(self, ga: dict, gb: dict, tolerance: float):
        self.ga, self.gb = ga, gb
        self.scales = (ga.get('length_scale') or 1.0,
                       gb.get('length_scale') or 1.0)
        sizes = [diagonal(self.box(side)) for side in (0, 1)]
        self.tol = Tolerance(tolerance, max(sizes) or 1.0)
        self.dims = level_dims(ga['dimension'])
        self.tables: dict[tuple[int, str], Table] = {}
        self.matches: dict[str, tuple] = {}
        self.cell_maps: dict[tuple[int, str], dict] = {}
        # whether the shapes differ (set by the comparison)
        self.differs = False

    def box(self, side: int) -> list[tuple[float, float]]:
        geometry = (self.ga, self.gb)[side]
        found = geometry.get('bounding_box') or {}
        scale = self.scales[side]
        return [(found[a][0] * scale, found[a][1] * scale)
                for a in AXES if a in found]

    def comparable(self) -> bool:
        return self.ga['dimension'] == self.gb['dimension'] and \
            self.ga.get('axisymmetric') == self.gb.get('axisymmetric')

    def table(self, side: int, level: str) -> Table:
        key = (side, level)
        if key not in self.tables:
            geometry = (self.ga, self.gb)[side]
            self.tables[key] = Table(
                geometry['entities'].get(level, []), self.scales[side],
                self.dims[level])
        return self.tables[key]

    def match(self, level: str):
        if level not in self.matches:
            self.matches[level] = match_tables(
                self.table(0, level), self.table(1, level), self.tol)
        return self.matches[level]

    def top_levels(self) -> list[str]:
        sdim = self.ga['dimension']
        return [name for name, dim in self.dims.items()
                if dim >= sdim - 1 or sdim == 1][:2]

    def leftovers(self) -> dict[str, tuple[list, list]]:
        """The rows of the top levels without a counterpart that do not
        cover the same region on both sides, by level."""
        found = {}
        for level in self.top_levels():
            _, left_a, left_b = self.match(level)
            ta, tb = self.table(0, level), self.table(1, level)
            if (left_a or left_b) and \
                    not same_region(ta, left_a, tb, left_b, self.tol):
                found[level] = (left_a, left_b)
        return found

    def differences(self) -> list[str]:
        """What differs in shape: dimension, axisymmetry, voids, box and
        the entities of the top levels."""
        ga, gb = self.ga, self.gb
        if not self.comparable():
            return [f'dimension {ga["dimension"]} in a, {gb["dimension"]} '
                    'in b' if ga['dimension'] != gb['dimension'] else
                    'axisymmetric in one model only']
        found = []
        if ga.get('voids') != gb.get('voids'):
            found.append(f'{ga.get("voids")} voids in a, {gb.get("voids")} '
                         'in b')
        box_a, box_b = self.box(0), self.box(1)
        if box_a and box_b and not self.tol.boxes(box_a, box_b):
            found.append('bounding box differs')
        if ga.get('finalize') == gb.get('finalize'):
            for level, (left_a, left_b) in self.leftovers().items():
                found.append(self.rows_text(level, left_a, left_b))
        return found

    def rows_text(self, level: str, left_a, left_b) -> str:
        ta, tb = self.table(0, level), self.table(1, level)
        parts = []
        for rows, table, side in ((left_a, ta, 'a'), (left_b, tb, 'b')):
            if rows:
                places = [table.rows[r]['place'] for r in rows]
                parts.append(f'{len(rows)} {level} only in {side} '
                             f'({places_text(places)})')
        return ', '.join(parts)

    def score(self) -> float:
        """How much of both geometries' domains pair up (0 to 1)."""
        if not self.comparable():
            return -1.0
        level = 'domain'
        cells, _, _ = self.match(level)
        ta, tb = self.table(0, level), self.table(1, level)
        total = ta.size(range(len(ta.rows))) + tb.size(range(len(tb.rows)))
        if not total:
            return 1.0
        common = sum(ta.size(ca) + tb.size(cb) for ca, cb in cells)
        return common / total

    def selection(self, sa: dict, sb: dict) -> dict:
        """Compares two selections on this pair of geometries."""
        level = sa['level']
        ea, eb = sa.get('entities'), sb.get('entities')
        if ea == 'all' and eb == 'all':
            return _same()
        if level not in self.dims or not isinstance(ea, (list, str)) or \
                not isinstance(eb, (list, str)):
            return _unknown()
        ta, tb = self.table(0, level), self.table(1, level)
        rows_a, rows_b = ta.rows_of(ea), tb.rows_of(eb)
        if rows_a is None or rows_b is None:
            return _unknown()
        cells, left_a, left_b = self.match(level)
        common = 0.0
        only_a: set[int] = set()
        only_b: set[int] = set()
        differing: set = set()
        for index in self.touched(level, rows_a, rows_b):
            cell_a, cell_b = cells[index]
            in_a, in_b = cell_a & rows_a, cell_b & rows_b
            if in_a == cell_a and in_b == cell_b:
                common += ta.size(cell_a)
            elif in_a or in_b:
                only_a |= in_a
                only_b |= in_b
                differing.add(index)
        # what differs in entities both geometries have, and in the rest
        inside = (set(only_a), set(only_b))
        rest_a = sorted(rows_a & set(left_a))
        rest_b = sorted(rows_b & set(left_b))
        if same_region(ta, rest_a, tb, rest_b, self.tol):
            common += ta.size(rest_a)
        else:
            only_a |= set(rest_a)
            only_b |= set(rest_b)
            differing.add('rest')
        total = max(ta.size(rows_a), tb.size(rows_b))
        found: dict[str, Any] = {
            'same': not only_a and not only_b,
            'overlap': common / total if total else 1.0, 'unknown': False}
        if not found['same']:
            found['cells'] = differing
            found.update(self.split(level, only_a, only_b, inside))
        return found

    def split(self, level: str, only_a: set, only_b: set,
              inside: tuple[set, set]) -> dict:
        """The rows that differ, and where the shapes differ, the part in
        entities both geometries have ('inside', None if none) and how
        many other entities differ ('elsewhere')."""
        ta, tb = self.table(0, level), self.table(1, level)
        found = _rows_found(level, ta, tb, only_a, only_b)
        if self.differs:
            found['inside'] = _rows_found(level, ta, tb, *inside) \
                if inside[0] or inside[1] else None
            found['elsewhere'] = sum(
                len(table.rows[r]['numbers'])
                for rows, table in ((only_a - inside[0], ta),
                                    (only_b - inside[1], tb))
                for r in rows)
        return found

    def touched(self, level: str, rows_a: set, rows_b: set) -> list[int]:
        """The cells (in order) that rows of either side are in."""
        found: set[int] = set()
        for side, rows in ((0, rows_a), (1, rows_b)):
            cells = self.cell_of(side, level)
            found.update(cells[row] for row in rows if cells[row] != 'rest')
        return sorted(found)

    def cell_of(self, side: int, level: str) -> dict:
        """A level's rows of one side by the cell they are in, 'rest' for
        rows without a counterpart."""
        key = (side, level)
        if key not in self.cell_maps:
            cells, left_a, left_b = self.match(level)
            found: dict = {row: 'rest' for row in (left_a, left_b)[side]}
            for index, cell in enumerate(cells):
                for row in cell[side]:
                    found[row] = index
            self.cell_maps[key] = found
        return self.cell_maps[key]

    def applied(self, sa: dict, sb: dict) -> dict | None:
        """
        Compares where two nodes apply, in the cells where they select the
        same entities (a node's own change of selection is a selection
        difference). Returns None if they apply alike, else the rows that
        differ as `selection` does, with the differing `cells` (indices,
        'rest' for the rows without a counterpart).
        """
        level = sa['level']
        ea, eb = sa.get('entities'), sb.get('entities')
        aa, ab = sa.get('applied', ea), sb.get('applied', eb)
        ta, tb = self.table(0, level), self.table(1, level)
        rows: list[set[int]] = []
        for table, places in ((ta, ea), (tb, eb), (ta, aa), (tb, ab)):
            found_rows = table.rows_of(places) \
                if isinstance(places, (list, str)) else None
            if found_rows is None:
                return None
            rows.append(found_rows)
        rows_a, rows_b, applied_a, applied_b = rows
        cells, left_a, left_b = self.match(level)
        only_a: set[int] = set()
        only_b: set[int] = set()
        differing: set = set()

        def alike(cell_a, cell_b, in_a, in_b) -> bool:
            return (in_a == cell_a and in_b == cell_b) or \
                (not in_a and not in_b)

        for index in self.touched(level, rows_a | applied_a,
                                  rows_b | applied_b):
            cell_a, cell_b = cells[index]
            if not alike(cell_a, cell_b, cell_a & rows_a, cell_b & rows_b):
                continue
            in_a, in_b = cell_a & applied_a, cell_b & applied_b
            if not alike(cell_a, cell_b, in_a, in_b):
                only_a |= in_a
                only_b |= in_b
                differing.add(index)
        inside = (set(only_a), set(only_b))
        rest_a, rest_b = set(left_a), set(left_b)
        if rest_a or rest_b:
            selected = (rows_a & rest_a, rows_b & rest_b)
            if (ea == 'all' and eb == 'all') or alike(
                    rest_a, rest_b, *selected) or same_region(
                    ta, sorted(selected[0]), tb, sorted(selected[1]),
                    self.tol):
                in_a, in_b = applied_a & rest_a, applied_b & rest_b
                if not alike(rest_a, rest_b, in_a, in_b) and \
                        not same_region(ta, sorted(in_a), tb, sorted(in_b),
                                        self.tol):
                    only_a |= in_a
                    only_b |= in_b
                    differing.add('rest')
        if not differing:
            return None
        found: dict[str, Any] = {'same': False, 'unknown': False,
                                 'cells': differing}
        found.update(self.split(level, only_a, only_b, inside))
        return found


def _rows_found(level: str, ta: Table, tb: Table, only_a: set,
                only_b: set) -> dict:
    """The entity numbers, places and text of rows that differ."""
    places = {'a': [ta.rows[r]['place'] for r in sorted(only_a)],
              'b': [tb.rows[r]['place'] for r in sorted(only_b)]}
    parts = [f'{len(places[side])} {level} only in {side} '
             f'({places_text(places[side])})'
             for side in 'ab' if places[side]]
    return {'numbers': {
        'a': sorted(n for r in only_a for n in ta.rows[r]['numbers']),
        'b': sorted(n for r in only_b for n in tb.rows[r]['numbers'])},
        'places': places, 'text': ', '.join(parts)}


def _same() -> dict:
    return {'same': True, 'overlap': 1.0, 'unknown': False}


def _unknown() -> dict:
    return {'same': True, 'overlap': 0.0, 'unknown': True}


def _different(text: str) -> dict:
    return {'same': False, 'overlap': 0.0, 'unknown': False, 'text': text}


###############
# Translation #
###############

class Translator:
    """Translates model b's tags in values to model a's."""

    def __init__(self, a_words: set[str], b_words: set[str]):
        self.maps: dict[str, dict[str, str]] = {
            kind: {} for kind in ('component', 'geometry', 'physics',
                                  'identifier', 'multiphysics', 'material',
                                  'coordinate', 'pair', 'study',
                                  'sequence', 'massprop', 'mesh')}
        # by b's identifier (and physics tag): b feature tag to a's
        self.features: dict[str, dict[str, str]] = {}
        # by b's study tag: b step tag to a's
        self.steps: dict[str, dict[str, str]] = {}
        # by b's component (None for global): b name to a's
        self.names: dict[str | None, dict[str, str]] = {}
        self.a_words, self.b_words = a_words, b_words

    def add(self, kind: str, b_tag, a_tag):
        if b_tag is not None and a_tag is not None:
            self.maps[kind][str(b_tag)] = str(a_tag)

    def whole(self, value: str) -> str | None:
        for kind in ('physics', 'multiphysics', 'material', 'coordinate',
                     'pair', 'study', 'sequence', 'mesh'):
            if value in self.maps[kind]:
                return self.maps[kind][value]
        return None

    def value(self, value, component=None):
        """Translates a value of model b: tags, lists, step maps (not
        file names)."""
        if isinstance(value, str):
            found = self.whole(value)
            if found is not None:
                return found
            if PATH.fullmatch(value):
                return self.path(value)
            if FILE.match(value):
                return value
            return self.expression(value, component)
        if isinstance(value, list):
            return [self.value(item, component) for item in value]
        if isinstance(value, dict):
            return {self.key(key): self.value(item, component)
                    for key, item in value.items()}
        return value

    def path(self, value: str) -> str:
        """Translates 'ht/temp1' (a physics feature) or 'std1/stat'."""
        head, _, rest = value.partition('/')
        if head in self.maps['physics']:
            features = self.features.get(head, {})
            return '/'.join([self.maps['physics'][head]] +
                            [features.get(part, part)
                             for part in rest.split('/')])
        if head in self.maps['study']:
            steps = self.steps.get(head, {})
            return '/'.join([self.maps['study'][head]] +
                            [steps.get(part, part)
                             for part in rest.split('/')])
        return value

    def key(self, key: str) -> str:
        """Translates a key of a study step's map: physics, 'multi:...',
        'frame:spatialN', component or geometry tags."""
        for kind in ('physics', 'component', 'geometry'):
            if key in self.maps[kind]:
                return self.maps[kind][key]
        if key.startswith('multi:'):
            tag = key[len('multi:'):]
            return 'multi:' + self.maps['multiphysics'].get(tag, tag)
        match = re.fullmatch(r'(frame:\D+?)(\d+)', key)
        if match:
            mapped = self.maps['component'].get(f'comp{match.group(2)}')
            if mapped and mapped.startswith('comp') and mapped[4:].isdigit():
                return match.group(1) + mapped[4:]
        return key

    def expression(self, text: str, component: str | None = None) -> str:
        """
        Translates the tags in an expression: component prefixes
        ('comp2.T'), physics identifiers and their features ('ht2.q0',
        'ht.pc6.T'), multiphysics tags and calls of functions and
        operators by name ('intop2(T)').
        """
        out = []
        last = 0
        for match in WORD.finditer(text):
            start, end = match.span()
            word = match.group()
            rest = text[end:]
            new = word
            if start and text[start - 1] == '.':
                owner = re.search(r'([A-Za-z_]\w*)\.$', text[:start])
                owner_word = owner.group(1) if owner else ''
                if rest.startswith('(') and owner_word in \
                        self.maps['component']:
                    new = self.called(word, owner_word)
                elif rest.startswith('.') and owner_word in \
                        self.maps['component']:
                    new = self.prefix(word)
                elif rest.startswith('.') and owner_word in self.features:
                    new = self.features[owner_word].get(word, word)
            elif rest.startswith('('):
                new = self.called(word, component)
            elif rest.startswith('.'):
                new = self.prefix(word)
            if new != word:
                out.append(text[last:start])
                out.append(new)
                last = end
        out.append(text[last:])
        return ''.join(out)

    def prefix(self, word: str) -> str:
        """Translates a tag before a dot: a component, physics
        identifier, multiphysics coupling, material ('mat1.def.rho'),
        coordinate system or pair; marks one of b's tags without a
        partner that a uses for something else."""
        for kind in ('component', 'identifier', 'multiphysics', 'material',
                     'coordinate', 'pair', 'geometry', 'massprop'):
            if word in self.maps[kind]:
                return self.maps[kind][word]
        if word in self.b_words and word in self.a_words and \
                not any(word in names for names in self.names.values()):
            return f'<b only:{word}>'
        return word

    def called(self, word: str, component: str | None) -> str:
        for scope in (component, None):
            names = self.names.get(scope, {})
            if word in names:
                return names[word]
        return word


def words(described: dict) -> set[str]:
    """
    The tags a model writes before a dot in expressions: components,
    geometries, physics identifiers, couplings, materials, coordinate
    systems, pairs and mass properties names.
    """
    found: set[str] = set()
    for key in ('materials', 'coordinate_systems'):
        found.update(n['tag'] for n in described.get(key, []) if 'tag' in n)
    for component in described.get('components', []):
        found.add(component['tag'])
        for key in ('geometries', 'multiphysics', 'pairs'):
            found.update(n['tag'] for n in component.get(key, [])
                         if 'tag' in n)
        found.update(p['identifier'] for p in component.get('physics', [])
                     if p.get('identifier'))
        for node in component.get('mass_properties', []):
            found.update(node.get('names') or [])
    return found


##############
# Comparison #
##############

class _Context:
    """Where two paired nodes are: their components (a's and b's tags),
    whether in a mesh (and the geometries' length scales), whether the
    component's geometry differs or the pair is a guess by order, and the
    bundle of nodes that override each other's entities (physics of an
    interface or a component, materials of a component, pairs)."""

    def __init__(self, component=None, component_b=None, mesh=False,
                 scales=(1.0, 1.0), differs=False, by_order=False,
                 bundle=None):
        self.component, self.component_b = component, component_b
        self.mesh, self.scales = mesh, scales
        self.differs, self.by_order = differs, by_order
        self.bundle = bundle

    def ordered(self) -> _Context:
        return _Context(self.component, self.component_b, self.mesh,
                        self.scales, self.differs, True, self.bundle)

    def bundled(self, bundle) -> _Context:
        return _Context(self.component, self.component_b, self.mesh,
                        self.scales, self.differs, self.by_order, bundle)


class _Comparison:
    """Pairs the nodes of two descriptions and lists their differences."""

    def __init__(self, a: dict, b: dict, tolerance: float):
        self.a, self.b = a, b
        self.tolerance = tolerance
        self.items: list[dict] = []
        self.translator = Translator(words(a), words(b))
        self.parameters = frozenset(a.get('parameters', {})) | \
            frozenset(b.get('parameters', {}))
        self.solutions = set(a.get('solutions', [])) | \
            set(b.get('solutions', []))
        # geometry pairs by a's tag, b's geometry tag to a's
        self.geometries: dict[str, GeometryPair] = {}
        # a's components whose geometry differs, with its geometry item
        self.differing: dict[str | None, dict] = {}
        # components only one model has (b's as ('b only', tag)), with
        # their items
        self.lone: dict[Any, dict] = {}
        self.pending: list[tuple] = []
        self.overlaps: dict[tuple[int, int], tuple] = {}
        self.unchecked_paths: set[tuple] = set()
        self.count = 0
        self.components = ({c['tag'] for c in a.get('components', [])},
                           {c['tag'] for c in b.get('components', [])})
        # each side's components by tag; tree positions of nodes by path
        self.component_nodes = {
            (side, c['tag']): c for side, described in (('a', a), ('b', b))
            for c in described.get('components', [])}
        self.ranks: dict[str, dict[str, int]] = {
            side: _tree_ranks(described)
            for side, described in (('a', a), ('b', b))}

    def run(self) -> list[dict]:
        self.compare_parameters()
        pairs = self.pair_components()
        self.pair_lists(('functions', 'couplings', 'probes'), named=True)
        self.pair_lists(('coordinate_systems',), kind='coordinate')
        self.pair_materials(pairs)
        for ca, cb, context in pairs:
            self.pair_component(ca, cb, context)
        self.pair_lists(('definitions',))
        self.compare_variables()
        self.pair_studies()
        for compare, args in self.pending:
            compare(*args)
        self.fold_children()
        self.link()
        self.add_notes()
        self.fold()
        self.items.sort(key=_section)
        return self.items

    # Items

    def item(self, kind: str, na, nb, message: str, a=None, b=None,
             context: _Context | None = None, fold: bool = False,
             section: int | None = None, **extra) -> dict:
        entry = {'kind': kind,
                 'path': {'a': _path(na), 'b': _path(nb)},
                 'label': {'a': _label(na), 'b': _label(nb)},
                 'message': message, 'a': a, 'b': b, **extra}
        if section is not None:
            entry['_part'] = section
        if context is not None:
            # a node only one model has was not paired at all
            if context.by_order and not kind.startswith('only_in'):
                entry['matched_by_order'] = True
                entry['message'] += ' (paired by order)'
            if context.mesh:
                entry['_mesh'] = True
            entry['_component'] = context.component
            entry['_fold'] = fold
            entry['_bundle'] = context.bundle
        entry['_order'] = self.count
        self.count += 1
        self.items.append(entry)
        return entry

    def unchecked(self, side: str, node, message: str,
                  context: _Context | None = None):
        key = (side, _path(node), message)
        if key in self.unchecked_paths:
            return
        self.unchecked_paths.add(key)
        na, nb = (node, None) if side == 'a' else (None, node)
        self.item('unchecked', na, nb, message, context=context)

    def only(self, side: str, node: dict, context: _Context,
             others: list[dict]):
        """Reports a node that only one model has."""
        kind = node.get('type')
        empty = _empty(node.get('selection'))
        na, nb = (node, None) if side == 'a' else (None, node)
        fold = any(other.get('type') == kind for other in others)
        message = f'{self.head(na, nb)}: only in {side}'
        if empty:
            message += ' (selects nothing)'
        extra = {'empty': True} if empty else {}
        entry = self.item(f'only_in_{side}', na, nb, message,
                          a=node if side == 'a' else None,
                          b=node if side == 'b' else None, context=context,
                          fold=fold, **extra)
        if context.bundle is not None and node.get('active') is not False:
            entry['_area'] = self.area(side, node)

    def head(self, na, nb) -> str:
        """
        Names a pair of nodes: a's label (b's if a has none) and each
        side's path without its component, e.g. 'Temperature 1 (a
        ht/temp1, b ht/temp3)'.
        """
        node = na if na is not None else nb
        label = (node or {}).get('label') or (node or {}).get('tag') or ''
        tags = []
        for side, n, components in (('a', na, self.components[0]),
                                    ('b', nb, self.components[1])):
            if not isinstance(n, dict):
                continue
            path = n.get('path') or n.get('tag')
            if not path:
                continue
            first, _, rest = path.partition('/')
            short = rest if rest and first in components else path
            tags.append(f'{side} {short}')
        return f'{label} ({", ".join(tags)})' if tags else label

    def side_head(self, side: str, node) -> str:
        return self.head(node, None) if side == 'a' else \
            self.head(None, node)

    # Parameters

    def compare_parameters(self):
        pa, pb = self.a.get('parameters', {}), self.b.get('parameters', {})
        for name in list(pa) + [n for n in pb if n not in pa]:
            node = {'path': f'parameters/{name}', 'label': name}
            if name not in pb or name not in pa:
                side = 'a' if name in pa else 'b'
                entry = (pa if side == 'a' else pb)[name]
                na, nb = (node, None) if side == 'a' else (None, node)
                self.item(f'only_in_{side}', na, nb,
                          f'parameter {name} only in {side}',
                          a=entry if side == 'a' else None,
                          b=entry if side == 'b' else None, section=0)
                continue
            ea, eb = pa[name], pb[name]
            found = same_value(ea['expression'], eb['expression'],
                               _parameter_si(ea), _parameter_si(eb),
                               self.parameters)
            if found:
                self.item('parameter' if found == 'property' else found,
                          node, node,
                          f'parameter {name} is {shown(ea["expression"])} '
                          f'in a, {shown(eb["expression"])} in b',
                          a=ea['expression'], b=eb['expression'], section=0)

    # Components and geometries

    def pair_components(self) -> list[tuple]:
        ca_list = self.a.get('components', [])
        cb_list = self.b.get('components', [])
        scored = []
        for i, ca in enumerate(ca_list):
            for j, cb in enumerate(cb_list):
                score = -1.0
                ga, gb = ca['geometries'], cb['geometries']
                if ga and gb:
                    score = GeometryPair(ga[0], gb[0], self.tolerance).score()
                scored.append((-score, ca['tag'] != cb['tag'], abs(i - j),
                               i, j))
        pairs = []
        used_a: set[int] = set()
        used_b: set[int] = set()
        for _, _, _, i, j in sorted(scored):
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
        found = []
        for i, j in sorted(pairs):
            ca, cb = ca_list[i], cb_list[j]
            self.translator.add('component', cb['tag'], ca['tag'])
            differs = self.pair_geometries(ca, cb)
            found.append((ca, cb, _Context(ca['tag'], cb['tag'],
                                           differs=differs)))
        for i, ca in enumerate(ca_list):
            if i not in used_a:
                node = {'path': ca['tag'], 'label': ca.get('label')}
                self.lone[ca['tag']] = self.item(
                    'only_in_a', node, None,
                    f'{self.head(node, None)}: component only in a', a=ca,
                    section=1)
        for j, cb in enumerate(cb_list):
            if j not in used_b:
                node = {'path': cb['tag'], 'label': cb.get('label')}
                self.lone[('b only', cb['tag'])] = self.item(
                    'only_in_b', None, node,
                    f'{self.head(None, node)}: component only in b', b=cb,
                    section=1)
        return found

    def pair_geometries(self, ca: dict, cb: dict) -> bool:
        """Pairs the geometries of two components and reports how they
        differ; returns whether the shape differs."""
        # geometries have no path of their own: name them by component
        ga_list = [{**g, 'path': f"{ca['tag']}/{g['tag']}"}
                   for g in ca['geometries']]
        gb_list = [{**g, 'path': f"{cb['tag']}/{g['tag']}"}
                   for g in cb['geometries']]
        scored = sorted(
            (-GeometryPair(ga, gb, self.tolerance).score(),
             ga['tag'] != gb['tag'], i, j)
            for i, ga in enumerate(ga_list) for j, gb in enumerate(gb_list))
        used_a: set[int] = set()
        used_b: set[int] = set()
        differs = False
        for _, _, i, j in scored:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            ga, gb = ga_list[i], gb_list[j]
            self.translator.add('geometry', gb['tag'], ga['tag'])
            pair = GeometryPair(ga, gb, self.tolerance)
            self.geometries[ga['tag']] = pair
            context = _Context(ca['tag'], cb['tag'])
            for side, geometry in (('a', ga), ('b', gb)):
                if geometry.get('length_scale') is None:
                    self.unchecked(side, geometry,
                                   f'{self.side_head(side, geometry)}: '
                                   'length unit without a known size',
                                   context)
            found = pair.differences()
            if found:
                differs = pair.differs = True
                entry = self.item(
                    'geometry', ga, gb, f'{self.head(ga, gb)}: geometry '
                    f'differs: {"; ".join(found)}',
                    a=_shape(ga), b=_shape(gb), context=context)
                entry['consequences'] = []
                self.differing.setdefault(ca['tag'], entry)
            if ga.get('finalize') != gb.get('finalize'):
                self.item('geometry', ga, gb,
                          f'{self.head(ga, gb)}: a forms {_article(ga)}, '
                          f'b {_article(gb)}', a=ga.get('finalize'),
                          b=gb.get('finalize'), context=context)
                for level, (left_a, left_b) in pair.leftovers().items():
                    self.item('unchecked', ga, gb,
                              f'{self.head(ga, gb)}: entities that differ '
                              f'with union against assembly: '
                              f'{pair.rows_text(level, left_a, left_b)}',
                              context=context)
        for side, geometries, used in (('a', ga_list, used_a),
                                       ('b', gb_list, used_b)):
            for i, geometry in enumerate(geometries):
                if i not in used:
                    na, nb = (geometry, None) if side == 'a' else \
                        (None, geometry)
                    self.item(f'only_in_{side}', na, nb,
                              f'{self.head(na, nb)}: geometry only in {side}',
                              a=na, b=nb, section=1,
                              context=_Context(ca['tag'], cb['tag']))
                    differs = True
        return differs

    def selection(self, sa, sb) -> dict:
        """Compares two selections, on the geometries they are on."""
        if sa is None or sb is None:
            return _same() if sa is sb else _different(
                f'selection {_level(sa)} in a, {_level(sb)} in b')
        la, lb = sa.get('level'), sb.get('level')
        if la != lb:
            return _different(f'level {la} in a, {lb} in b')
        if la == 'several' or sa.get('entities') == 'unknown' or \
                sb.get('entities') == 'unknown':
            return _unknown()
        if la in LEVELS_ONLY or 'entities' not in sa:
            return _same()
        gtag_a, gtag_b = sa.get('geometry'), sb.get('geometry')
        pair = self.geometries.get(gtag_a)
        if pair is None or \
                self.translator.maps['geometry'].get(gtag_b) != gtag_a:
            found = _different('on other geometries')
            found['other'] = True
            return found
        found = pair.selection(sa, sb)
        if _exterior(sa) != _exterior(sb):
            found = _with_exterior(found, 'a' if _exterior(sa) else 'b')
        return found

    def overlap(self, na: dict, nb: dict) -> float:
        key = (id(na), id(nb))
        if key not in self.overlaps:
            # the nodes are kept with the value: their ids stay theirs
            self.overlaps[key] = (na, nb, self.measure_overlap(na, nb))
        return self.overlaps[key][2]

    def measure_overlap(self, na: dict, nb: dict) -> float:
        if 'source' in na or 'source' in nb:      # an identity pair
            straight = [self.selection(na.get(s), nb.get(s))['overlap']
                        for s in ('source', 'destination')]
            swapped = [self.selection(na.get(s), nb.get(t))['overlap']
                       for s, t in (('source', 'destination'),
                                    ('destination', 'source'))]
            return max(sum(straight), sum(swapped)) / 2
        return self.selection(na.get('selection'),
                              nb.get('selection'))['overlap']

    # Pairing

    def content(self, na: dict, nb: dict, context: _Context) -> float:
        """How alike two nodes' settings are (0 to 2), for ties."""
        key = 'settings' if 'settings' in na else 'properties'
        pa = na.get(key) or {}
        pb = nb.get(key) or {}
        if 'groups' in na or 'groups' in nb:
            pa, pb = _flat_groups(na), _flat_groups(nb)
        if not pa and not pb:
            score = 2.0
        else:
            keys_a, keys_b = set(pa), set(pb)
            common = keys_a & keys_b
            score = len(common) / len(keys_a | keys_b)
            if common:
                score += sum(
                    same_value(pa[k], self.translator.value(
                        pb[k], context.component_b)) is None
                    for k in common) / len(common)
        if 'identifier' in na:
            # interfaces of one type: their features tell them apart
            fa, fb = na.get('features', []), nb.get('features', [])
            types_a = sorted(f.get('type', '') for f in fa)
            types_b = sorted(f.get('type', '') for f in fb)
            score += _jaccard(types_a, types_b)
            overlaps = [max([self.overlap(x, y) for y in fb
                             if y.get('type') == x.get('type')] + [0.0])
                        for x in fa]
            score += sum(overlaps) / len(overlaps) if overlaps else 1.0
        return score

    def pair(self, list_a: list[dict], list_b: list[dict],
             context: _Context, named: bool = False,
             kind=None) -> list[tuple]:
        """
        Pairs two lists of nodes: by name first if `named`, then within
        each type by how much their selections overlap and how alike
        their settings are (then the same tag and label first), a node
        that selects nothing with one of the same tag and label, the last
        one of a type left on each side with each other, and in a
        component whose geometry differs the rest by order. Reports the
        nodes left over; returns the pairs with their contexts.
        """
        def type_of(node):
            return kind(node) if kind else node.get('type', 'variables')

        pairs: list[tuple] = []
        left_a = list(range(len(list_a)))
        left_b = list(range(len(list_b)))
        if named:
            for i in list(left_a):
                names_a = set(list_a[i].get('names') or [])
                for j in left_b:
                    nb = list_b[j]
                    if names_a & set(nb.get('names') or []) and \
                            type_of(nb) == type_of(list_a[i]):
                        pairs.append((i, j, context))
                        left_a.remove(i)
                        left_b.remove(j)
                        break
        for group in dict.fromkeys(type_of(list_a[i]) for i in left_a):
            group_a = [i for i in left_a if type_of(list_a[i]) == group]
            group_b = [j for j in left_b if type_of(list_b[j]) == group]
            scored = []
            for i in group_a:
                for j in group_b:
                    overlap = self.overlap(list_a[i], list_b[j])
                    if overlap > 0:
                        scored.append((-overlap, -self.content(
                            list_a[i], list_b[j], context),
                            not _same_name(list_a[i], list_b[j]), i, j))
            for _, _, _, i, j in sorted(scored):
                if i in group_a and j in group_b:
                    pairs.append((i, j, context))
                    group_a.remove(i)
                    group_b.remove(j)
                    left_a.remove(i)
                    left_b.remove(j)
            # one side selects nothing yet: the same tag and label tell
            for i in list(group_a):
                for j in group_b:
                    na, nb = list_a[i], list_b[j]
                    if _same_name(na, nb) and (
                            _nothing(na.get('selection')) or
                            _nothing(nb.get('selection'))):
                        pairs.append((i, j, context))
                        group_a.remove(i)
                        group_b.remove(j)
                        left_a.remove(i)
                        left_b.remove(j)
                        break
            guess = context.ordered() if context.differs else context
            if len(group_a) == 1 and len(group_b) == 1:
                pairs.append((group_a[0], group_b[0], guess))
                left_a.remove(group_a[0])
                left_b.remove(group_b[0])
            elif context.differs and len(group_a) == len(group_b):
                for i, j in zip(group_a, group_b):
                    pairs.append((i, j, guess))
                    left_a.remove(i)
                    left_b.remove(j)
        for i in left_a:
            self.only('a', list_a[i], context, list_b)
        for j in left_b:
            self.only('b', list_b[j], context, list_a)
        return [(list_a[i], list_b[j], c) for i, j, c in sorted(pairs)]

    def pair_lists(self, keys, named: bool = False, kind: str | None = None):
        """Pairs the nodes of model-wide lists (functions, ...), within
        each component, and records their names and tags."""
        for key in keys:
            list_a, list_b = self.a.get(key, []), self.b.get(key, [])
            scopes = dict.fromkeys(
                [n.get('component') for n in list_a] +
                [self.scope_b(n.get('component')) for n in list_b])
            for scope in scopes:
                in_a = [n for n in list_a if n.get('component') == scope]
                in_b = [n for n in list_b
                        if self.scope_b(n.get('component')) == scope]
                scope_b = in_b[0].get('component') if in_b else None
                context = self.context_of(scope, scope_b)
                for na, nb, ctx in self.pair(in_a, in_b, context, named):
                    names = self.translator.names.setdefault(
                        nb.get('component'), {})
                    for name_b, name_a in zip(nb.get('names') or [nb['tag']],
                                              na.get('names') or [na['tag']]):
                        names[name_b] = name_a
                    if kind:
                        self.translator.add(kind, nb['tag'], na['tag'])
                    self.later(self.compare_node, na, nb, ctx)

    def scope_b(self, component):
        """Where a node of model b is, as a's component: a component
        without a partner in a is a scope of its own."""
        if component is None:
            return None
        return self.translator.maps['component'].get(component,
                                                     ('b only', component))

    def context_of(self, component, component_b) -> _Context:
        return _Context(component, component_b,
                        differs=component in self.differing)

    def later(self, compare, *args):
        self.pending.append((compare, args))

    # Materials

    def pair_materials(self, components: list[tuple]):
        materials_a = self.a.get('materials', [])
        materials_b = self.b.get('materials', [])
        used_a: set[str] = set()
        used_b: set[str] = set()
        for ca, cb, context in components:
            in_a = [m for m in materials_a if m.get('component') == ca['tag']]
            in_b = [m for m in materials_b if m.get('component') == cb['tag']]
            bundled = context.bundled(('material', ca['tag']))
            for na, nb, ctx in self.pair(in_a, in_b, bundled,
                                         kind=_material_kind):
                self.translator.add('material', nb['tag'], na['tag'])
                for node, used in ((na, used_a), (nb, used_b)):
                    if node.get('type') == 'Link':
                        used.add(str(node.get('properties', {}).get(
                            'link', '')))
                self.later(self.compare_material, na, nb, ctx)
        global_a = [m for m in materials_a if m.get('component') is None
                    and m['tag'] not in used_a]
        global_b = [m for m in materials_b if m.get('component') is None
                    and m['tag'] not in used_b]
        for na, nb, ctx in self.pair(global_a, global_b, _Context()):
            self.translator.add('material', nb['tag'], na['tag'])
            self.later(self.compare_material, na, nb, ctx)

    def groups(self, material: dict, side: str) -> dict:
        """A material's property groups; a link's are its material's."""
        if material.get('type') == 'Link':
            link = str(material.get('properties', {}).get('link', ''))
            for other in (self.a if side == 'a' else self.b).get(
                    'materials', []):
                if other['tag'] == link and other.get('component') is None:
                    return other.get('groups', {})
        return material.get('groups', {})

    def compare_material(self, na: dict, nb: dict, context: _Context):
        if na.get('type') == nb.get('type'):
            self.compare_values(na, nb, context, skip=frozenset({'link'}),
                                info=frozenset(MATERIAL_INFO))
        self.compare_active(na, nb, context)
        self.compare_selection(na, nb, context)
        self.compare_applied(na, nb, context)
        groups_a, groups_b = self.groups(na, 'a'), self.groups(nb, 'b')
        for gtag in list(groups_a) + [g for g in groups_b
                                      if g not in groups_a]:
            self.compare_group(na, nb, gtag, groups_a.get(gtag) or {},
                               groups_b.get(gtag) or {}, context)
        self.pair_children(na, nb, context, self.compare_material)

    def compare_group(self, na, nb, gtag: str, ga: dict, gb: dict,
                      context: _Context):
        pa = dict(ga.get('properties', {}))
        pb = dict(gb.get('properties', {}))
        for ftag, function in ga.get('functions', {}).items():
            pa[f'functions/{ftag}'] = function
        for ftag, function in gb.get('functions', {}).items():
            pb[f'functions/{ftag}'] = function
        si_a, si_b = ga.get('si', {}), gb.get('si', {})
        one_sided: dict[str, dict] = {'a': {}, 'b': {}}
        info: dict[str, dict] = {'a': {}, 'b': {}}
        head = f'{self.head(na, nb)} {gtag}'
        for key in list(pa) + [k for k in pb if k not in pa]:
            if key in pa and key in pb:
                value_b = self.translator.value(pb[key], context.component_b)
                if isinstance(pa[key], dict) and isinstance(value_b, dict):
                    self.compare_function(na, nb, head, key, pa[key],
                                          pb[key], value_b, gtag, context)
                    continue
                found = same_value(pa[key], value_b, si_a.get(key),
                                   si_b.get(key), self.parameters)
                if found:
                    ending = in_a_tags(pa[key], pb[key], value_b)
                    self.item(found, na, nb,
                              f'{head}: {key} is {shown(pa[key])} in a, '
                              f'{shown(pb[key])} in b{ending}', a=pa[key],
                              b=pb[key], context=context, group=gtag,
                              name=key)
                continue
            side = 'a' if key in pa else 'b'
            target = info if key in MATERIAL_INFO or \
                key.endswith('_symmetry') else one_sided
            target[side][key] = (pa if side == 'a' else pb)[key]
        for keys, extra in ((one_sided, {}),
                            (info, {'material_info': True})):
            if keys['a'] or keys['b']:
                parts = [f'{", ".join(keys[s])} only in {s}'
                         for s in 'ab' if keys[s]]
                self.item('property', na, nb, f'{head}: {"; ".join(parts)}',
                          a=keys['a'] or None, b=keys['b'] or None,
                          context=context, group=gtag, **extra)

    def compare_function(self, na, nb, head: str, key: str, fa: dict,
                         fb: dict, translated: dict, gtag: str,
                         context: _Context):
        """Compares a material's function (a dict of settings), naming
        only the settings that differ."""
        names = [k for k in list(fa) + [k for k in translated
                                         if k not in fa]
                 if same_value(fa.get(k), translated.get(k),
                               parameters=self.parameters)]
        if not names:
            return
        parts = [f'{k} is {shown(fa.get(k), 40)} in a, '
                 f'{shown(fb.get(k), 40)} in b'
                 f'{in_a_tags(fa.get(k), fb.get(k), translated.get(k), 40)}'
                 for k in names]
        extra = {'material_info': True} if set(names) <= set(MATERIAL_INFO) \
            else {}
        self.item('property', na, nb, f'{head}: {key}: {"; ".join(parts)}',
                  a=fa, b=fb, context=context, group=gtag, name=key, **extra)

    # Components

    def pair_component(self, ca: dict, cb: dict, context: _Context):
        component = context.component
        for na, nb, ctx in self.pair(ca.get('pairs', []),
                                     cb.get('pairs', []),
                                     context.bundled(('pair', component))):
            self.translator.add('pair', nb['tag'], na['tag'])
            self.later(self.compare_pair, na, nb, ctx)
        # couplings override the features of the interfaces they couple,
        # and do not say which: then all of the component's physics are
        # one bundle; else each interface is one
        coupled = bool(ca.get('multiphysics') or cb.get('multiphysics'))
        bundle = ('physics', component) if coupled else None
        interfaces = self.pair(ca.get('physics', []), cb.get('physics', []),
                               context.bundled(bundle))
        for na, nb, _ in interfaces:
            self.translator.add('physics', nb['tag'], na['tag'])
            self.translator.add('identifier', nb.get('identifier'),
                                na.get('identifier'))
        for na, nb, ctx in interfaces:
            self.pair_interface(na, nb, ctx.bundled(
                bundle or ('physics', component, na['tag'])))
        for na, nb, ctx in self.pair(ca.get('multiphysics', []),
                                     cb.get('multiphysics', []),
                                     context.bundled(bundle)):
            self.translator.add('multiphysics', nb['tag'], na['tag'])
            self.later(self.compare_node, na, nb, ctx)
        self.pair_meshes(ca, cb, context)
        # mass properties are called by name: 'mass1.mass'
        for na, nb, ctx in self.pair(ca.get('mass_properties', []),
                                     cb.get('mass_properties', []), context,
                                     named=True):
            for name_b, name_a in zip(nb.get('names') or [],
                                      na.get('names') or []):
                self.translator.add('massprop', name_b, name_a)
            self.later(self.compare_node, na, nb, ctx)

    def pair_interface(self, na: dict, nb: dict, context: _Context):
        rows_a = [f for f in na.get('features', []) if 'rows' in f]
        rows_b = [f for f in nb.get('features', []) if 'rows' in f]
        self.pair_features(
            [f for f in na.get('features', []) if 'rows' not in f],
            [f for f in nb.get('features', []) if 'rows' not in f],
            context, nb)
        groups = self.pair_rows(rows_a, rows_b, nb, context)
        self.later(self.compare_interface, na, nb, context)
        self.later(self.compare_row_features, groups, context)
        self.later(self.compare_rows, na, nb, rows_a, rows_b, context)

    def pair_rows(self, rows_a: list, rows_b: list, owner_b: dict,
                  context: _Context) -> list[tuple[list, list, list]]:
        """
        Pairs features that list equations by row (global equations) of
        one type by the equations they share, several to several, records
        b's tags under a's (the first partner) and pairs their subfeatures
        in each group of features linked that way. Returns the groups as
        (a's features, b's features, pairs).
        """
        def names(feature) -> set:
            return set(feature['rows'].get('name') or [])

        pairs = [(i, j) for i, fa in enumerate(rows_a)
                 for j, fb in enumerate(rows_b)
                 if fa.get('type') == fb.get('type')
                 and names(fa) & names(fb)]
        for key in (owner_b.get('identifier'), owner_b.get('tag')):
            if key:
                tags = self.translator.features.setdefault(key, {})
                for i, j in pairs:
                    tags.setdefault(rows_b[j]['tag'], rows_a[i]['tag'])
        # the groups of features linked by shared equations
        group: dict[tuple, int] = {}
        for i, j in pairs:
            ga, gb = group.get(('a', i)), group.get(('b', j))
            if ga is None and gb is None:
                group[('a', i)] = group[('b', j)] = len(group)
            elif ga is not None and gb is None:
                group[('b', j)] = ga
            elif ga is None and gb is not None:
                group[('a', i)] = gb
            elif ga is not None and ga != gb:
                for key, value in group.items():
                    if value == gb:
                        group[key] = ga
        found = []
        for number in dict.fromkeys(group.values()):
            in_a = [f for i, f in enumerate(rows_a)
                    if group.get(('a', i)) == number]
            in_b = [f for j, f in enumerate(rows_b)
                    if group.get(('b', j)) == number]
            linked = [(rows_a[i], rows_b[j]) for i, j in pairs
                      if group[('a', i)] == number]
            found.append((in_a, in_b, linked))
            # now, so that their tags are known before values are compared
            self.pair_features(
                [sub for f in in_a for sub in f.get('features', [])],
                [sub for f in in_b for sub in f.get('features', [])],
                context, owner_b)
        return found

    def compare_row_features(self, groups: list, context: _Context):
        """Compares the features of each group from `pair_rows`: one
        item for each disabled one, labels where one is paired with one.
        Their selection is global, the same for all."""
        disabled = set()
        for in_a, in_b, linked in groups:
            for fa, fb in linked:
                if fa.get('active', True) == fb.get('active', True):
                    continue
                off = ('a', fa['path']) if not fa.get('active', True) \
                    else ('b', fb['path'])
                if off not in disabled:
                    disabled.add(off)
                    self.compare_active(fa, fb, context)
            if len(in_a) == 1 and len(in_b) == 1:
                self.compare_label(in_a[0], in_b[0], context)

    def pair_features(self, list_a, list_b, context, owner_b,
                      ordered=None) -> list:
        """Pairs features (also of features) and records their tags under
        the interface's identifier and tag in model b. With the nodes
        that hold the lists as `ordered`, also compares their order."""
        pairs = self.pair(list_a, list_b, context)
        for key in (owner_b.get('identifier'), owner_b.get('tag')):
            if key:
                tags = self.translator.features.setdefault(key, {})
                for fa, fb, _ in pairs:
                    tags[fb['tag']] = fa['tag']
        if ordered is not None:
            owner_a, owner_b = ordered
            self.compare_order(owner_a, owner_b, list_b, pairs, context)
        for fa, fb, ctx in pairs:
            self.later(self.compare_node, fa, fb, ctx, False)
            self.pair_features(fa.get('features', []),
                               fb.get('features', []), ctx, owner_b,
                               None if ordered is None else (fa, fb))
        return pairs

    def compare_order(self, owner_a: dict, owner_b: dict, list_b: list,
                      pairs: list, context: _Context):
        """
        Reports paired nodes in another order (mesh operations, study
        steps): the ones outside the longest run in the same order (the
        first such run in a's order) moved.
        """
        positions = [next(k for k, node in enumerate(list_b) if node is fb)
                     for _, fb, _ in pairs]
        kept = _longest_run(positions)
        if len(kept) == len(pairs):
            return
        moved = [self.head(fa, fb) for k, (fa, fb, _) in enumerate(pairs)
                 if k not in kept]
        in_b = [pairs[k][1]['tag'] for k in
                sorted(range(len(pairs)), key=positions.__getitem__)]
        self.item('order', owner_a, owner_b,
                  f'{self.head(owner_a, owner_b)}: order differs: '
                  f'{", ".join(moved)} moved; b: {", ".join(in_b)}',
                  a=[fa['tag'] for fa, _, _ in pairs], b=in_b,
                  context=context)

    def pair_meshes(self, ca: dict, cb: dict, context: _Context):
        geometries = self.translator.maps['geometry']
        meshes_b = list(cb.get('meshes', []))
        pairs = []
        only_a = []
        meshes_a = ca.get('meshes', [])

        def same_geometry(ma: dict, mb: dict) -> bool:
            return geometries.get(mb.get('geometry') or '') == \
                ma.get('geometry')

        # on one geometry, the same tag first, then in order
        for ma in meshes_a:
            match = next((mb for mb in meshes_b if same_geometry(ma, mb)
                          and mb.get('tag') == ma.get('tag')), None)
            if match is not None:
                meshes_b.remove(match)
                pairs.append((ma, match))
        paired = [ma for ma, _ in pairs]
        for ma in meshes_a:
            if any(ma is other for other in paired):
                continue
            match = next((mb for mb in meshes_b if same_geometry(ma, mb)),
                         None)
            if match is None:
                only_a.append(ma)
            else:
                meshes_b.remove(match)
                pairs.append((ma, match))
        pairs.sort(key=lambda pair: next(
            i for i, ma in enumerate(meshes_a) if ma is pair[0]))
        for ma, mb in pairs:
            self.translator.add('mesh', mb.get('tag'), ma.get('tag'))
        mesh_context = _Context(context.component, context.component_b,
                                mesh=True, differs=context.differs)
        for ma in only_a:
            self.only('a', ma, mesh_context, [])
        for mb in meshes_b:
            self.only('b', mb, mesh_context, [])
        for ma, mb in pairs:
            pair = self.geometries.get(ma.get('geometry'))
            ctx = _Context(context.component, context.component_b, mesh=True,
                           scales=pair.scales if pair else (1.0, 1.0),
                           differs=context.differs)
            if ma.get('automatic') and mb.get('automatic') and \
                    ma.get('size_level') != mb.get('size_level'):
                self.item('property', ma, mb,
                          f'{self.head(ma, mb)}: size level is '
                          f'{ma.get("size_level")} in a, '
                          f'{mb.get("size_level")} in b',
                          a=ma.get('size_level'), b=mb.get('size_level'),
                          context=ctx)
            self.pair_features(ma.get('features', []),
                               mb.get('features', []), ctx, {}, (ma, mb))

    # Studies

    def pair_studies(self):
        studies = self.pair(self.a.get('studies', []),
                            self.b.get('studies', []), _Context(),
                            kind=lambda study: 'study')
        for sa, sb, _ in studies:
            self.translator.add('study', sb['tag'], sa['tag'])
            sequence_a = (sa.get('solver') or {}).get('sequence')
            sequence_b = (sb.get('solver') or {}).get('sequence')
            self.translator.add('sequence', sequence_b, sequence_a)
        for sa, sb, ctx in studies:
            steps = self.pair(sa.get('steps', []), sb.get('steps', []), ctx)
            self.compare_order(sa, sb, sb.get('steps', []), steps, ctx)
            tags = self.translator.steps.setdefault(sb['tag'], {})
            for fa, fb, _ in steps:
                tags[fb['tag']] = fa['tag']
                self.later(self.compare_node, fa, fb, ctx)
            self.later(self.compare_solver, sa, sb)

    def change_key(self, change: dict, side: str) -> str:
        """
        A solver change as a comparable key: model b's tags translated to
        a's, and solution tags as one placeholder (a store-solution node
        names a new solution each time).
        """
        def plain(value):
            if isinstance(value, list):
                return [plain(item) for item in value]
            if isinstance(value, str) and value in self.solutions:
                return '<solution>'
            return value

        def mapped(value):
            if side == 'b':
                value = self.translator.value(value)
            return plain(value)

        return json.dumps([mapped(change.get('path')), change.get('type'),
                           change.get('change'),
                           {k: mapped(v[0]) for k, v in
                            (change.get('properties') or {}).items()}],
                          sort_keys=True)

    def compare_solver(self, sa: dict, sb: dict):
        solver_a, solver_b = sa.get('solver') or {}, sb.get('solver') or {}
        statuses = {solver_a.get('status'), solver_b.get('status')}
        if 'not_asked' in statuses:
            # one model's solver was read: say how to compare them; a
            # model without a sequence (COMSOL's own) needs no reading
            for side, solver, other, study in (
                    ('a', solver_a, solver_b, sa),
                    ('b', solver_b, solver_a, sb)):
                if solver.get('status') == 'not_asked' and \
                        other.get('status') in ('compared', 'not_compared'):
                    self.item('note', sa, sb, f'{self.head(sa, sb)}: '
                              f'solvers not compared; describe {side} with '
                              'solver=True')
                elif solver.get('status') == 'not_compared':
                    self.unchecked(side, study,
                                   f'{self.side_head(side, study)}: solver '
                                   f'not compared: {solver.get("reason")}')
            return
        if not statuses <= {'compared', 'automatic'}:
            for side, solver, study in (('a', solver_a, sa),
                                        ('b', solver_b, sb)):
                if solver.get('status') not in ('compared', 'automatic'):
                    self.unchecked(side, study,
                                   f'{self.side_head(side, study)}: solver '
                                   f'not compared: {solver.get("reason")}')
            return
        keys_a = {self.change_key(c, 'a'): c
                  for c in solver_a.get('changes', [])}
        keys_b = {self.change_key(c, 'b'): c
                  for c in solver_b.get('changes', [])}
        for keys, side in ((keys_a, 'a'), (keys_b, 'b')):
            other = keys_b if side == 'a' else keys_a
            for key, change in keys.items():
                if key not in other:
                    values = ', '.join(
                        f'{name} {shown(pair[0], 40)} (COMSOL: '
                        f'{shown(pair[1], 40)})' for name, pair in
                        (change.get('properties') or {}).items())
                    self.item('solver', sa, sb,
                              f'{self.head(sa, sb)}: solver {change["path"]} '
                              f'({change["type"]}) {change["change"]}'
                              f'{" " + values if values else ""} only '
                              f'in {side}',
                              a=change if side == 'a' else None,
                              b=change if side == 'b' else None)

    # Variables

    def compare_variables(self):
        rows: dict[tuple, dict[str, list]] = {}
        for side, described in (('a', self.a), ('b', self.b)):
            for node in described.get('variables', []):
                scope = node.get('component')
                if side == 'b':
                    scope = self.scope_b(scope)
                for name, expression in node.get('variables', {}).items():
                    rows.setdefault((scope, name), {'a': [], 'b': []})[
                        side].append((node, expression))
        for (scope, name), sides in rows.items():
            scope_b = next((node.get('component')
                            for node, _ in sides['b']), None)
            context = self.context_of(scope, scope_b)
            left_b = list(sides['b'])
            for node_a, expression_a in sides['a']:
                match = max(left_b, default=None, key=lambda row: self.overlap(
                    node_a, row[0]))
                if match is None:
                    self.item('only_in_a', node_a, None,
                              f'variable {name} ({node_a["tag"]}) only in a',
                              a=expression_a, context=context, name=name)
                    continue
                left_b.remove(match)
                node_b, expression_b = match
                self.compare_variable(name, node_a, expression_a, node_b,
                                      expression_b, context)
            for node_b, expression_b in left_b:
                self.item('only_in_b', None, node_b,
                          f'variable {name} ({node_b["tag"]}) only in b',
                          b=expression_b, context=context, name=name)

    def compare_variable(self, name, na, expression_a, nb, expression_b,
                         context: _Context):
        head = f'variable {name} (a {na["tag"]}, b {nb["tag"]})'
        translated = self.translator.value(expression_b, context.component_b)
        if same_value(expression_a, translated, None, None,
                      self.parameters):
            ending = in_a_tags(expression_a, expression_b, translated)
            self.item('variable', na, nb, f'{head}: {shown(expression_a)} '
                      f'in a, {shown(expression_b)} in b{ending}',
                      a=expression_a, b=expression_b, context=context,
                      name=name)
        if na.get('active', True) != nb.get('active', True):
            self.item('variable', na, nb, f'{head}: active in '
                      f'{"a" if na.get("active", True) else "b"} only',
                      a=na.get('active'), b=nb.get('active'),
                      context=context, name=name)
        found = self.selection(na.get('selection'), nb.get('selection'))
        if found['unknown']:
            self.unchecked('a', na, f'{head}: selection not compared',
                           context)
        elif not found['same']:
            self.selection_item('variable', na, nb, f'{head}: selection '
                                'differs: ', found, context,
                                a=na.get('selection'),
                                b=nb.get('selection'), name=name)

    # Comparing paired nodes

    def compare_node(self, na: dict, nb: dict, context: _Context,
                     children: bool = True):
        self.compare_active(na, nb, context)
        self.compare_label(na, nb, context)
        self.compare_values(na, nb, context)
        self.compare_selection(na, nb, context)
        self.compare_applied(na, nb, context)
        for name in list(na.get('selections') or {}) + [
                n for n in (nb.get('selections') or {})
                if n not in (na.get('selections') or {})]:
            self.compare_selection(na, nb, context, name)
        if children:
            self.pair_children(na, nb, context, self.compare_node)

    def pair_children(self, na, nb, context, compare):
        for fa, fb, ctx in self.pair(na.get('features', []),
                                     nb.get('features', []), context):
            compare(fa, fb, ctx)

    def compare_interface(self, na: dict, nb: dict, context: _Context):
        self.compare_active(na, nb, context)
        self.compare_label(na, nb, context)
        self.compare_values(na, nb, context, key='settings')
        self.compare_selection(na, nb, context)

    def compare_pair(self, na: dict, nb: dict, context: _Context):
        self.compare_active(na, nb, context)
        if na.get('type') != nb.get('type'):
            self.item('property', na, nb, f'{self.head(na, nb)}: type '
                      f'{na.get("type")} in a, {nb.get("type")} in b',
                      a=na.get('type'), b=nb.get('type'), context=context)
        straight = [self.selection(na.get(s), nb.get(s))
                    for s in ('source', 'destination')]
        for side, found in zip(('source', 'destination'), straight):
            if found['unknown']:
                self.unchecked('a', na, f'{self.head(na, nb)}: {side} not '
                               'compared (several levels or unmeasured '
                               'entities)', context)

        def same(found: dict) -> bool:
            # a selection that could not be compared is not the same
            return found['same'] and not found['unknown']

        if all(same(found) for found in straight):
            return
        swapped = [self.selection(na.get(s), nb.get(t))
                   for s, t in (('source', 'destination'),
                                ('destination', 'source'))]
        if all(same(found) for found in swapped):
            entry = self.item('selection', na, nb, f'{self.head(na, nb)}: '
                              'source and destination swapped',
                              context=context)
            entry['_area'] = self.area('a', na) | self.area('b', nb)
            return
        for side, found in zip(('source', 'destination'), straight):
            if not found['same'] and not found['unknown']:
                entry = self.selection_item(
                    'selection', na, nb, f'{self.head(na, nb)}: {side} '
                    'differs: ', found, context, a=na.get(side),
                    b=nb.get(side))
                entry['_area'] = _area_of(na.get(side), found)

    def compare_active(self, na, nb, context):
        if na.get('active', True) != nb.get('active', True):
            side = 'a' if na.get('active', True) else 'b'
            entry = self.item('active', na, nb, f'{self.head(na, nb)}: '
                              f'active in {side} only', a=na.get('active'),
                              b=nb.get('active'), context=context)
            entry['_on'] = side
            if context.bundle is not None:
                # what the enabled one covers (a disabled node overrides
                # nothing; its own entities read as if it were enabled)
                entry['_area'] = self.area(side, na if side == 'a' else nb)

    def compare_label(self, na, nb, context):
        if na.get('label') != nb.get('label'):
            self.item('label', na, nb, f'{self.head(na, nb)}: label '
                      f'{na.get("label")!r} in a, {nb.get("label")!r} in b',
                      a=na.get('label'), b=nb.get('label'), context=context)

    def compare_selection(self, na, nb, context, name: str | None = None):
        if name is None:
            sa, sb = na.get('selection'), nb.get('selection')
        else:
            sa = (na.get('selections') or {}).get(name)
            sb = (nb.get('selections') or {}).get(name)
        found = self.selection(sa, sb)
        what = 'selection' if name is None else f'selection {name}'
        if found['unknown']:
            self.unchecked('a', na, f'{self.head(na, nb)}: {what} not '
                           'compared (several levels or unmeasured '
                           'entities)', context)
            return
        if found['same']:
            return
        extra: dict = {}
        ending = ''
        applied_a = (sa or {}).get('applied', (sa or {}).get('entities'))
        applied_b = (sb or {}).get('applied', (sb or {}).get('entities'))
        if sa and sb and sa.get('level') == sb.get('level') and \
                (applied_a, applied_b) != (sa.get('entities'),
                                           sb.get('entities')) and \
                self.selection({**sa, 'entities': applied_a},
                               {**sb, 'entities': applied_b})['same']:
            extra['same_applied'] = True
            ending = '; applies nowhere in either' if \
                applied_a == [] and applied_b == [] and not (
                    _exterior(sa, True) or _exterior(sb, True)) else \
                '; applies to the same entities'
        entry = self.selection_item(
            'selection', na, nb, f'{self.head(na, nb)}: {what} differs: ',
            found, context, ending, a=sa, b=sb, **extra)
        if name is None:
            entry['_area'] = _area_of(sa, found)

    def compare_applied(self, na: dict, nb: dict, context: _Context):
        """
        Reports where two nodes of a bundle (physics, materials) apply
        differently in the entities both select: another node overrides
        them in one model only (linked to its cause later).
        """
        if context.bundle is None or context.bundle[0] == 'pair':
            return
        if na.get('active') is False or nb.get('active') is False:
            return      # its 'active' item says it
        sa, sb = na.get('selection'), nb.get('selection')
        if not isinstance(sa, dict) or not isinstance(sb, dict):
            return
        level = sa.get('level')
        if level != sb.get('level') or level in LEVELS_ONLY or \
                level == 'several':
            return
        gtag = str(sa.get('geometry'))
        pair = self.geometries.get(gtag)
        if pair is None or level not in pair.dims or \
                self.translator.maps['geometry'].get(
                    str(sb.get('geometry'))) != gtag:
            return
        found = pair.applied(sa, sb)
        if found is None:
            return
        entry = self.selection_item(
            'applied', na, nb, f'{self.head(na, nb)}: applies differently: ',
            found, context, a=sa, b=sb)
        cells = found['cells'] - {'rest'} if pair.differs else found['cells']
        entry['_area'] = {(gtag, level, cell) for cell in cells}
        entry['_selections'] = (sa, sb)

    def selection_item(self, kind: str, na, nb, head: str, found: dict,
                       context: _Context, ending: str = '',
                       **extra) -> dict:
        """
        Reports a selection that differs. Where the geometry differs, a
        difference only in entities without a counterpart follows from it
        (it goes under the geometry item); one in entities both geometries
        have stays, with those entities only and how many others differ
        (`elsewhere`).
        """
        shown_part, fold, tail = found, False, ''
        if found.get('other'):
            fold = context.differs
        elif 'elsewhere' in found:
            inside = found.get('inside')
            if inside is None:
                fold = True
            else:
                shown_part = inside
                if found['elsewhere']:
                    extra['elsewhere'] = found['elsewhere']
                    tail = (f' (+{found["elsewhere"]} entities without a '
                            'counterpart)')
        return self.item(kind, na, nb,
                         f'{head}{shown_part["text"]}{tail}{ending}',
                         context=context, fold=fold, **_numbers(shown_part),
                         **extra)

    def compare_values(self, na: dict, nb: dict, context: _Context,
                       key: str = 'properties', skip=frozenset(),
                       info=frozenset()):
        """Compares the settings of two nodes of one type: values only one
        side sets are taken from the other side's defaults. Differences in
        `info` are marked as library entries ('material_info')."""
        pa, pb = na.get(key) or {}, nb.get(key) or {}
        da, db = na.get('defaults') or {}, nb.get('defaults') or {}
        si_a, si_b = na.get('si') or {}, nb.get('si') or {}
        sd_a, sd_b = na.get('si_defaults') or {}, nb.get('si_defaults') or {}
        unused = set(na.get('unused') or []) | set(nb.get('unused') or [])
        if na.get('type') == 'MassProperties':
            skip = skip | {'name'}      # compared through the pairing
        blocked = []
        in_units: list[str] = []
        translate = self.translator.value
        component_b = context.component_b
        for name in list(pa) + [n for n in pb if n not in pa]:
            if name in unused or name in NAME_KEYS or name in skip:
                continue
            from_default = False
            # a mesh length from the other's defaults is in its unit
            scale_a, scale_b = context.scales
            converted = context.mesh and name in LENGTH_KEYS
            if name in pa:
                raw_a, value_a, s_a = pa[name], pa[name], si_a.get(name)
            elif name in db:
                s_a = sd_b.get(name)
                value_a = translate(db[name], component_b)
                if converted and _number(value_a):
                    value_a = float(f'{value_a * scale_b / scale_a:.12g}')
                raw_a = value_a
                from_default = True
            else:
                blocked.append(name)
                continue
            if name in pb:
                raw_b, s_b = pb[name], si_b.get(name)
                value_b = translate(pb[name], component_b)
            elif name in da:
                value_b, s_b = da[name], sd_a.get(name)
                if converted and _number(value_b):
                    value_b = float(f'{value_b * scale_a / scale_b:.12g}')
                raw_b = value_b
                from_default = True
            else:
                blocked.append(name)
                continue
            if isinstance(value_a, dict) or isinstance(value_b, dict):
                self.compare_map(na, nb, name, value_a, value_b,
                                 pb.get(name) if name in pb else None, da, db,
                                 context)
                continue
            found = self.compare_one(name, value_a, value_b, s_a, s_b,
                                     context)
            if found == 'unchecked':
                in_units.append(name)
            elif found:
                extra = {'from_default': True} if from_default else {}
                # a library material's own entry; a coordinate system only
                # when one side has the library's 'none'
                if name in info and (name != 'sys' or
                                     'none' in (raw_a, raw_b)):
                    extra['material_info'] = True
                # mesh lengths from the defaults follow the geometry's size
                fold = from_default and context.mesh and \
                    name in LENGTH_KEYS
                ending = in_a_tags(raw_a, raw_b, value_b) \
                    if name in pb else ''
                self.item(found, na, nb,
                          f'{self.head(na, nb)}: {name} is {shown(raw_a)} '
                          f'in a, {shown(raw_b)} in b{ending}', a=raw_a,
                          b=raw_b, context=context, name=name, fold=fold,
                          **extra)
        if in_units:
            self.unchecked('a', na, f'{self.head(na, nb)}: the length units '
                           f'differ, not compared: {", ".join(in_units)}',
                           context)
        if blocked:
            self.unchecked('a', na, f'{self.head(na, nb)}: defaults '
                           f'unknown, not compared: {", ".join(blocked)}',
                           context)
        if key == 'properties':
            self.used_only(na, nb, context)

    def used_only(self, na: dict, nb: dict, context: _Context):
        """
        Reports the values one side sets and uses where the other's
        settings leave them unused (sizes with `custom` on in one mesh
        node and off in the other).
        """
        if na.get('all_properties') or nb.get('all_properties'):
            return
        unused_a = set(na.get('unused') or [])
        unused_b = set(nb.get('unused') or [])
        found = {}
        for side, node, others in (('a', na, unused_b - unused_a),
                                   ('b', nb, unused_a - unused_b)):
            found[side] = {
                key: value for key, value in
                (node.get('properties') or {}).items()
                if key in others and key not in NAME_KEYS
                and not key.endswith('active')}
        if not found['a'] and not found['b']:
            return
        parts = ['used only in ' + side + ': ' + ', '.join(
            f'{key} {shown(value)}' for key, value in found[side].items())
            for side in 'ab' if found[side]]
        self.item('property', na, nb,
                  f'{self.head(na, nb)}: {"; ".join(parts)}',
                  a=found['a'] or None, b=found['b'] or None,
                  context=context, used_only=True)

    def compare_one(self, name: str, value_a, value_b, si_a, si_b,
                    context: _Context) -> str | None:
        if context.mesh and _number(value_a) and _number(value_b):
            scale_a, scale_b = context.scales
            if name in LENGTH_KEYS:
                value_a, value_b = value_a * scale_a, value_b * scale_b
            elif name not in PLAIN_MESH_KEYS and scale_a != scale_b and \
                    not close(value_a, value_b):
                return 'unchecked'
        return same_value(value_a, value_b, si_a, si_b, self.parameters)

    def compare_map(self, na, nb, name, value_a, value_b, raw_b, da, db,
                    context: _Context):
        """Compares a study step's map (physics to 'on', ...) key by
        key, the keys in a's tags; a key one side lacks has its default
        there. b's own values are shown, with a's tags where they differ.
        """
        map_a = value_a if isinstance(value_a, dict) else {}
        map_b = value_b if isinstance(value_b, dict) else {}
        own_b = {self.translator.key(key): item
                 for key, item in raw_b.items()} \
            if isinstance(raw_b, dict) else {}
        defaults_a = da.get(name) if isinstance(da.get(name), dict) else {}
        defaults_b = self.translator.value(db.get(name), context.component_b)
        defaults_b = defaults_b if isinstance(defaults_b, dict) else {}
        for key in list(map_a) + [k for k in map_b if k not in map_a]:
            x = map_a[key] if key in map_a else defaults_b.get(key)
            y = map_b[key] if key in map_b else defaults_a.get(key)
            if same_value(x, y, parameters=self.parameters):
                shown_b = own_b.get(key, y) if key in map_b else y
                ending = f" (in a's tags: {shown(y)})" \
                    if shown_b != y else ''
                self.item('property', na, nb, f'{self.head(na, nb)}: '
                          f'{name}[{key}] is {shown(x)} in a, '
                          f'{shown(shown_b)} in b{ending}', a=x, b=shown_b,
                          context=context, name=f'{name}[{key}]')

    def compare_rows(self, na, nb, rows_a: list, rows_b: list,
                     context: _Context):
        """Compares global equations by the name of each equation, in
        whatever feature and order they are."""
        def table(features):
            found = {}
            for feature in features:
                rows = feature['rows']
                defaults = feature.get('row_defaults', {})
                for i, name in enumerate(rows.get('name', [])):
                    row = {k: (v[i] if i < len(v) else defaults.get(k))
                           for k, v in rows.items()
                           if k not in ('name', 'description')}
                    found[name] = (feature, row)
            return found

        def settings(fa: dict, fb: dict) -> tuple[dict, dict, list]:
            """The settings of two features as each has them: a value
            one sets comes from the other's defaults (of the same type)."""
            pa, pb = fa.get('properties') or {}, fb.get('properties') or {}
            da, db = fa.get('defaults') or {}, fb.get('defaults') or {}
            same_type = fa.get('type') == fb.get('type')
            own_a, own_b, blocked = {}, {}, []
            for key in list(pa) + [k for k in pb if k not in pa]:
                if same_type and (key not in pa and key not in db or
                                  key not in pb and key not in da):
                    blocked.append(key)
                    continue
                if not same_type:
                    own_a[key], own_b[key] = pa.get(key), (pb.get(key), False)
                    continue
                # b's values are translated below; a's defaults are a's
                own_a[key] = pa[key] if key in pa else \
                    self.translator.value(db[key], context.component_b)
                own_b[key] = (pb[key], False) if key in pb else \
                    (da[key], True)
            return own_a, own_b, blocked

        found_a, found_b = table(rows_a), table(rows_b)
        for name in list(found_a) + [n for n in found_b
                                     if n not in found_a]:
            if name not in found_b or name not in found_a:
                side = 'a' if name in found_a else 'b'
                feature, row = (found_a if side == 'a' else found_b)[name]
                fa, fb = (feature, None) if side == 'a' else (None, feature)
                self.item(f'only_in_{side}', fa, fb,
                          f'{self.head(fa, fb)}: equation {name} only in '
                          f'{side}',
                          a=row if side == 'a' else None,
                          b=row if side == 'b' else None, context=context,
                          name=name)
                continue
            (fa, row_a), (fb, row_b) = found_a[name], found_b[name]
            own_a, own_b, blocked = settings(fa, fb)
            if blocked:
                self.unchecked('a', fa, f'{self.head(fa, fb)}: defaults '
                               f'unknown, not compared: {", ".join(blocked)}',
                               context)
            values_a = {**row_a, **own_a}
            values_b = {**{k: (v, False) for k, v in row_b.items()},
                        **own_b}
            for key in list(values_a) + [k for k in values_b
                                         if k not in values_a]:
                x = values_a.get(key)
                raw, of_a = values_b.get(key, (None, False))
                y = raw if of_a else self.translator.value(
                    raw, context.component_b)
                if same_value(x, y, parameters=self.parameters):
                    ending = '' if of_a else in_a_tags(x, raw, y)
                    self.item('property', fa, fb,
                              f'{self.head(fa, fb)}: equation {name}: '
                              f'{key} is {shown(x)} in a, '
                              f'{shown(raw)} in b{ending}', a=x,
                              b=raw, context=context, name=name)

    # Where nodes apply, and why it differs

    def cover(self, side: str, selection) -> set:
        """The cells a selection applies to, as (a's geometry, level,
        cell index or 'rest')."""
        if not isinstance(selection, dict):
            return set()
        level = selection.get('level')
        places = selection.get('applied', selection.get('entities'))
        gtag = selection.get('geometry')
        if side == 'b':
            gtag = self.translator.maps['geometry'].get(str(gtag))
        pair = self.geometries.get(str(gtag)) if gtag is not None else None
        if pair is None or level not in pair.dims or \
                not (isinstance(places, list) or places == 'all'):
            return set()
        index = 0 if side == 'a' else 1
        rows = pair.table(index, level).rows_of(places)
        if rows is None:
            return set()
        cells = pair.cell_of(index, level)
        return {(gtag, level, cells[row]) for row in rows}

    def area(self, side: str, node: dict) -> set:
        """
        Where a node can change where others apply: its own entities; for
        a pair its source and destination; for an interface those of all
        its features and of the component's couplings.
        """
        if 'source' in node or 'destination' in node:
            return self.cover(side, node.get('source')) | \
                self.cover(side, node.get('destination'))
        if 'identifier' not in node:
            return self.cover(side, node.get('selection'))
        found = set()
        for feature in _features(node):
            found |= self.cover(side, feature.get('selection'))
        component = self.component_nodes.get(
            (side, str(node.get('path', '')).partition('/')[0]), {})
        for coupling in component.get('multiphysics', []):
            found |= self.cover(side, coupling.get('selection'))
        return found

    def fold_children(self):
        """
        Moves the `active` items of nodes under a node disabled in the
        same model into the outermost such node's item (COMSOL reads all
        nodes under a disabled one as disabled).
        """
        actives = {item['path']['a']: item for item in self.items
                   if item['kind'] == 'active' and item['path']['a']}
        kept = []
        for item in self.items:
            parent = None
            if item['kind'] == 'active' and item['path']['a']:
                parts = item['path']['a'].split('/')
                for end in range(1, len(parts)):
                    found = actives.get('/'.join(parts[:end]))
                    if found is not None and found.get('_on') == \
                            item.get('_on'):
                        parent = found
                        break
            if parent is None:
                kept.append(item)
            else:
                parent.setdefault('consequences', []).append(item)
        self.items = kept

    def link(self):
        """
        Moves each `applied` item into the item that explains it: of the
        same bundle (or a pair, for a node that applies only on pairs),
        covering all of the cells where it differs. Prefers one that stays
        at the top, then one that covers it alone, then the first in the
        model tree; all candidates are in its `causes`.
        """
        causes = [item for item in self.items if item.get('_area') and
                  item['kind'] in ('selection', 'active', 'only_in_a',
                                   'only_in_b')]
        kept = []
        for item in self.items:
            if item['kind'] != 'applied' or item.get('_fold') or \
                    not item.get('_area'):
                kept.append(item)
                continue
            area = item['_area']
            found = [c for c in causes if c['_component'] ==
                     item['_component'] and c['path'] != item['path'] and
                     c['_area'] & area and self.may_cause(c, item)]
            covered = set().union(*(c['_area'] for c in found))
            if not found or not area <= covered:
                kept.append(item)
                continue
            found.sort(key=lambda c: (
                bool(c.get('_fold')) and c['_component'] in self.differing,
                not area <= c['_area'], self.rank(c)))
            item['causes'] = [c['path'] for c in found]
            found[0].setdefault('consequences', []).append(item)
        self.items = kept

    def may_cause(self, cause: dict, item: dict) -> bool:
        """Whether a node can change where another applies: the same
        bundle, or a pair for a node that applies on pairs only."""
        if cause['_bundle'] == item['_bundle']:
            return True
        if (cause['_bundle'] or ('',))[0] != 'pair' or \
                item['_bundle'][0] != 'physics':
            return False
        pairs = {}
        for side in 'ab':
            component = item['_component'] if side == 'a' else \
                self.translator_component(item['_component'])
            node = self.component_nodes.get((side, component), {})
            pairs[side] = set().union(*(
                self.area(side, pair) for pair in node.get('pairs', [])))
        sa, sb = item['_selections']
        return self.cover('a', sa) <= pairs['a'] and \
            self.cover('b', sb) <= pairs['b']

    def translator_component(self, component_a) -> str | None:
        for b_tag, a_tag in self.translator.maps['component'].items():
            if a_tag == component_a:
                return b_tag
        return None

    def rank(self, item: dict) -> tuple:
        """Where an item's node is in the model tree: a's nodes first,
        then the nodes only b has."""
        path_a, path_b = item['path']['a'], item['path']['b']
        if path_a in self.ranks['a']:
            return (0, self.ranks['a'][path_a])
        if path_b in self.ranks['b']:
            return (1, self.ranks['b'][path_b])
        return (2, item['_order'])

    # Notes and folding

    def add_notes(self):
        for side, described in (('a', self.a), ('b', self.b)):
            for note in described.get('notes', []):
                kind = note.get('kind')
                if kind in ('defaults_unknown', 'solver_not_compared'):
                    continue
                node = {'path': note.get('path'), 'label': None}
                na, nb = (node, None) if side == 'a' else (None, node)
                self.item('note' if kind == 'model_changed' else 'unchecked',
                          na, nb, f'{side}: {note.get("message")}')

    def fold(self):
        """Moves what a differing geometry causes into its item, and what
        the nodes of a component only one model has give into the
        component's item."""
        kept = []
        for item in self.items:
            lone = self.lone.get(item.get('_component'))
            if lone is not None and item is not lone:
                lone.setdefault('consequences', []).append(item)
                continue
            target = self.differing.get(item.get('_component'))
            if target is not None and item is not target and \
                    item.get('_fold') and item['kind'] != 'geometry':
                target['consequences'].append(item)
            else:
                kept.append(item)
        self.items = kept


def _longest_run(positions: list[int]) -> set[int]:
    """The indices of the longest increasing run of positions, the one
    with the smallest indices among equally long ones."""
    count = len(positions)
    longest = [1] * count
    for i in range(count - 1, -1, -1):
        for j in range(i + 1, count):
            if positions[j] > positions[i]:
                longest[i] = max(longest[i], longest[j] + 1)
    need = max(longest, default=0)
    found: set[int] = set()
    last = None
    for i in range(count):
        if need and longest[i] == need and \
                (last is None or positions[i] > positions[last]):
            found.add(i)
            last = i
            need -= 1
    return found


def _area_of(selection, found: dict) -> set:
    """The cells where a selection differs, as `cover` gives them."""
    if not isinstance(selection, dict):
        return set()
    return {(selection.get('geometry'), selection.get('level'), cell)
            for cell in found.get('cells', ())}


def _features(node: dict):
    for feature in node.get('features', []):
        yield feature
        yield from _features(feature)


def _tree_ranks(described: dict) -> dict[str, int]:
    """The paths of the nodes that override each other, in tree order:
    pairs, physics with their features, couplings, materials."""
    found: dict[str, int] = {}

    def add(node):
        path = node.get('path') if isinstance(node, dict) else None
        if isinstance(path, str) and path not in found:
            found[path] = len(found)

    for component in described.get('components', []):
        for pair in component.get('pairs', []):
            add(pair)
        for interface in component.get('physics', []):
            add(interface)
            for feature in _features(interface):
                add(feature)
        for coupling in component.get('multiphysics', []):
            add(coupling)
    for material in described.get('materials', []):
        add(material)
    return found


def _path(node) -> str | None:
    return node.get('path') if isinstance(node, dict) else None


def _label(node) -> str | None:
    return node.get('label') if isinstance(node, dict) else None


def _level(selection) -> str:
    return 'none' if selection is None else str(selection.get('level'))


def _empty(selection) -> bool:
    if not isinstance(selection, dict):
        return False
    if _exterior(selection, True):
        return False
    return selection.get('level') == 'none' or \
        selection.get('entities') == [] or selection.get('applied') == []


def _nothing(selection) -> bool:
    """Tells whether a selection selects nothing (not counting where a
    node applies: one overridden everywhere still says where it is)."""
    return isinstance(selection, dict) and not _exterior(selection) and (
        selection.get('level') == 'none' or selection.get('entities') == [])


def _exterior(selection, applied: bool = False) -> bool:
    """Whether a selection holds the exterior of boundary elements
    (domain 0): as selected, or where the node applies."""
    if not isinstance(selection, dict):
        return False
    selected = bool(selection.get('exterior'))
    return bool(selection.get('exterior_applied', selected)) if applied \
        else selected


def _with_exterior(found: dict, side: str) -> dict:
    """A selection comparison that also differs in the exterior."""
    text = f'the exterior (domain 0) only in {side}'
    found = {**found, 'same': False}
    found['text'] = ', '.join(t for t in (found.get('text'), text) if t)
    found.setdefault('numbers', {'a': [], 'b': []})
    found.setdefault('places', {'a': [], 'b': []})
    if 'elsewhere' in found:
        # both geometries have the exterior: never folded
        inside: dict[str, Any] = dict(
            found.get('inside') or {'numbers': {'a': [], 'b': []}})
        inside['text'] = ', '.join(
            str(t) for t in (inside.get('text'), text) if t)
        found['inside'] = inside
    return found


def _same_name(na: dict, nb: dict) -> bool:
    return na.get('tag') == nb.get('tag') and \
        na.get('label') == nb.get('label')


def _numbers(found: dict) -> dict:
    return {'numbers': found['numbers']} if 'numbers' in found else {}


def _shape(geometry: dict) -> dict:
    return {key: geometry.get(key) for key in (
        'dimension', 'axisymmetric', 'length_unit', 'voids',
        'bounding_box', 'finalize')}


def _article(geometry: dict) -> str:
    finalize = str(geometry.get('finalize'))
    return {'union': 'a union', 'assembly': 'an assembly'}.get(
        finalize, finalize)


def _parameter_si(entry: dict):
    if entry.get('value') is None:
        return None
    return {'value': entry['value'], 'unit': entry.get('unit')}


def _material_kind(node: dict) -> str:
    kind = node.get('type', '')
    return 'material' if kind in ('Common', 'Link') else kind


def _flat_groups(node: dict) -> dict:
    found = {}
    for gtag, group in (node.get('groups') or {}).items():
        for key, value in group.get('properties', {}).items():
            found[f'{gtag}/{key}'] = value
    return found


def _jaccard(a: list, b: list) -> float:
    if not a and not b:
        return 1.0
    counts_a = {x: a.count(x) for x in a}
    counts_b = {x: b.count(x) for x in b}
    keys = set(counts_a) | set(counts_b)
    common = sum(min(counts_a.get(k, 0), counts_b.get(k, 0)) for k in keys)
    return common / sum(max(counts_a.get(k, 0), counts_b.get(k, 0))
                        for k in keys)
