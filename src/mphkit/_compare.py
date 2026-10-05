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

import json
import math
import re
from collections.abc import Iterable
from typing import Any

from mph.model import Model

from . import _describe

# What `ignore` may leave out besides kinds, and what `show` may add
IGNORABLE = frozenset({'empty', 'applied', 'order', 'solver', 'mesh',
                       'expression', 'unchecked', 'note', 'same_applied',
                       'material_info'})
SHOWABLE = frozenset({'label'})
# Sections of the result, in order
SECTIONS = {'parameter': 0, 'geometry': 1, 'unchecked': 3, 'note': 4}
# Properties that name a node, compared through the pairing instead
NAME_KEYS = frozenset({'funcname', 'opname', 'probename', 'funcnametable'})
# Material properties that describe a library entry, not the material
MATERIAL_INFO = ('sys', 'argders')
# Mesh size properties in the geometry's length unit
LENGTH_KEYS = frozenset({'hmax', 'hmin'})
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


def compare(a, b, /, *, tolerance: float = 1e-6, ignore=(),
            show=()) -> list[dict]:
    """
    Returns the differences between the settings of two models, as a
    list of dicts, e.g. to check a script that rebuilds a model made in
    the COMSOL Desktop:

    ```python
    differences = mk.compare(old, model)
    for d in differences:
        if d['kind'] != 'note':
            print(d['kind'], d['message'])
    # property Temperature 1 (a temp1, b temp3): T0 is 100[degC] in a,
    #     50[degC] in b
    ```

    `a` and `b` are models or results of `mk.describe` (also read back
    from JSON); describe a model with `solver=True` to compare solvers
    too. Nodes are paired without their tags or order: by name where
    expressions call them by name (parameters, variables, functions,
    operators, probes, global equations), else by type and where their
    selections lie, on a table that pairs the entities of the two
    geometries by bounding box and size (also where a face is split into
    pieces differently). Values are compared after model b's tags in
    them are translated to model a's (`ht2.T`, `comp2.`, an operator
    called by another name); values COMSOL evaluates to the same SI value
    and unit are equal ('100[degC]' and '373.15[K]'), lists like
    'range(0,0.1,1)' are compared by their numbers.

    Each item has `kind`, `path` and `label` (each {'a', 'b'}; None on
    the side that lacks it), `message` (one line), and the values `a`
    and `b` as each model has them. Kinds, in this order:
    'parameter', 'geometry' (dimension, bounding box, entities without a
    counterpart, union against assembly; what follows from it, e.g.
    selections that differ because a face moved, is in its
    `consequences`: fix the geometry first), then per node 'only_in_a',
    'only_in_b' (`a` or `b` holds the whole node; `empty: True` if it
    selects nothing), 'property', 'expression' (same value, but one side
    uses parameters or leaves out the unit), 'variable', 'active',
    'selection' (with the entity `numbers` and places of the
    difference; `same_applied: True` if both apply to the same
    entities), 'solver' and 'label'; then 'unchecked' (what could not
    be compared, e.g. unknown defaults) and 'note'. `from_default: True`
    marks a value one side has from its defaults, `matched_by_order:
    True` a node paired by order in a component whose geometry differs.
    Places are in each model's length unit; comparisons are in SI.

    `tolerance` is relative to the size of the geometries. `ignore` takes
    kinds and 'empty', 'mesh' (all of the meshes), 'same_applied' and
    'material_info' (library entries of materials, e.g. 'sys');
    `show={'label'}` adds labels that differ.

    Not compared: results, where nodes apply when features overlap and
    the order of mesh operations and study steps (both noted as
    'unchecked'), expressions COMSOL cannot evaluate other than as
    written ('2*a' and 'a*2' differ), the two faces of a pair in an
    assembly and other entities with the same box and size (they are one
    row of the table), a probe's name used as a variable, and tags in
    properties other than the ones of nodes, physics, materials,
    coordinate systems, pairs, studies and solvers (e.g. load groups).
    """
    names = set(ignore) | set(show)
    unknown = names - IGNORABLE - SHOWABLE - set(KINDS)
    if unknown:
        raise ValueError(
            f'mk.compare does not know {sorted(unknown)}; ignore takes '
            f'{sorted(IGNORABLE | set(KINDS))}, show takes '
            f'{sorted(SHOWABLE)}.')
    found = _Comparison(_described(a, 'a'), _described(b, 'b'),
                        tolerance).run()
    hidden = (set(ignore) | {'label'}) - set(show)
    return _filtered(found, hidden)


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
    return side


def _filtered(items: list[dict], hidden: set[str]) -> list[dict]:
    """Leaves out what `ignore` names, also among consequences, and the
    internal keys (starting with '_')."""
    found = []
    for item in items:
        if item['kind'] in hidden or \
                ('empty' in hidden and item.get('empty')) or \
                ('same_applied' in hidden and item.get('same_applied')) or \
                ('material_info' in hidden and item.get('material_info')) \
                or ('mesh' in hidden and item.get('_mesh')):
            continue
        item = {key: value for key, value in item.items()
                if not key.startswith('_')}
        if 'consequences' in item:
            item['consequences'] = _filtered(item['consequences'], hidden)
        found.append(item)
    return found


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
            if step == 0 or (stop - start) * step < 0:
                return None
            slack = 1e-9 * abs(step)
            count = 0
            while count < 10**6:
                value = start + count * step
                if (value - stop) * math.copysign(1, step) > slack:
                    break
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
    if _number(a) and _number(b):
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
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def place_text(place: dict) -> str:
    """Writes the box of an entity, e.g. 'x=100, y 0..50, z 0..10'."""
    if 'unknown' in place:
        return f'entity {place["unknown"]} (not measured)'
    parts = []
    for axis in AXES:
        if axis in place:
            low, high = place[axis]
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
        self.unknown: list[int] = []
        for number, place in enumerate(places, 1):
            if 'unknown' in place:
                self.unknown.append(number)
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
        for i in list(singles):
            row = one.rows[i]
            if row['size'] is None:
                continue
            pieces = [j for j in pieces_left
                      if tol.inside(many.rows[j]['box'], row['box'])]
            if not pieces:
                continue
            if sizes_close(row['size'], many.size(pieces)) and \
                    tol.boxes(row['box'], many.box(pieces)):
                singles.remove(i)
                for j in pieces:
                    pieces_left.remove(j)
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
        for cell_a, cell_b in cells:
            in_a, in_b = cell_a & rows_a, cell_b & rows_b
            if in_a == cell_a and in_b == cell_b:
                common += ta.size(cell_a)
            elif in_a or in_b:
                only_a |= in_a
                only_b |= in_b
        rest_a = sorted(rows_a & set(left_a))
        rest_b = sorted(rows_b & set(left_b))
        if same_region(ta, rest_a, tb, rest_b, self.tol):
            common += ta.size(rest_a)
        else:
            only_a |= set(rest_a)
            only_b |= set(rest_b)
        total = max(ta.size(rows_a), tb.size(rows_b))
        found: dict[str, Any] = {
            'same': not only_a and not only_b,
            'overlap': common / total if total else 1.0, 'unknown': False}
        if not found['same']:
            found['numbers'] = {
                'a': sorted(n for r in only_a for n in ta.rows[r]['numbers']),
                'b': sorted(n for r in only_b for n in tb.rows[r]['numbers'])}
            found['places'] = {
                'a': [ta.rows[r]['place'] for r in sorted(only_a)],
                'b': [tb.rows[r]['place'] for r in sorted(only_b)]}
            parts = [f'{len(found["places"][side])} {level} only in {side} '
                     f'({places_text(found["places"][side])})'
                     for side in 'ab' if found['places'][side]]
            found['text'] = ', '.join(parts)
        return found


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
                                  'sequence')}
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
                     'pair', 'study', 'sequence'):
            if value in self.maps[kind]:
                return self.maps[kind][value]
        return None

    def value(self, value, component: str | None = None):
        """Translates a value of model b: tags, lists, step maps."""
        if isinstance(value, str):
            found = self.whole(value)
            if found is not None:
                return found
            if PATH.fullmatch(value):
                return self.path(value)
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
        for kind in ('component', 'identifier', 'multiphysics'):
            if word in self.maps[kind]:
                return self.maps[kind][word]
        if word in self.b_words and word in self.a_words:
            return f'<b only:{word}>'
        return word

    def called(self, word: str, component: str | None) -> str:
        for scope in (component, None):
            names = self.names.get(scope, {})
            if word in names:
                return names[word]
        return word


def words(described: dict) -> set[str]:
    """All tags, identifiers and names of a description."""
    found: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for key in ('tag', 'identifier'):
                if isinstance(value.get(key), str):
                    found.add(value[key])
            found.update(n for n in value.get('name', [])
                         if isinstance(value.get('name'), list))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk({key: described.get(key) for key in (
        'functions', 'couplings', 'coordinates', 'materials',
        'definitions', 'probes', 'components', 'studies')})
    return found


##############
# Comparison #
##############

class _Context:
    """Where two paired nodes are: their components (a's and b's tags),
    whether in a mesh (and the geometries' length scales), and whether
    the component's geometry differs or the pair is a guess by order."""

    def __init__(self, component=None, component_b=None, mesh=False,
                 scales=(1.0, 1.0), differs=False, by_order=False):
        self.component, self.component_b = component, component_b
        self.mesh, self.scales = mesh, scales
        self.differs, self.by_order = differs, by_order

    def ordered(self) -> _Context:
        return _Context(self.component, self.component_b, self.mesh,
                        self.scales, self.differs, True)


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
        self.pending: list[tuple] = []
        self.unchecked_paths: set[tuple] = set()
        self.count = 0

    def run(self) -> list[dict]:
        self.compare_parameters()
        pairs = self.pair_components()
        self.pair_lists(('functions', 'couplings', 'probes'), named=True)
        self.pair_lists(('coordinates',), kind='coordinate')
        self.pair_materials(pairs)
        for ca, cb, context in pairs:
            self.pair_component(ca, cb, context)
        self.pair_lists(('definitions',))
        self.compare_variables()
        self.pair_studies()
        for compare, args in self.pending:
            compare(*args)
        self.add_notes()
        self.fold()
        self.items.sort(key=lambda item: (
            SECTIONS.get(item['kind'], 2),
            item['kind'] == 'only_in_b', item['_order']))
        return self.items

    # Items

    def item(self, kind: str, na, nb, message: str, a=None, b=None,
             context: _Context | None = None, fold: bool = False,
             **extra) -> dict:
        entry = {'kind': kind,
                 'path': {'a': _path(na), 'b': _path(nb)},
                 'label': {'a': _label(na), 'b': _label(nb)},
                 'message': message, 'a': a, 'b': b, **extra}
        if context is not None:
            if context.by_order:
                entry['matched_by_order'] = True
            if context.mesh:
                entry['_mesh'] = True
            entry['_component'] = context.component
            entry['_fold'] = fold or context.by_order
        entry['_order'] = self.count
        self.count += 1
        self.items.append(entry)
        return entry

    def unchecked(self, side: str, node, message: str,
                  context: _Context | None = None):
        key = (side, _path(node))
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
        message = f'{_head(na, nb)}: only in {side}'
        if empty:
            message += ' (selects nothing)'
        extra = {'empty': True} if empty else {}
        self.item(f'only_in_{side}', na, nb, message,
                  a=node if side == 'a' else None,
                  b=node if side == 'b' else None, context=context,
                  fold=fold, **extra)

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
                          b=entry if side == 'b' else None)
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
                          a=ea['expression'], b=eb['expression'])

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
                self.item('only_in_a', ca, None,
                          f'{_head(ca, None)}: component only in a', a=ca)
        for j, cb in enumerate(cb_list):
            if j not in used_b:
                self.item('only_in_b', None, cb,
                          f'{_head(None, cb)}: component only in b', b=cb)
        return found

    def pair_geometries(self, ca: dict, cb: dict) -> bool:
        """Pairs the geometries of two components and reports how they
        differ; returns whether the shape differs."""
        ga_list, gb_list = ca['geometries'], cb['geometries']
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
                    self.unchecked(side, geometry, f'{_head_one(geometry)}: '
                                   'length unit without a known size',
                                   context)
            found = pair.differences()
            if found:
                differs = True
                entry = self.item(
                    'geometry', ga, gb,
                    f'{_head(ga, gb)}: geometry differs: {"; ".join(found)}',
                    a=_shape(ga), b=_shape(gb), context=context)
                entry['consequences'] = []
                self.differing.setdefault(ca['tag'], entry)
            if ga.get('finalize') != gb.get('finalize'):
                self.item('geometry', ga, gb,
                          f'{_head(ga, gb)}: a forms {_article(ga)}, b '
                          f'{_article(gb)}', a=ga.get('finalize'),
                          b=gb.get('finalize'), context=context)
                for level, (left_a, left_b) in pair.leftovers().items():
                    self.item('unchecked', ga, gb,
                              f'{_head(ga, gb)}: entities that differ with '
                              f'union against assembly: '
                              f'{pair.rows_text(level, left_a, left_b)}',
                              context=context)
        for side, geometries, used in (('a', ga_list, used_a),
                                       ('b', gb_list, used_b)):
            for i, geometry in enumerate(geometries):
                if i not in used:
                    na, nb = (geometry, None) if side == 'a' else \
                        (None, geometry)
                    self.item(f'only_in_{side}', na, nb,
                              f'{_head(na, nb)}: geometry only in {side}',
                              a=na, b=nb, context=_Context(ca['tag'],
                                                           cb['tag']))
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
            return _different('on other geometries')
        return pair.selection(sa, sb)

    def overlap(self, na: dict, nb: dict) -> float:
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
        their settings are, the last one of a type left on each side
        with each other, and in a component whose geometry differs the
        rest by order. Reports the nodes left over; returns the pairs
        with their contexts.
        """
        def type_of(node):
            return kind(node) if kind else node.get('type', 'variables')

        pairs: list[tuple] = []
        left_a = list(range(len(list_a)))
        left_b = list(range(len(list_b)))
        if named:
            for i in list(left_a):
                names_a = set(list_a[i].get('name') or [])
                for j in left_b:
                    nb = list_b[j]
                    if names_a & set(nb.get('name') or []) and \
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
                            list_a[i], list_b[j], context), i, j))
            for _, _, i, j in sorted(scored):
                if i in group_a and j in group_b:
                    pairs.append((i, j, context))
                    group_a.remove(i)
                    group_b.remove(j)
                    left_a.remove(i)
                    left_b.remove(j)
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
        components = self.translator.maps['component']
        for key in keys:
            list_a, list_b = self.a.get(key, []), self.b.get(key, [])
            scopes = dict.fromkeys(
                [n.get('component') for n in list_a] +
                [components.get(n.get('component'), n.get('component'))
                 for n in list_b])
            for scope in scopes:
                in_a = [n for n in list_a if n.get('component') == scope]
                in_b = [n for n in list_b if components.get(
                    n.get('component'), n.get('component')) == scope]
                scope_b = in_b[0].get('component') if in_b else None
                context = self.context_of(scope, scope_b)
                for na, nb, ctx in self.pair(in_a, in_b, context, named):
                    names = self.translator.names.setdefault(
                        nb.get('component'), {})
                    for name_b, name_a in zip(nb.get('name') or [nb['tag']],
                                              na.get('name') or [na['tag']]):
                        names[name_b] = name_a
                    if kind:
                        self.translator.add(kind, nb['tag'], na['tag'])
                    self.later(self.compare_node, na, nb, ctx)

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
            for na, nb, ctx in self.pair(in_a, in_b, context,
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
            self.compare_values(na, nb, context, skip={'link'},
                                info=frozenset(MATERIAL_INFO))
        self.compare_active(na, nb, context)
        self.compare_selection(na, nb, context)
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
        head = f'{_head(na, nb)} {gtag}'
        for key in list(pa) + [k for k in pb if k not in pa]:
            if key in pa and key in pb:
                value_b = self.translator.value(pb[key], context.component_b)
                found = same_value(pa[key], value_b, si_a.get(key),
                                   si_b.get(key), self.parameters)
                if found:
                    self.item(found, na, nb,
                              f'{head}: {key} is {shown(pa[key])} in a, '
                              f'{shown(pb[key])} in b', a=pa[key], b=pb[key],
                              context=context, group=gtag, name=key)
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

    # Components

    def pair_component(self, ca: dict, cb: dict, context: _Context):
        component = context.component
        for na, nb, ctx in self.pair(ca.get('pairs', []),
                                     cb.get('pairs', []), context):
            self.translator.add('pair', nb['tag'], na['tag'])
            self.later(self.compare_pair, na, nb, ctx)
        interfaces = self.pair(ca.get('physics', []), cb.get('physics', []),
                               context)
        for na, nb, _ in interfaces:
            self.translator.add('physics', nb['tag'], na['tag'])
            self.translator.add('identifier', nb.get('identifier'),
                                na.get('identifier'))
        for na, nb, ctx in interfaces:
            self.pair_interface(na, nb, ctx)
        for na, nb, ctx in self.pair(ca.get('multiphysics', []),
                                     cb.get('multiphysics', []), context):
            self.translator.add('multiphysics', nb['tag'], na['tag'])
            self.later(self.compare_node, na, nb, ctx)
        self.pair_meshes(ca, cb, context)
        del component

    def pair_interface(self, na: dict, nb: dict, context: _Context):
        rows_a = [f for f in na.get('features', []) if 'rows' in f]
        rows_b = [f for f in nb.get('features', []) if 'rows' in f]
        features = self.pair_features(
            [f for f in na.get('features', []) if 'rows' not in f],
            [f for f in nb.get('features', []) if 'rows' not in f],
            context, nb)
        del features
        self.later(self.compare_interface, na, nb, context)
        self.later(self.compare_rows, na, nb, rows_a, rows_b, context)

    def pair_features(self, list_a, list_b, context, owner_b) -> list:
        """Pairs features (also of features) and records their tags under
        the interface's identifier and tag in model b."""
        pairs = self.pair(list_a, list_b, context)
        for key in (owner_b.get('identifier'), owner_b.get('tag')):
            if key:
                tags = self.translator.features.setdefault(key, {})
                for fa, fb, _ in pairs:
                    tags[fb['tag']] = fa['tag']
        for fa, fb, ctx in pairs:
            self.later(self.compare_node, fa, fb, ctx, False)
            self.pair_features(fa.get('features', []),
                               fb.get('features', []), ctx, owner_b)
        return pairs

    def pair_meshes(self, ca: dict, cb: dict, context: _Context):
        geometries = self.translator.maps['geometry']
        meshes_b = list(cb.get('meshes', []))
        pairs = []
        only_a = []
        for ma in ca.get('meshes', []):
            match = next((mb for mb in meshes_b if geometries.get(
                mb.get('geometry')) == ma.get('geometry')), None)
            if match is None:
                only_a.append(ma)
            else:
                meshes_b.remove(match)
                pairs.append((ma, match))
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
                          f'{_head(ma, mb)}: size level is '
                          f'{ma.get("size_level")} in a, '
                          f'{mb.get("size_level")} in b',
                          a=ma.get('size_level'), b=mb.get('size_level'),
                          context=ctx)
            self.pair_features(ma.get('features', []),
                               mb.get('features', []), ctx, {})

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
            tags = self.translator.steps.setdefault(sb['tag'], {})
            for fa, fb, _ in steps:
                tags[fb['tag']] = fa['tag']
                self.later(self.compare_node, fa, fb, ctx)
            self.later(self.compare_solver, sa, sb)

    def compare_solver(self, sa: dict, sb: dict):
        solver_a, solver_b = sa.get('solver') or {}, sb.get('solver') or {}
        statuses = {solver_a.get('status'), solver_b.get('status')}
        if 'not asked' in statuses:
            self.item('note', sa, sb, f'{_head(sa, sb)}: solvers not '
                      'compared; describe both with solver=True')
            return
        if not statuses <= {'compared', 'automatic'}:
            for side, solver, study in (('a', solver_a, sa),
                                        ('b', solver_b, sb)):
                if solver.get('status') not in ('compared', 'automatic'):
                    self.unchecked(side, study, f'{_head_one(study)}: solver '
                                   f'not compared: {solver.get("reason")}')
            return
        keys_a = {_change_key(c): c for c in solver_a.get('changes', [])}
        keys_b = {_change_key(c): c for c in solver_b.get('changes', [])}
        for keys, side in ((keys_a, 'a'), (keys_b, 'b')):
            other = keys_b if side == 'a' else keys_a
            for key, change in keys.items():
                if key not in other:
                    self.item('solver', sa, sb,
                              f'{_head(sa, sb)}: solver {change["path"]} '
                              f'({change["type"]}) {change["change"]} '
                              f'{shown(change.get("properties", ""))} only '
                              f'in {side}',
                              a=change if side == 'a' else None,
                              b=change if side == 'b' else None)

    # Variables

    def compare_variables(self):
        components = self.translator.maps['component']
        rows: dict[tuple, dict[str, list]] = {}
        for side, described in (('a', self.a), ('b', self.b)):
            for node in described.get('variables', []):
                scope = node.get('component')
                if side == 'b':
                    scope = components.get(scope, scope)
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
                self.compare_variable(name, node_a, expression_a, *match,
                                      context)
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
            self.item('variable', na, nb, f'{head}: {shown(expression_a)} '
                      f'in a, {shown(expression_b)} in b', a=expression_a,
                      b=expression_b, context=context, name=name)
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
            self.item('variable', na, nb,
                      f'{head}: selection differs: {found["text"]}',
                      a=na.get('selection'), b=nb.get('selection'),
                      context=context, fold=True, name=name,
                      **_numbers(found))

    # Comparing paired nodes

    def compare_node(self, na: dict, nb: dict, context: _Context,
                     children: bool = True):
        self.compare_active(na, nb, context)
        self.compare_label(na, nb, context)
        self.compare_values(na, nb, context)
        self.compare_selection(na, nb, context)
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
            self.item('property', na, nb, f'{_head(na, nb)}: type '
                      f'{na.get("type")} in a, {nb.get("type")} in b',
                      a=na.get('type'), b=nb.get('type'), context=context)
        straight = [self.selection(na.get(s), nb.get(s))
                    for s in ('source', 'destination')]
        if all(found['same'] for found in straight):
            return
        swapped = [self.selection(na.get(s), nb.get(t))
                   for s, t in (('source', 'destination'),
                                ('destination', 'source'))]
        if all(found['same'] for found in swapped):
            self.item('selection', na, nb, f'{_head(na, nb)}: source and '
                      'destination swapped', context=context)
            return
        for side, found in zip(('source', 'destination'), straight):
            if not found['same'] and not found['unknown']:
                self.item('selection', na, nb, f'{_head(na, nb)}: {side} '
                          f'differs: {found["text"]}', a=na.get(side),
                          b=nb.get(side), context=context, fold=True,
                          **_numbers(found))

    def compare_active(self, na, nb, context):
        if na.get('active', True) != nb.get('active', True):
            self.item('active', na, nb, f'{_head(na, nb)}: active in '
                      f'{"a" if na.get("active", True) else "b"} only',
                      a=na.get('active'), b=nb.get('active'),
                      context=context)

    def compare_label(self, na, nb, context):
        if na.get('label') != nb.get('label'):
            self.item('label', na, nb, f'{_head(na, nb)}: label '
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
            self.unchecked('a', na, f'{_head(na, nb)}: {what} not '
                           'compared (several levels or unmeasured '
                           'entities)', context)
            return
        if found['same']:
            return
        extra = _numbers(found)
        message = f'{_head(na, nb)}: {what} differs: {found["text"]}'
        applied_a = (sa or {}).get('applied', (sa or {}).get('entities'))
        applied_b = (sb or {}).get('applied', (sb or {}).get('entities'))
        if sa and sb and sa.get('level') == sb.get('level') and \
                (applied_a, applied_b) != (sa.get('entities'),
                                           sb.get('entities')) and \
                self.selection({**sa, 'entities': applied_a},
                               {**sb, 'entities': applied_b})['same']:
            extra['same_applied'] = True
            message += '; applies to the same entities'
        self.item('selection', na, nb, message, a=sa, b=sb, context=context,
                  fold=True, **extra)

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
        blocked = []
        translate = self.translator.value
        component_b = context.component_b
        for name in list(pa) + [n for n in pb if n not in pa]:
            if name in unused or name in NAME_KEYS or name in skip:
                continue
            from_default = False
            if name in pa:
                raw_a, value_a, s_a = pa[name], pa[name], si_a.get(name)
            elif name in db:
                raw_a, s_a = db[name], sd_b.get(name)
                value_a = translate(db[name], component_b)
                from_default = True
            else:
                blocked.append(name)
                continue
            if name in pb:
                raw_b, s_b = pb[name], si_b.get(name)
                value_b = translate(pb[name], component_b)
            elif name in da:
                raw_b, value_b, s_b = da[name], da[name], sd_a.get(name)
                from_default = True
            else:
                blocked.append(name)
                continue
            if isinstance(value_a, dict) or isinstance(value_b, dict):
                self.compare_map(na, nb, name, value_a, value_b, da, db,
                                 context)
                continue
            found = self.compare_one(name, value_a, value_b, s_a, s_b,
                                     context)
            if found == 'unchecked':
                self.unchecked('a', na, f'{_head(na, nb)}: {name} is in '
                               'the length units, which differ', context)
            elif found:
                extra = {'from_default': True} if from_default else {}
                if name in info:
                    extra['material_info'] = True
                self.item(found, na, nb,
                          f'{_head(na, nb)}: {name} is {shown(raw_a)} in a, '
                          f'{shown(raw_b)} in b', a=raw_a, b=raw_b,
                          context=context, name=name,
                          fold=from_default and context.mesh, **extra)
        if blocked:
            self.unchecked('a', na, f'{_head(na, nb)}: defaults unknown, '
                           f'not compared: {", ".join(blocked)}', context)

    def compare_one(self, name: str, value_a, value_b, si_a, si_b,
                    context: _Context) -> str | None:
        if context.mesh and _number(value_a) and _number(value_b):
            scale_a, scale_b = context.scales
            if name in LENGTH_KEYS:
                value_a, value_b = value_a * scale_a, value_b * scale_b
            elif scale_a != scale_b and not close(value_a, value_b):
                return 'unchecked'
        return same_value(value_a, value_b, si_a, si_b, self.parameters)

    def compare_map(self, na, nb, name, value_a, value_b, da, db,
                    context: _Context):
        """Compares a study step's map (physics to 'on', ...) key by
        key; a key one side lacks has its default there."""
        map_a = value_a if isinstance(value_a, dict) else {}
        map_b = value_b if isinstance(value_b, dict) else {}
        defaults_a = da.get(name) if isinstance(da.get(name), dict) else {}
        defaults_b = self.translator.value(db.get(name), context.component_b)
        defaults_b = defaults_b if isinstance(defaults_b, dict) else {}
        for key in list(map_a) + [k for k in map_b if k not in map_a]:
            x = map_a[key] if key in map_a else defaults_b.get(key)
            y = map_b[key] if key in map_b else defaults_a.get(key)
            if same_value(x, y, parameters=self.parameters):
                self.item('property', na, nb, f'{_head(na, nb)}: '
                          f'{name}[{key}] is {shown(x)} in a, {shown(y)} '
                          'in b', a=x, b=y, context=context,
                          name=f'{name}[{key}]')

    def compare_rows(self, na, nb, rows_a: list, rows_b: list,
                     context: _Context):
        """Compares global equations by the name of each equation, in
        whatever feature and order they are."""
        def table(features):
            found = {}
            for feature in features:
                rows = feature['rows']
                defaults = feature.get('row_defaults', {})
                settings = {k: v for k, v in
                            (feature.get('properties') or {}).items()}
                for i, name in enumerate(rows.get('name', [])):
                    row = {k: (v[i] if i < len(v) else defaults.get(k))
                           for k, v in rows.items()
                           if k not in ('name', 'description')}
                    row.update(settings)
                    found[name] = (feature, row)
            return found

        found_a, found_b = table(rows_a), table(rows_b)
        for name in list(found_a) + [n for n in found_b
                                     if n not in found_a]:
            if name not in found_b or name not in found_a:
                side = 'a' if name in found_a else 'b'
                feature, row = (found_a if side == 'a' else found_b)[name]
                fa, fb = (feature, None) if side == 'a' else (None, feature)
                self.item(f'only_in_{side}', fa, fb,
                          f'{_head(fa, fb)}: equation {name} only in {side}',
                          a=row if side == 'a' else None,
                          b=row if side == 'b' else None, context=context,
                          name=name)
                continue
            (fa, row_a), (fb, row_b) = found_a[name], found_b[name]
            for key in list(row_a) + [k for k in row_b if k not in row_a]:
                x = row_a.get(key)
                y = self.translator.value(row_b.get(key),
                                          context.component_b)
                if same_value(x, y, parameters=self.parameters):
                    self.item('property', fa, fb,
                              f'{_head(fa, fb)}: equation {name}: {key} is '
                              f'{shown(x)} in a, {shown(row_b.get(key))} in '
                              'b', a=x, b=row_b.get(key), context=context,
                              name=name)

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
        self.item('unchecked', None, None,
                  'Where nodes apply when features overlap and the order of '
                  'mesh operations and study steps are not compared yet.')
        self.item('note', None, None, 'Results (plots, datasets, tables) '
                  'are not compared.')

    def fold(self):
        """Moves what a differing geometry causes into its item."""
        kept = []
        for item in self.items:
            target = self.differing.get(item.get('_component'))
            if target is not None and item is not target and \
                    item.get('_fold') and item['kind'] != 'geometry':
                target['consequences'].append(item)
            else:
                kept.append(item)
        for target in self.differing.values():
            count = len(target['consequences'])
            if count:
                target['message'] += (f' (+{count} consequences: fix the '
                                      'geometry first)')
        self.items = kept


def _path(node) -> str | None:
    return node.get('path') if isinstance(node, dict) else None


def _label(node) -> str | None:
    return node.get('label') if isinstance(node, dict) else None


def _head(na, nb) -> str:
    """Names a pair of nodes: a's label (b's if a has none) and tags."""
    node = na if na is not None else nb
    label = (node or {}).get('label') or (node or {}).get('tag') or ''
    tags = [f'{side} {n["tag"]}' for side, n in (('a', na), ('b', nb))
            if isinstance(n, dict) and n.get('tag')]
    return f'{label} ({", ".join(tags)})' if tags else label


def _head_one(node) -> str:
    return _head(node, None)


def _level(selection) -> str:
    return 'none' if selection is None else str(selection.get('level'))


def _empty(selection) -> bool:
    if not isinstance(selection, dict):
        return False
    return selection.get('level') == 'none' or \
        selection.get('entities') == [] or selection.get('applied') == []


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


def _change_key(change: dict) -> str:
    return json.dumps([change.get('path'), change.get('type'),
                       change.get('change'),
                       {k: v[0] for k, v in
                        (change.get('properties') or {}).items()}],
                      sort_keys=True)
