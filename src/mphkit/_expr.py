"""Conversion of Python values to COMSOL expression strings."""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

Expr = str | Real
Vector = Sequence[Expr]


def expr(value: Expr) -> str:
    """Return `value` as a COMSOL expression string.

    Numbers are written in plain form (interpreted in the geometry's length
    unit); strings are passed through, so `'5[mm]'` or `'r0/2'` work as-is.
    """
    if isinstance(value, bool):
        raise TypeError(f'Expected a number or expression, got {value!r}.')
    if isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    raise TypeError(f'Expected a number or expression, got {value!r}.')


def vector(values: Vector) -> list[str]:
    """Return a sequence of numbers/expressions as a list of strings.

    MPh casts a list based on its first item, so a mixed list such as
    `[0, 'x1', 0]` fails there. Converting every item to a string avoids that.
    """
    if isinstance(values, str):
        raise TypeError(f'Expected a sequence, got the string {values!r}.')
    return [expr(v) for v in values]
