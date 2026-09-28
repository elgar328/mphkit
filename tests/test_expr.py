"""Tests for the conversion of Python values to COMSOL expressions."""
import pytest

from mphkit._expr import expr, vector


@pytest.mark.parametrize('value, expected', [
    (5, '5'),
    (5.0, '5'),
    (2.5, '2.5'),
    (-0.1, '-0.1'),
    (1e-7, '1e-07'),
    ('r0/2', 'r0/2'),
    ('5[mm]', '5[mm]'),
])
def test_expr(value, expected):
    assert expr(value) == expected


@pytest.mark.parametrize('value', [True, None, [1, 2]])
def test_expr_rejects(value):
    with pytest.raises(TypeError):
        expr(value)


def test_vector_mixed():
    assert vector([0, 'x1', 2.5]) == ['0', 'x1', '2.5']


def test_vector_length():
    assert vector((1, 2, 3), length=3) == ['1', '2', '3']
    with pytest.raises(ValueError):
        vector((1, 2), length=3)


def test_vector_rejects_string():
    with pytest.raises(TypeError):
        vector('abc')


def test_expr_large_integer():
    # beyond float range, integers are passed on exactly
    assert expr(10**400) == str(10**400)
    assert expr(10**17 + 1) == '100000000000000001'
