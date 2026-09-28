"""Tests for measurements of the finished geometry."""
import math

import numpy
import pytest

import mphkit as mk


@pytest.fixture
def pair(model, geom):
    """Two 10 mm cubes: domain 1 at x = 0…10, domain 2 at x = 20…30."""
    mk.block(geom, (10, 10, 10))
    mk.block(geom, (10, 10, 10), (20, 0, 0))
    model.build(geom)
    return geom


@pytest.mark.parametrize('unit', ['m', 'mm'])
def test_measure_length_unit(model, unit):
    geom = mk.geometry(model, 3, length_unit=unit)
    mk.block(geom, (10, 10, 10))
    model.build(geom)
    assert mk.measure(geom, 'domain') == pytest.approx(1000)
    assert mk.bounding_box(geom, 'domain')['x'] == pytest.approx((0, 10))


def test_measure_levels(pair):
    assert mk.measure(pair, 'domain', 1) == pytest.approx(1000)
    assert mk.measure(pair, 'domain') == pytest.approx(2000)
    assert mk.measure(pair, 'domain', [1, 2]) == pytest.approx(2000)
    assert mk.measure(pair, 'domain', numpy.array([2])) == pytest.approx(1000)
    left = mk.sel.box(pair, 'boundary', x=0)
    assert mk.measure(pair, 'boundary', left) == pytest.approx(100)
    top = mk.sel.box(pair, 'edge', x=(-1, 11), z=10)
    assert mk.measure(pair, 'edge', top) == pytest.approx(40)
    assert mk.measure(pair, 'domain', []) == 0.0


def test_measure_adjacent(pair):
    walls = mk.sel.adjacent(pair, mk.sel.box(pair, 'domain', x=(-1, 11)))
    assert mk.measure(pair, 'boundary', walls) == pytest.approx(600)
    with pytest.raises(ValueError, match='not a domain selection'):
        mk.measure(pair, 'domain', walls)


def test_measure_errors(pair):
    with pytest.raises(ValueError, match='Points have no size'):
        mk.measure(pair, 'point', 1)
    with pytest.raises(ValueError, match='has 2 domain'):
        mk.measure(pair, 'domain', 3)
    with pytest.raises(TypeError):
        mk.measure(pair, 'domain', True)
    with pytest.raises(TypeError):
        mk.measure(pair, 'domain', 1.5)


def test_bounding_box(pair):
    box = mk.bounding_box(pair, 'domain', 2)
    assert sorted(box) == ['x', 'y', 'z']
    assert box['x'] == pytest.approx((20, 30))
    assert box['y'] == box['z'] == pytest.approx((0, 10))
    assert mk.sel.find(pair, 'domain', **box) == [2]
    corner = mk.sel.find(pair, 'point', x=30, y=10, z=10)
    point = mk.bounding_box(pair, 'point', corner)
    assert [point[axis] for axis in 'xyz'] == \
        [pytest.approx((30, 30)), pytest.approx((10, 10)),
         pytest.approx((10, 10))]
    assert mk.bounding_box(pair, 'domain', []) is None


def test_empty_level(model, geom):
    plane = mk.workplane(geom, quickz=0)
    mk.square(plane, 1)
    model.build(geom)
    assert geom.java.getNDomains() == 0
    assert mk.measure(geom, 'domain') == 0.0
    assert mk.bounding_box(geom, 'domain') is None
    assert mk.measure(geom, 'boundary') == pytest.approx(1)


def test_curved_is_approximate(model, geom):
    mk.cylinder(geom, 2, 5)
    model.build(geom)
    assert mk.measure(geom, 'domain') == pytest.approx(math.pi*4*5, rel=0.01)


def test_2d_and_1d(model):
    square = mk.geometry(model, 2)
    mk.square(square, 2)
    model.build(square)
    assert mk.measure(square, 'domain') == pytest.approx(4)
    assert mk.measure(square, 'boundary') == pytest.approx(8)
    assert sorted(mk.bounding_box(square, 'domain')) == ['x', 'y']
    line = mk.geometry(model, 1)
    mk.feature(line, 'Interval', coord=[0, 2])
    model.build(line)
    assert mk.measure(line, 'domain') == pytest.approx(2)
    assert mk.bounding_box(line, 'domain') == {'x': pytest.approx((0, 2))}
    with pytest.raises(ValueError, match='Points have no size'):
        mk.measure(line, 'boundary')
