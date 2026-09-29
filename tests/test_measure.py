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


@pytest.mark.parametrize('pos', [(0, 0, 1000), (0, 0, 1000.3), (0, 0, 1),
                                 (1000, 0, 0)])
def test_measure_single_precision(model, pos):
    # COMSOL measures with coordinates rounded to float32
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 0.001), pos)
    model.build(geom)
    z = pos[2]
    thickness = float(numpy.float32(z + 0.001)) - float(numpy.float32(z))
    assert mk.measure(geom, 'domain') == pytest.approx(thickness, rel=1e-6)


def test_bounding_box_points_keep_full_precision(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 0.001), (0, 0, 1000))
    model.build(geom)
    top = mk.bounding_box(geom, 'point')['z'][1]
    assert top == pytest.approx(1000.001, abs=1e-9)
    top = mk.bounding_box(geom, 'domain')['z'][1]
    assert top == float(numpy.float32(1000.001))


def test_bounding_box_precision(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (13, 2, 3), (1.1, 0.7, 0.3))
    model.build(geom)
    assert mk.bounding_box(geom, 'domain')['x'][0] == \
        float(numpy.float32(1.1))
    corner = mk.sel.find(geom, 'point', x=1.1, y=0.7, z=0.3)
    assert mk.bounding_box(geom, 'point', corner)['x'] == (1.1, 1.1)
    # after a rotation, 0 may read as a tiny number at every level
    turned = mk.geometry(model, 3)
    mk.rotate(turned, mk.block(turned, (2, 2, 2)), 90)
    model.build(turned)
    point = mk.sel.find(turned, 'point', x=0, y=2, z=0)
    assert len(point) == 1
    assert abs(mk.bounding_box(turned, 'point', point)['x'][0]) < 1e-12
    assert abs(mk.bounding_box(turned, 'domain')['x'][1]) < 1e-12


@pytest.fixture
def two_blocks(model, geom):
    """Two 1 mm cubes side by side: boundary 6 is the face between them."""
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    model.build(geom)
    return geom


def test_summary(model, two_blocks):
    info = mk.summary(two_blocks)
    box = info.pop('bounding_box')
    assert info == {'dimension': 3, 'domains': 2, 'boundaries': 11,
                    'edges': 20, 'points': 12, 'voids': 0,
                    'length_unit': 'mm'}
    assert box == {'x': pytest.approx((0, 2)), 'y': pytest.approx((0, 1)),
                   'z': pytest.approx((0, 1))}


def test_summary_voids_and_dimensions(model):
    hollow = mk.geometry(model, 3)
    mk.difference(hollow, mk.block(hollow, (4, 4, 4), (-2, -2, -2)),
                  [mk.sphere(hollow, 1)])
    model.build(hollow)
    assert mk.summary(hollow)['voids'] == 1
    flat = mk.geometry(model, 2)
    mk.difference(flat, mk.rectangle(flat, (4, 3)),
                  [mk.circle(flat, 0.5, (2, 1.5))])
    model.build(flat)
    info = mk.summary(flat)
    assert set(info) == {'dimension', 'domains', 'boundaries', 'points',
                         'voids', 'bounding_box', 'length_unit'}
    assert (info['dimension'], info['voids']) == (2, 1)
    line = mk.geometry(model, 1)
    mk.interval(line, [0, 1, 3])
    model.build(line)
    assert mk.summary(line) == {'dimension': 1, 'domains': 2, 'points': 3,
                                'voids': 0, 'bounding_box': {'x': (0, 3)},
                                'length_unit': 'm'}


def test_summary_needs_a_built_geometry(model, geom):
    mk.block(geom, (1, 1, 1))
    with pytest.raises(RuntimeError, match='not built'):
        mk.summary(geom)
    with pytest.raises(TypeError, match='work plane'):
        mk.summary(mk.workplane(geom))


def test_coordinates(model, two_blocks):
    assert mk.coordinates(two_blocks, 'boundary', 6) == {
        5: (1, 0, 0), 6: (1, 0, 1), 7: (1, 1, 0), 8: (1, 1, 1)}
    every = mk.coordinates(two_blocks)
    assert len(every) == mk.summary(two_blocks)['points']
    assert mk.coordinates(two_blocks, 'point', [1, 12]) == \
        {1: every[1], 12: every[12]}


def test_coordinates_are_exact(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (13, 2, 3), (1.1, 0.7, 0.3))
    model.build(geom)
    corner = mk.sel.find(geom, 'point', x=1.1, y=0.7, z=0.3)
    [(number, point)] = mk.coordinates(geom, 'point', corner).items()
    box = mk.bounding_box(geom, 'point', number)
    assert point == (box['x'][0], box['y'][0], box['z'][0])
    flat = mk.geometry(model, 2)
    mk.rectangle(flat, (1, 2))
    model.build(flat)
    assert mk.coordinates(flat, 'point', 1) == {1: (0, 0)}
